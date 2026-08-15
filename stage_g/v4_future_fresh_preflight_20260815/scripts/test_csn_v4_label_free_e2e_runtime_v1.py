from __future__ import annotations

"""Synthetic, label-free end-to-end runtime smoke test for the V4 predictors.

The test creates temporary PIT-shaped numeric/technical/fundamental inputs and a
small daily-price source, then runs the frozen candidate on CUDA followed by the
production anchor on bounded CPU workers.  It never creates or opens a label file.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


ORIGINS = (
    "2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10",
    "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07",
)
TECHNICAL = (
    "momentum_20d", "momentum_60d", "momentum_120d",
    "realized_volatility_20d", "realized_volatility_60d",
    "downside_volatility_60d", "current_drawdown_60d", "rsi_14",
    "macd_scaled", "bollinger_position_20", "amihud_20d",
    "volume_ratio_20d_60d", "intraday_range_mean_20d", "technical_available",
)
FUNDAMENTALS = (
    "log_total_assets", "debt_to_assets", "equity_to_assets", "return_on_assets",
    "net_margin", "asset_turnover", "revenue_yoy", "profit_yoy", "asset_growth_yoy",
    "leverage_change_yoy", "report_age_anchor_days", "has_fundamental_event",
)


def run_checked(command: list[str], label: str) -> tuple[dict, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"FAIL_CLOSED_{label}\n{completed.stdout}\n{completed.stderr}")
    try:
        return json.loads(completed.stdout), elapsed
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FAIL_CLOSED_{label}_RECEIPT_OUTPUT") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_stocks(count: int) -> tuple[str, ...]:
    return tuple(f"{number:06d}.SZ" for number in range(1, count + 1))


def make_daily_source(root: Path, stocks: tuple[str, ...]) -> Path:
    daily = root / "daily"
    daily.mkdir()
    dates = pd.bdate_range("2025-10-01", "2026-09-07")
    for stock_index, stock in enumerate(stocks):
        close = 20.0 + stock_index + np.linspace(0.0, 2.0, len(dates))
        close += 0.1 * np.sin(np.arange(len(dates)) / 5.0)
        rows = []
        for index, (date, price) in enumerate(zip(dates, close)):
            volume = 1_000_000 + stock_index * 100_000 + index * 10
            rows.append([
                f"row{index}", date.strftime("%Y-%m-%d"), float(price), "x",
                float(price * 1.01), float(price * 0.99), "x", "x", "x",
                float(volume), float(price * volume), 1.0,
            ])
        pd.DataFrame(rows).to_csv(daily / f"{stock}.csv", index=False, header=False)
    return daily


def make_candidate_inputs(root: Path, stocks: tuple[str, ...]) -> tuple[Path, Path, Path, Path]:
    universe = pd.DataFrame({
        "origin_date": np.repeat(ORIGINS, len(stocks)),
        "stock_code": np.tile(stocks, len(ORIGINS)),
    })
    universe_path = root / "FRESH_UNIVERSE.parquet"
    universe.to_parquet(universe_path, index=False)

    numeric = np.zeros((len(ORIGINS), 8, len(stocks), 6), dtype=np.float32)
    for origin_index in range(len(ORIGINS)):
        for stock_index in range(len(stocks)):
            numeric[origin_index, :, stock_index, :] = (
                0.01 * (origin_index + 1)
                + 0.001 * (stock_index + 1)
                + np.arange(48, dtype=np.float32).reshape(8, 6) * 0.0001
            )
    numeric_path = root / "FRESH_NUMERIC.npz"
    np.savez(numeric_path, x=numeric, origin_dates=np.asarray(ORIGINS), stock_codes=np.asarray(stocks))

    rows = []
    for origin_index, origin in enumerate(ORIGINS):
        for stock_index, stock in enumerate(stocks):
            row = {"origin_date": origin, "stock_code": stock}
            row.update({name: float((origin_index + 1) * 0.01 + (stock_index + 1) * 0.001) for name in TECHNICAL})
            row.update({name: float((origin_index + 1) * 0.02 + (stock_index + 1) * 0.002) for name in FUNDAMENTALS})
            rows.append(row)
    table = pd.DataFrame(rows)
    technical_path = root / "FRESH_TECHNICAL.parquet"
    fundamentals_path = root / "FRESH_FUNDAMENTALS.parquet"
    table[["origin_date", "stock_code", *TECHNICAL]].to_parquet(technical_path, index=False)
    table[["origin_date", "stock_code", *FUNDAMENTALS]].to_parquet(fundamentals_path, index=False)
    return numeric_path, technical_path, fundamentals_path, universe_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--anchor-model", type=Path, required=True)
    parser.add_argument("--candidate-predictor", type=Path, required=True)
    parser.add_argument("--anchor-predictor", type=Path, required=True)
    parser.add_argument("--binding-audit", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--stock-count", type=int, default=2, choices=range(2, 301))
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("FAIL_CLOSED_E2E_OUTPUT_EXISTS")
    required = (args.candidate_predictor, args.anchor_predictor, args.anchor_model)
    if args.binding_audit is not None:
        required = (*required, args.binding_audit)
        if args.stock_count < 200:
            raise ValueError("FAIL_CLOSED_BINDING_SMOKE_REQUIRES_200_TO_300_STOCKS")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("FAIL_CLOSED_E2E_RUNTIME_INPUT")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("FAIL_CLOSED_CUDA_UNAVAILABLE")

    with tempfile.TemporaryDirectory(prefix="csn_v4_label_free_e2e_") as temporary:
        root = Path(temporary)
        stocks = make_stocks(args.stock_count)
        daily = make_daily_source(root, stocks)
        numeric, technical, fundamentals, universe = make_candidate_inputs(root, stocks)
        candidate_output = root / "CANDIDATE.parquet"
        candidate_receipt = root / "CANDIDATE_RECEIPT.json"
        anchor_output = root / "ANCHOR.parquet"
        anchor_receipt = root / "ANCHOR_RECEIPT.json"

        candidate_command = [
            sys.executable, str(args.candidate_predictor),
            "--candidate-root", str(args.candidate_root), "--numeric", str(numeric),
            "--technical", str(technical), "--fundamentals", str(fundamentals),
            "--output", str(candidate_output), "--receipt", str(candidate_receipt),
            "--device", args.device, "--batch-size", "1024",
        ]
        candidate_payload, candidate_seconds = run_checked(candidate_command, "CANDIDATE_E2E_PREDICTION")

        anchor_command = [
            sys.executable, str(args.anchor_predictor), "--model", str(args.anchor_model),
            "--source-root", str(daily), "--universe", str(universe),
            "--output", str(anchor_output), "--receipt", str(anchor_receipt), "--workers", "1",
        ]
        anchor_payload, anchor_seconds = run_checked(anchor_command, "ANCHOR_E2E_PREDICTION")

        binding_payload = None
        if args.binding_audit is not None:
            input_hashes = {
                "FRESH_NUMERIC.npz": sha256(numeric),
                "FRESH_TECHNICAL.parquet": sha256(technical),
                "FRESH_FUNDAMENTALS.parquet": sha256(fundamentals),
                "FRESH_UNIVERSE.parquet": sha256(universe),
                "SEALED_FRESH_H4_LABELS.parquet": "0" * 64,
            }
            materialization = {
                "status": "PASS_V4_SEALED_INPUT_MATERIALIZATION", "origin_dates": list(ORIGINS),
                "labels_read": False, "labels_opened_by_materialization": False, "output_sha256": input_hashes,
            }
            materialization_path = root / "MATERIALIZATION_RECEIPT.json"
            materialization_path.write_text(json.dumps(materialization), encoding="utf-8")
            contract = {
                "status": "PASS_V4_LABEL_FREE_INPUT_CONTRACT", "origin_dates": list(ORIGINS),
                "input_sha256": {
                    **{name: input_hashes[name] for name in input_hashes if name != "SEALED_FRESH_H4_LABELS.parquet"},
                    "MATERIALIZATION_RECEIPT.json": sha256(materialization_path),
                },
            }
            contract_path = root / "V4_INPUT_CONTRACT.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            binding_output = root / "BINDING_RECEIPT.json"
            binding_command = [
                sys.executable, str(args.binding_audit), "--candidate-predictions", str(candidate_output),
                "--anchor-predictions", str(anchor_output), "--candidate-receipt", str(candidate_receipt),
                "--anchor-receipt", str(anchor_receipt), "--v4-input-contract", str(contract_path),
                "--materialization-receipt", str(materialization_path), "--universe", str(universe),
                "--candidate-manifest", str(args.candidate_root / "MODEL_MANIFEST.csv"),
                "--candidate-specification", str(args.candidate_root / "MODEL_SPECIFICATION.json"),
                "--anchor-model", str(args.anchor_model), "--output", str(binding_output),
            ]
            binding_payload, _ = run_checked(binding_command, "E2E_BINDING_AUDIT")

        candidate = pd.read_parquet(candidate_output)
        anchor = pd.read_parquet(anchor_output)
        expected_rows = len(ORIGINS) * len(stocks)
        keys = ["origin_date", "stock_code"]
        if len(candidate) != expected_rows or len(anchor) != expected_rows:
            raise RuntimeError("FAIL_CLOSED_E2E_ROW_COUNT")
        if candidate.duplicated(keys).any() or anchor.duplicated(keys).any():
            raise RuntimeError("FAIL_CLOSED_E2E_KEY_DOMAIN")
        if not np.allclose(candidate[["p_down", "p_neutral", "p_up"]].sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("FAIL_CLOSED_E2E_CANDIDATE_PROBABILITY")
        if not np.allclose(anchor[["p_down", "p_neutral", "p_up"]].sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("FAIL_CLOSED_E2E_ANCHOR_PROBABILITY")
        if candidate_payload.get("gpu_jobs_concurrent") != 1 or candidate_payload.get("labels_read"):
            raise RuntimeError("FAIL_CLOSED_E2E_CANDIDATE_RESOURCE_RECEIPT")
        if anchor_payload.get("workers") != 1 or anchor_payload.get("labels_read"):
            raise RuntimeError("FAIL_CLOSED_E2E_ANCHOR_RESOURCE_RECEIPT")
        if binding_payload is not None and binding_payload.get("status") != "PASS_V4_PRECONSUMPTION_PREDICTION_BINDING_READY_FOR_CUSTODIAN":
            raise RuntimeError("FAIL_CLOSED_E2E_BINDING_RECEIPT")
        if any("label" in path.name.lower() for path in root.rglob("*")):
            raise RuntimeError("FAIL_CLOSED_E2E_LABEL_ARTIFACT")

    args.output_root.mkdir(parents=True)
    summary = {
        "node_id": "AA_GFMNET_CSN_V4_LABEL_FREE_E2E_RUNTIME_SMOKE_TEST_V1",
        "status": "PASS_LABEL_FREE_E2E_SYNTHETIC_CANDIDATE_GPU_ANCHOR_CPU",
        "origins": list(ORIGINS), "stock_count": len(stocks), "rows_per_prediction": expected_rows,
        "candidate_device": candidate_payload.get("device"),
        "candidate_gpu_jobs_concurrent": candidate_payload.get("gpu_jobs_concurrent"),
        "candidate_cpu_thread_cap": candidate_payload.get("cpu_thread_cap"),
        "candidate_seconds": round(candidate_seconds, 3),
        "anchor_workers": anchor_payload.get("workers"), "anchor_seconds": round(anchor_seconds, 3),
        "binding_audit_run": binding_payload is not None,
        "binding_status": binding_payload.get("status") if binding_payload is not None else None,
        "labels_created_or_read": False, "returns_read": False,
        "production_kernel_modified": False,
    }
    (args.output_root / "E2E_RUNTIME_SMOKE_TEST_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
