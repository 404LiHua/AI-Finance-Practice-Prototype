from __future__ import annotations

"""V2 label-free, fail-closed contract audit for V4 pre-registered inputs.

The interface intentionally accepts no label path.  It validates only feature-side
artifacts, the materialization receipt, candidate identity, and anchor chronology.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ORIGINS = ("2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07")
EXPECTED_MANIFEST_SHA = "c7ee9368bf0e71eb21efc3f4de05b86f0a68335008891afc401e8a7b0fa6908e"
EXPECTED_SPEC_SHA = "49b4e6d6c441bb44e94949da35a86d4bc2fb2324c8952394b99bb4c5a6786741"
TECH = ("momentum_20d", "momentum_60d", "momentum_120d", "realized_volatility_20d", "realized_volatility_60d", "downside_volatility_60d", "current_drawdown_60d", "rsi_14", "macd_scaled", "bollinger_position_20", "amihud_20d", "volume_ratio_20d_60d", "intraday_range_mean_20d", "technical_available")
FUND = ("log_total_assets", "debt_to_assets", "equity_to_assets", "return_on_assets", "net_margin", "asset_turnover", "revenue_yoy", "profit_yoy", "asset_growth_yoy", "leverage_change_yoy", "report_age_anchor_days", "has_fundamental_event")
KEYS = ["origin_date", "stock_code"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_contract(path: Path, require_v4_origins: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not {"x", "origin_dates", "stock_codes"}.issubset(archive.files):
            raise RuntimeError("FAIL_CLOSED_V4_NUMERIC_SCHEMA")
        numeric = archive["x"]
        origins = archive["origin_dates"].astype(str)
        stocks = archive["stock_codes"].astype(str)
    if numeric.ndim != 4 or numeric.shape[0] != len(origins) or numeric.shape[1] != 8 or numeric.shape[2] != len(stocks) or numeric.shape[3] != 6:
        raise RuntimeError("FAIL_CLOSED_V4_NUMERIC_SHAPE")
    if len(origins) != len(set(origins)) or len(stocks) != len(set(stocks)) or not 200 <= len(stocks) <= 300:
        raise RuntimeError("FAIL_CLOSED_V4_NUMERIC_KEY_DOMAIN")
    parsed = pd.to_datetime(origins, errors="coerce")
    if parsed.isna().any() or not parsed.is_monotonic_increasing:
        raise RuntimeError("FAIL_CLOSED_V4_NUMERIC_ORIGIN_ORDER")
    if require_v4_origins and (tuple(origins) != ORIGINS or not bool((parsed.dayofweek == 0).all()) or len(set(parsed.to_period("W-SUN"))) != 8):
        raise RuntimeError("FAIL_CLOSED_V4_ORIGIN_SEMANTICS")
    if not np.isfinite(numeric).any():
        raise RuntimeError("FAIL_CLOSED_V4_NUMERIC_ALL_MISSING")
    return numeric, origins, stocks


def panel_contract(path: Path, origins: np.ndarray, stocks: np.ndarray, fields: tuple[str, ...], label: str) -> None:
    try:
        panel = pd.read_parquet(path, columns=[*KEYS, *fields])
    except Exception as error:
        raise RuntimeError(f"FAIL_CLOSED_V4_{label}_SCHEMA") from error
    panel.origin_date = pd.to_datetime(panel.origin_date, errors="coerce").dt.date.astype(str)
    panel.stock_code = panel.stock_code.astype(str).str.upper()
    expected = pd.MultiIndex.from_product([origins, stocks], names=KEYS)
    actual = pd.MultiIndex.from_frame(panel[KEYS])
    if panel.origin_date.isna().any() or panel.duplicated(KEYS).any() or not actual.sort_values().equals(expected.sort_values()):
        raise RuntimeError(f"FAIL_CLOSED_V4_{label}_KEY_DOMAIN")


def universe_contract(path: Path, origins: np.ndarray, stocks: np.ndarray) -> None:
    try:
        universe = pd.read_parquet(path, columns=KEYS)
    except Exception as error:
        raise RuntimeError("FAIL_CLOSED_V4_UNIVERSE_SCHEMA") from error
    universe.origin_date = pd.to_datetime(universe.origin_date, errors="coerce").dt.date.astype(str)
    universe.stock_code = universe.stock_code.astype(str).str.upper()
    expected = pd.MultiIndex.from_product([origins, stocks], names=KEYS)
    actual = pd.MultiIndex.from_frame(universe[KEYS])
    if universe.origin_date.isna().any() or universe.duplicated(KEYS).any() or not actual.sort_values().equals(expected.sort_values()):
        raise RuntimeError("FAIL_CLOSED_V4_UNIVERSE_KEY_DOMAIN")


def main() -> None:
    parser = argparse.ArgumentParser()
    for option in ("--development-numeric", "--fresh-numeric", "--fresh-technical", "--fresh-fundamentals", "--fresh-universe", "--materialization-receipt", "--candidate-manifest", "--candidate-specification", "--anchor-model", "--output"):
        parser.add_argument(option, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("FAIL_CLOSED_V4_AUDIT_OUTPUT_EXISTS")
    _, development_origins, _ = numeric_contract(args.development_numeric, require_v4_origins=False)
    numeric, origins, stocks = numeric_contract(args.fresh_numeric, require_v4_origins=True)
    if str(development_origins[-1]) >= ORIGINS[0] or set(development_origins).intersection(ORIGINS):
        raise RuntimeError("FAIL_CLOSED_V4_CANDIDATE_TEMPORAL_BOUNDARY")
    panel_contract(args.fresh_technical, origins, stocks, TECH, "TECHNICAL")
    panel_contract(args.fresh_fundamentals, origins, stocks, FUND, "FUNDAMENTALS")
    universe_contract(args.fresh_universe, origins, stocks)
    receipt = json.loads(args.materialization_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_V4_SEALED_INPUT_MATERIALIZATION" or receipt.get("origin_dates") != list(ORIGINS) or receipt.get("labels_read") is not False or receipt.get("labels_opened_by_materialization") is not False:
        raise RuntimeError("FAIL_CLOSED_V4_MATERIALIZATION_RECEIPT")
    bound_hashes = receipt.get("output_sha256", {})
    for output_name, source in {"FRESH_NUMERIC.npz": args.fresh_numeric, "FRESH_TECHNICAL.parquet": args.fresh_technical, "FRESH_FUNDAMENTALS.parquet": args.fresh_fundamentals, "FRESH_UNIVERSE.parquet": args.fresh_universe}.items():
        if bound_hashes.get(output_name) != sha256(source):
            raise RuntimeError("FAIL_CLOSED_V4_MATERIALIZATION_HASH_BINDING")
    if sha256(args.candidate_manifest) != EXPECTED_MANIFEST_SHA or sha256(args.candidate_specification) != EXPECTED_SPEC_SHA:
        raise RuntimeError("FAIL_CLOSED_V4_CANDIDATE_IDENTITY")
    specification = json.loads(args.candidate_specification.read_text(encoding="utf-8"))
    if specification.get("candidate_id") != "AA_GFMNET_CROSS_SECTIONAL_NEUTRALIZED_RESIDUAL_TCN_V1":
        raise RuntimeError("FAIL_CLOSED_V4_CANDIDATE_SPECIFICATION")
    anchor = json.loads(args.anchor_model.read_text(encoding="utf-8"))
    anchor_last = str(anchor.get("fit_receipt", {}).get("last_origin_date", ""))
    if anchor.get("model_id") != "RG_OBGNET_CONFIRMED_SAFE_V1_1" or anchor_last >= ORIGINS[0]:
        raise RuntimeError("FAIL_CLOSED_V4_ANCHOR_TEMPORAL_BOUNDARY")
    decision = {
        "node_id": "AA_GFMNET_CSN_FUTURE_ANCHOR_ELIGIBLE_FRESH_V4_LABEL_FREE_INPUT_CONTRACT_AUDIT_V2",
        "status": "PASS_V4_LABEL_FREE_INPUT_CONTRACT",
        "origin_dates": list(ORIGINS),
        "origin_semantics": "exactly_eight_pre_registered_monday_0930_weekly_unique",
        "stock_count": int(len(stocks)),
        "numeric_shape": list(numeric.shape),
        "development_last_origin": str(development_origins[-1]),
        "anchor_last_origin": anchor_last,
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "candidate_specification_sha256": sha256(args.candidate_specification),
        "anchor_model_sha256": sha256(args.anchor_model),
        "input_sha256": {
            "FRESH_NUMERIC.npz": sha256(args.fresh_numeric),
            "FRESH_TECHNICAL.parquet": sha256(args.fresh_technical),
            "FRESH_FUNDAMENTALS.parquet": sha256(args.fresh_fundamentals),
            "FRESH_UNIVERSE.parquet": sha256(args.fresh_universe),
            "MATERIALIZATION_RECEIPT.json": sha256(args.materialization_receipt),
        },
        "labels_read": False,
        "fresh_labels_read": False,
        "returns_read": False,
        "test_split_read": False,
        "production_kernel_modified": False,
        "gpu_jobs_concurrent": 0,
        "cpu_thread_cap": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
