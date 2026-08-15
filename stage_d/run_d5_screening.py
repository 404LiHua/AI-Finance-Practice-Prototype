from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BAOSTOCK_SITE = REPO_ROOT / ".venv-baostock/Lib/site-packages"
if str(BAOSTOCK_SITE) not in sys.path:
    sys.path.insert(0, str(BAOSTOCK_SITE))

from experiments.core import DataBundle, prediction_frame, write_json  # noqa: E402
from stage_d.d4_policy import evaluate_frozen_policy  # noqa: E402
from stage_d.freeze_stage_d_candidate import FREEZE_CONFIG, FREEZE_DIR, verify_freeze  # noqa: E402
from stage_d.inference import LoadedStageDFrozenCandidate, sha256_file  # noqa: E402


AUTHORIZATION = REPO_ROOT / "stage_d/authorizations/D5_AUTHORIZATION_20260724.json"
OUTPUT_ROOT = REPO_ROOT / "outputs/stage_d/d5_screening_20240614_20250613"
SEALED_DATA_ROOT = REPO_ROOT / "data/screening/stage_d_d5_20240614_20250613"
EXECUTION_RECEIPT = OUTPUT_ROOT / "D5_EXECUTION_RECEIPT.json"
START_DATE = pd.Timestamp("2024-06-14")
END_DATE = pd.Timestamp("2025-06-13")


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stock_to_baostock(stock_code: str) -> str:
    number, exchange = stock_code.split(".")
    return f"{exchange.lower()}.{number}"


def load_authorization() -> dict[str, Any]:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    scope = authorization["scope"]
    if authorization["authorization_text"] != "授权执行 D-5":
        raise RuntimeError("D-5 explicit authorization text is missing")
    if scope["screening_interval_start"] != START_DATE.date().isoformat():
        raise RuntimeError("authorized D-5 start date mismatch")
    if scope["screening_interval_end"] != END_DATE.date().isoformat():
        raise RuntimeError("authorized D-5 end date mismatch")
    if int(scope["execution_count"]) != 1:
        raise RuntimeError("D-5 authorization is not single-use")
    if EXECUTION_RECEIPT.exists():
        prior = json.loads(EXECUTION_RECEIPT.read_text(encoding="utf-8"))
        raise RuntimeError(f"D-5 single-use authorization already consumed: {prior.get('status')}")
    return authorization


def acquire_baostock_weekly(stock_codes: list[str]) -> pd.DataFrame:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
    rows = []
    fields = "date,code,open,high,low,close,volume,amount,pctChg"
    try:
        for stock_code in stock_codes:
            query = bs.query_history_k_data_plus(
                _stock_to_baostock(stock_code), fields,
                start_date=START_DATE.date().isoformat(),
                end_date=END_DATE.date().isoformat(),
                frequency="w", adjustflag="3",
            )
            if query.error_code != "0":
                raise RuntimeError(
                    f"BaoStock query failed for {stock_code}: {query.error_code} {query.error_msg}"
                )
            stock_rows = []
            while query.next():
                stock_rows.append(query.get_row_data())
            if not stock_rows:
                raise RuntimeError(f"BaoStock returned no authorized rows for {stock_code}")
            frame = pd.DataFrame(stock_rows, columns=query.fields)
            frame.insert(0, "stock_code", stock_code)
            rows.append(frame)
    finally:
        bs.logout()
    raw = pd.concat(rows, ignore_index=True)
    raw["trade_date"] = pd.to_datetime(raw["date"], errors="raise")
    if raw["trade_date"].min() < START_DATE or raw["trade_date"].max() > END_DATE:
        raise RuntimeError("BaoStock returned rows outside the authorized D-SCREENING interval")
    if raw["stock_code"].nunique() != len(stock_codes):
        raise RuntimeError("BaoStock stock coverage differs from frozen universe")
    if raw.duplicated(["stock_code", "trade_date"]).any():
        raise RuntimeError("BaoStock returned duplicate stock-week rows")
    return raw


def build_screening_bundle(raw: pd.DataFrame, stock_codes: list[str]) -> DataBundle:
    panel = raw.copy()
    for column in ("open", "high", "low", "close", "volume", "amount", "pctChg"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    required = ["close", "pctChg"]
    if panel[required].isna().any().any():
        raise RuntimeError("BaoStock authorized weekly data contain missing close or pctChg")
    panel["return_1w"] = panel["pctChg"] / 100.0
    panel["model_close"] = panel["close"]
    panel = panel.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    panel["target_date"] = panel.groupby("stock_code")["trade_date"].shift(-1)
    panel["target_close"] = panel.groupby("stock_code")["close"].shift(-1)
    panel["target_return"] = panel["target_close"] / panel["model_close"] - 1.0
    panel["target_direction"] = np.where(
        panel["target_return"].notna(), (panel["target_return"] > 0).astype(int), np.nan,
    )
    panel["split"] = "screening_context"
    eligible = panel["target_date"].notna() & panel["target_date"].le(END_DATE)
    panel.loc[eligible, "split"] = "screening"
    samples = panel.loc[eligible].copy().reset_index(drop=True)
    if samples["stock_code"].nunique() != len(stock_codes):
        raise RuntimeError("eligible D-SCREENING rows do not cover all frozen stocks")
    if samples.groupby("stock_code").size().min() < 4:
        raise RuntimeError("eligible D-SCREENING rows fail minimum per-stock coverage")
    if not set(samples["stock_code"]).issubset(set(stock_codes)):
        raise RuntimeError("D-SCREENING contains a stock outside the frozen universe")
    return DataBundle(
        panel=panel,
        samples={"screening": samples},
        feature_columns=["return_1w"],
    )


def write_started_receipt(authorization: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    write_json(EXECUTION_RECEIPT, {
        "authorization_id": authorization["authorization_id"],
        "status": "STARTED_AUTHORIZATION_CONSUMED",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_interval": [START_DATE.date().isoformat(), END_DATE.date().isoformat()],
        "runner_sha256": sha256_file(Path(__file__)),
        "candidate": "frets_return_l4__fixed_shrink_a075",
        "execution_count": 1,
    })


def finalize_receipt(status: str, details: dict[str, Any]) -> None:
    current = json.loads(EXECUTION_RECEIPT.read_text(encoding="utf-8"))
    write_json(EXECUTION_RECEIPT, {
        **current,
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    })


def run() -> dict[str, Any]:
    authorization = load_authorization()
    freeze_verification = verify_freeze()
    freeze = json.loads(FREEZE_CONFIG.read_text(encoding="utf-8"))
    d3 = json.loads((REPO_ROOT / "stage_d/configs/d3_diagnostics.json").read_text(encoding="utf-8"))
    stock_codes = [
        value.strip() for value in
        (REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/selected_stocks.txt").read_text(
            encoding="utf-8"
        ).splitlines() if value.strip()
    ]
    if len(stock_codes) != 30:
        raise RuntimeError("frozen stock universe is not exactly 30 stocks")
    write_started_receipt(authorization)
    try:
        raw = acquire_baostock_weekly(stock_codes)
        SEALED_DATA_ROOT.mkdir(parents=True, exist_ok=False)
        source_path = SEALED_DATA_ROOT / "baostock_weekly_authorized.csv.gz"
        raw.to_csv(source_path, index=False, encoding="utf-8-sig", compression="gzip")
        source_manifest = {
            "source": "BaoStock query_history_k_data_plus",
            "frequency": "w",
            "adjustflag": "3_unadjusted",
            "authorized_start": START_DATE.date().isoformat(),
            "authorized_end": END_DATE.date().isoformat(),
            "rows": len(raw),
            "stock_count": raw["stock_code"].nunique(),
            "minimum_date": raw["trade_date"].min().date().isoformat(),
            "maximum_date": raw["trade_date"].max().date().isoformat(),
            "sha256": sha256_file(source_path),
            "authorization_sha256": sha256_file(AUTHORIZATION),
        }
        write_json(SEALED_DATA_ROOT / "SOURCE_MANIFEST.json", source_manifest)

        data = build_screening_bundle(raw, stock_codes)
        loader = LoadedStageDFrozenCandidate(FREEZE_DIR / "INFERENCE_MANIFEST.json", REPO_ROOT)
        candidate, per_seed = loader.predict(data, "screening")
        decision = evaluate_frozen_policy(
            data.samples["screening"], candidate, per_seed, freeze, d3["return_groups"]
        )
        outcome = decision["outcome"]
        if outcome not in freeze["screening_policy"]["allowed_outcomes"]:
            raise RuntimeError(f"frozen policy returned unsupported outcome: {outcome}")

        predictions = prediction_frame(data.samples["screening"], candidate, "D_SCREENING")
        for seed, values in per_seed.items():
            predictions[f"seed_{seed}_prediction"] = values
        predictions["naive_prediction"] = 0.0
        predictions.to_csv(
            OUTPUT_ROOT / "screening_predictions.csv.gz", index=False,
            encoding="utf-8-sig", compression="gzip",
        )
        write_json(OUTPUT_ROOT / "screening_decision.json", {
            "outcome": outcome,
            "candidate": freeze["candidate"]["model_id"],
            "baseline": freeze["baseline"]["model_id"],
            "authorized_interval": [START_DATE.date().isoformat(), END_DATE.date().isoformat()],
            "sample_count": len(data.samples["screening"]),
            "stock_count": data.samples["screening"]["stock_code"].nunique(),
            "candidate_metrics": decision.get("candidate_metrics"),
            "naive_metrics": decision.get("naive_metrics"),
            "diagnostic_values": decision.get("diagnostic_values"),
            "pass_checks": decision.get("pass_checks"),
            "failure_checks": decision.get("failure_checks"),
            "return_group_mae_ratios": decision.get("return_group_mae_ratios"),
            "freeze_verification": freeze_verification,
            "source_manifest": source_manifest,
            "candidate_or_threshold_changes": 0,
            "execution_count": 1,
        })
        evidence = {
            "outcome": outcome,
            "screening_decision_sha256": sha256_file(OUTPUT_ROOT / "screening_decision.json"),
            "screening_predictions_sha256": sha256_file(OUTPUT_ROOT / "screening_predictions.csv.gz"),
            "source_sha256": source_manifest["sha256"],
            "freeze_root_sha256": freeze_verification["manifest_root_sha256"],
            "runner_sha256": sha256_file(Path(__file__)),
            "authorization_sha256": sha256_file(AUTHORIZATION),
        }
        write_json(OUTPUT_ROOT / "SCREENING_EVIDENCE.json", evidence)
        finalize_receipt("COMPLETED_SINGLE_USE", evidence)
        return json.loads((OUTPUT_ROOT / "screening_decision.json").read_text(encoding="utf-8"))
    except Exception as exc:
        finalize_receipt("INVALID_INTEGRITY_FAILURE_SINGLE_USE", {
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
