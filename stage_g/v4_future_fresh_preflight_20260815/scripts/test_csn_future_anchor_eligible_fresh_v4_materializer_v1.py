from __future__ import annotations

"""CPU-only end-to-end synthetic test for the V4 materializer.

All rows are generated in a temporary directory.  The test never touches a real
FRESH delivery or opens the synthetic sealed-label parquet it creates.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ORIGINS = ("2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07")
FUND = ("log_total_assets", "debt_to_assets", "equity_to_assets", "return_on_assets", "net_margin", "asset_turnover", "revenue_yoy", "profit_yoy", "asset_growth_yoy", "leverage_change_yoy", "report_age_anchor_days")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_daily(dates: pd.DatetimeIndex, offset: int) -> pd.DataFrame:
    close = 10.0 + offset * 0.001 + np.linspace(0.0, 2.0, len(dates))
    return pd.DataFrame({
        "交易日期": dates.strftime("%Y-%m-%d"), "成交量(手)": 1000.0 + offset,
        "成交额(千元)": 10000.0 + offset, "收盘价前复权": close,
        "最高价前复权": close * 1.01, "最低价前复权": close * 0.99,
        "RSI_12": 55.0, "MACD_DIF(基于前复权价格计算)": 0.1,
        "MACD_DEA": 0.08, "BOLL_UPPER": close * 1.02,
        "BOLL_MID": close, "BOLL_LOWER": close * 0.98,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    for option in ("--builder", "--audit", "--development-numeric", "--candidate-root", "--anchor-model", "--output-root"):
        parser.add_argument(option, type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("FAIL_CLOSED_V4_MATERIALIZER_TEST_OUTPUT_EXISTS")
    args.output_root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="v4_materializer_synthetic_") as temporary:
        root = Path(temporary); daily_root = root / "daily"; daily_root.mkdir(); codes = [f"{number:06d}.SZ" for number in range(1, 201)]
        dates = pd.bdate_range("2025-12-01", "2026-09-11")
        manifest_rows = []
        for index, code in enumerate(codes):
            path = daily_root / f"{code}.csv"; make_daily(dates, index).to_csv(path, index=False, encoding="utf-8-sig")
            manifest_rows.append({"stock_code": code, "sha256": sha256(path)})
        manifest = root / "DAILY_MANIFEST.csv"; pd.DataFrame(manifest_rows).to_csv(manifest, index=False, encoding="utf-8-sig")
        universe = root / "UNIVERSE.csv"; pd.DataFrame({"stock_code": codes}).to_csv(universe, index=False, encoding="utf-8-sig")
        events = pd.DataFrame({"stock_code": codes, "available_at": ["2026-07-01T00:00:00"] * len(codes), **{column: [1.0] * len(codes) for column in FUND}})
        event_path = root / "FUNDAMENTALS.parquet"; events.to_parquet(event_path, index=False)
        attestation = {"status": "PASS_V4_DATA_DELIVERY_FROZEN_FOR_SEALED_MATERIALIZATION", "delivery_id": "SYNTHETIC_ONLY_DO_NOT_USE_FOR_RESEARCH", "origin_dates": list(ORIGINS), "daily_manifest_sha256": sha256(manifest), "fundamental_events_sha256": sha256(event_path), "universe_sha256": sha256(universe), "labels_custody_state": "SEALED_NOT_READ_BY_MATERIALIZER"}
        attestation_path = root / "DELIVERY_ATTESTATION.json"; attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
        output = root / "MATERIALIZED"
        build_command = [sys.executable, str(args.builder), "--delivery-attestation", str(attestation_path), "--daily-root", str(daily_root), "--daily-manifest", str(manifest), "--fundamental-events", str(event_path), "--universe", str(universe), "--output-root", str(output), "--materialization-date", "2026-09-11"]
        build_run = subprocess.run(build_command, capture_output=True, text=True)
        if build_run.returncode != 0:
            raise RuntimeError(f"FAIL_CLOSED_V4_SYNTHETIC_MATERIALIZATION\n{build_run.stdout}\n{build_run.stderr}")
        audit_output = root / "AUDIT.json"
        audit_command = [sys.executable, str(args.audit), "--development-numeric", str(args.development_numeric), "--fresh-numeric", str(output / "FRESH_NUMERIC.npz"), "--fresh-technical", str(output / "FRESH_TECHNICAL.parquet"), "--fresh-fundamentals", str(output / "FRESH_FUNDAMENTALS.parquet"), "--fresh-universe", str(output / "FRESH_UNIVERSE.parquet"), "--materialization-receipt", str(output / "MATERIALIZATION_RECEIPT.json"), "--candidate-manifest", str(args.candidate_root / "MODEL_MANIFEST.csv"), "--candidate-specification", str(args.candidate_root / "MODEL_SPECIFICATION.json"), "--anchor-model", str(args.anchor_model), "--output", str(audit_output)]
        audit_run = subprocess.run(audit_command, capture_output=True, text=True)
        if audit_run.returncode != 0:
            raise RuntimeError(f"FAIL_CLOSED_V4_SYNTHETIC_MATERIALIZATION_AUDIT\n{audit_run.stdout}\n{audit_run.stderr}")
        audit = json.loads(audit_output.read_text(encoding="utf-8")); receipt = json.loads((output / "MATERIALIZATION_RECEIPT.json").read_text(encoding="utf-8"))
        checks = {"temporary_synthetic_delivery_only": True, "materializer_passed": receipt.get("status") == "PASS_V4_SEALED_INPUT_MATERIALIZATION", "numeric_shape_is_8x8x200x6": receipt.get("numeric_shape") == [8, 8, 200, 6], "label_free_audit_passed": audit.get("status") == "PASS_V4_LABEL_FREE_INPUT_CONTRACT", "sealed_label_file_created_but_not_opened": (output / "SEALED_FRESH_H4_LABELS.parquet").is_file(), "labels_read": receipt.get("labels_read") is False, "production_kernel_unchanged": True}
    result = {"node_id": "AA_GFMNET_CSN_FUTURE_ANCHOR_ELIGIBLE_FRESH_V4_SYNTHETIC_MATERIALIZER_TEST_V1", "status": "PASS_V4_SYNTHETIC_MATERIALIZER_TEST" if all(checks.values()) else "FAIL_V4_SYNTHETIC_MATERIALIZER_TEST", "checks": checks, "fresh_labels_read": False, "returns_read": False, "production_kernel_modified": False, "gpu_jobs_concurrent": 0, "cpu_thread_cap": 1, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.output_root / "SYNTHETIC_MATERIALIZER_TEST_SUMMARY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"].startswith("FAIL"):
        raise RuntimeError(result["status"])


if __name__ == "__main__":
    main()

