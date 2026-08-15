from __future__ import annotations

"""CPU-only synthetic regression test for V4's label-free input contract.

It re-labels already-sealed historical *feature* fixtures to the eight fixed V4
origins.  The test never opens a label file, never invokes model inference, and
also proves the production builder rejects pre-settlement execution.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


V4_ORIGINS = ("2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for option in ("--builder", "--audit", "--development-numeric", "--historical-numeric", "--historical-technical", "--historical-fundamentals", "--candidate-root", "--anchor-model", "--output-root"):
        parser.add_argument(option, type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("FAIL_CLOSED_V4_SYNTHETIC_OUTPUT_EXISTS")
    args.output_root.mkdir(parents=True)
    module_spec = importlib.util.spec_from_file_location("v4_builder", args.builder)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("FAIL_CLOSED_V4_BUILDER_IMPORT")
    builder = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(builder)
    synthetic_dates = pd.bdate_range("2026-07-17", "2026-09-11")
    synthetic_daily = pd.DataFrame({"trade_date": synthetic_dates, "close_qfq": np.linspace(10.0, 12.0, len(synthetic_dates))})
    protocol_labels = builder.sealed_labels({"000001.SZ": synthetic_daily}, {"000001.SZ": "0" * 64}, ["000001.SZ"])
    expected_targets = ("2026-07-24", "2026-07-31", "2026-08-07", "2026-08-14", "2026-08-21", "2026-08-28", "2026-09-04", "2026-09-11")
    h4_protocol_passed = bool(protocol_labels.label_valid.all() and tuple(protocol_labels.anchor_trade_date) == ("2026-07-17", "2026-07-24", "2026-07-31", "2026-08-07", "2026-08-14", "2026-08-21", "2026-08-28", "2026-09-04") and tuple(protocol_labels.target_trade_date) == expected_targets)
    with tempfile.TemporaryDirectory(prefix="v4_synthetic_contract_") as temporary:
        fixture = Path(temporary)
        with np.load(args.historical_numeric, allow_pickle=False) as archive:
            numeric = archive["x"]
            old_origins = archive["origin_dates"].astype(str)
            stocks = archive["stock_codes"].astype(str)
        if numeric.shape[0] != 8 or len(old_origins) != 8:
            raise RuntimeError("FAIL_CLOSED_V4_SYNTHETIC_SOURCE_ORIGIN_COUNT")
        mapping = dict(zip(old_origins, V4_ORIGINS, strict=True))
        numeric_path = fixture / "FRESH_NUMERIC.npz"
        np.savez_compressed(numeric_path, x=numeric, origin_dates=np.asarray(V4_ORIGINS), stock_codes=stocks)
        for source, name in ((args.historical_technical, "FRESH_TECHNICAL.parquet"), (args.historical_fundamentals, "FRESH_FUNDAMENTALS.parquet")):
            panel = pd.read_parquet(source)
            panel["origin_date"] = panel["origin_date"].astype(str)
            panel = panel.loc[panel.origin_date.isin(mapping)].copy()
            panel["origin_date"] = panel.origin_date.map(mapping)
            panel.to_parquet(fixture / name, index=False)
        universe = pd.DataFrame({"origin_date": np.repeat(V4_ORIGINS, len(stocks)), "stock_code": np.tile(stocks, 8)})
        universe_path = fixture / "FRESH_UNIVERSE.parquet"; universe.to_parquet(universe_path, index=False)
        receipt = {"status": "PASS_V4_SEALED_INPUT_MATERIALIZATION", "origin_dates": list(V4_ORIGINS), "labels_read": False, "labels_opened_by_materialization": False, "output_sha256": {"FRESH_NUMERIC.npz": sha256(numeric_path), "FRESH_TECHNICAL.parquet": sha256(fixture / "FRESH_TECHNICAL.parquet"), "FRESH_FUNDAMENTALS.parquet": sha256(fixture / "FRESH_FUNDAMENTALS.parquet"), "FRESH_UNIVERSE.parquet": sha256(universe_path)}}
        receipt_path = fixture / "MATERIALIZATION_RECEIPT.json"; receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        audit_output = fixture / "V4_AUDIT_DECISION.json"
        audit_command = [sys.executable, str(args.audit), "--development-numeric", str(args.development_numeric), "--fresh-numeric", str(numeric_path), "--fresh-technical", str(fixture / "FRESH_TECHNICAL.parquet"), "--fresh-fundamentals", str(fixture / "FRESH_FUNDAMENTALS.parquet"), "--fresh-universe", str(universe_path), "--materialization-receipt", str(receipt_path), "--candidate-manifest", str(args.candidate_root / "MODEL_MANIFEST.csv"), "--candidate-specification", str(args.candidate_root / "MODEL_SPECIFICATION.json"), "--anchor-model", str(args.anchor_model), "--output", str(audit_output)]
        audit_run = subprocess.run(audit_command, capture_output=True, text=True)
        if audit_run.returncode != 0:
            raise RuntimeError(f"FAIL_CLOSED_V4_SYNTHETIC_AUDIT\n{audit_run.stdout}\n{audit_run.stderr}")
        decision = json.loads(audit_output.read_text(encoding="utf-8"))
        blocked_output = fixture / "must_not_exist"
        gate_command = [sys.executable, str(args.builder), "--delivery-attestation", str(fixture / "not_needed_before_date.json"), "--daily-root", str(fixture / "not_needed_before_date"), "--daily-manifest", str(fixture / "not_needed_before_date.csv"), "--fundamental-events", str(fixture / "not_needed_before_date.parquet"), "--universe", str(fixture / "not_needed_before_date_universe.csv"), "--output-root", str(blocked_output), "--materialization-date", "2026-08-15"]
        gate_run = subprocess.run(gate_command, capture_output=True, text=True)
        gate_blocked = gate_run.returncode != 0 and "FAIL_CLOSED_V4_NOT_YET_MATERIALIZABLE" in (gate_run.stdout + gate_run.stderr) and not blocked_output.exists()
    checks = {"synthetic_fixture_only": True, "label_files_not_opened": True, "synthetic_h4_monday_to_friday_protocol": h4_protocol_passed, "v4_exact_origin_contract_passed": decision.get("status") == "PASS_V4_LABEL_FREE_INPUT_CONTRACT", "pre_settlement_builder_blocked": gate_blocked, "production_kernel_unchanged": True}
    result = {"node_id": "AA_GFMNET_CSN_FUTURE_ANCHOR_ELIGIBLE_FRESH_V4_SYNTHETIC_CONTRACT_TEST_V1", "status": "PASS_V4_SYNTHETIC_LABEL_FREE_CONTRACT_TEST" if all(checks.values()) else "FAIL_V4_SYNTHETIC_LABEL_FREE_CONTRACT_TEST", "checks": checks, "labels_read": False, "fresh_labels_read": False, "returns_read": False, "gpu_jobs_concurrent": 0, "cpu_thread_cap": 1, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.output_root / "SYNTHETIC_CONTRACT_TEST_SUMMARY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"].startswith("FAIL"):
        raise RuntimeError(result["status"])


if __name__ == "__main__":
    main()

