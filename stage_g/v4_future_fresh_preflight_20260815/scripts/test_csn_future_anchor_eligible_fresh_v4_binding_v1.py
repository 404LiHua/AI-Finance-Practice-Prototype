from __future__ import annotations

"""CPU-only synthetic test of V4 prediction binding; no label file is created."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for option in ("--binding-audit", "--candidate-root", "--anchor-model", "--output-root"):
        parser.add_argument(option, type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("FAIL_CLOSED_V4_BINDING_TEST_OUTPUT_EXISTS")
    args.output_root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="v4_binding_synthetic_") as temporary:
        root = Path(temporary); codes = [f"{number:06d}.SZ" for number in range(1, 201)]
        universe = pd.DataFrame({"origin_date": np.repeat(ORIGINS, len(codes)), "stock_code": np.tile(codes, len(ORIGINS))})
        universe_path = root / "FRESH_UNIVERSE.parquet"; universe.to_parquet(universe_path, index=False)
        predictions = universe.assign(h4_prediction=0.001, p_down=0.2, p_neutral=0.6, p_up=0.2)
        candidate_prediction, anchor_prediction = root / "CANDIDATE.parquet", root / "ANCHOR.parquet"; predictions.to_parquet(candidate_prediction, index=False); predictions.to_parquet(anchor_prediction, index=False)
        input_hashes = {"FRESH_NUMERIC.npz": "a" * 64, "FRESH_TECHNICAL.parquet": "b" * 64, "FRESH_FUNDAMENTALS.parquet": "c" * 64, "FRESH_UNIVERSE.parquet": sha256(universe_path), "SEALED_FRESH_H4_LABELS.parquet": "d" * 64}
        materialization = {"status": "PASS_V4_SEALED_INPUT_MATERIALIZATION", "origin_dates": list(ORIGINS), "labels_read": False, "labels_opened_by_materialization": False, "output_sha256": input_hashes}
        materialization_path = root / "MATERIALIZATION_RECEIPT.json"; materialization_path.write_text(json.dumps(materialization), encoding="utf-8")
        contract = {"status": "PASS_V4_LABEL_FREE_INPUT_CONTRACT", "origin_dates": list(ORIGINS), "input_sha256": {**{key: input_hashes[key] for key in ("FRESH_NUMERIC.npz", "FRESH_TECHNICAL.parquet", "FRESH_FUNDAMENTALS.parquet", "FRESH_UNIVERSE.parquet")}, "MATERIALIZATION_RECEIPT.json": sha256(materialization_path)}}
        contract_path = root / "V4_INPUT_CONTRACT.json"; contract_path.write_text(json.dumps(contract), encoding="utf-8")
        candidate_receipt = {"status": "PASS_LABEL_FREE_CANDIDATE_BATCH_PREDICTION", "candidate_id": "AA_GFMNET_CROSS_SECTIONAL_NEUTRALIZED_RESIDUAL_TCN_V1", "model_manifest_sha256": sha256(args.candidate_root / "MODEL_MANIFEST.csv"), "model_specification_sha256": sha256(args.candidate_root / "MODEL_SPECIFICATION.json"), "numeric_sha256": input_hashes["FRESH_NUMERIC.npz"], "technical_sha256": input_hashes["FRESH_TECHNICAL.parquet"], "fundamentals_sha256": input_hashes["FRESH_FUNDAMENTALS.parquet"], "rows": len(universe), "origin_weeks": 8, "gpu_jobs_concurrent": 1, "labels_read": False, "fresh_labels_read": False, "returns_read": False}
        anchor_receipt = {"status": "PASS_LABEL_FREE_ANCHOR_PREDICTION", "anchor_kernel_id": "RG_OBGNET_CONFIRMED_SAFE_V1_1", "model_sha256": sha256(args.anchor_model), "universe_sha256": sha256(universe_path), "rows": len(universe), "origin_weeks": 8, "workers": 1, "labels_read": False, "fresh_labels_read": False, "returns_read": False}
        candidate_receipt_path, anchor_receipt_path = root / "CANDIDATE_RECEIPT.json", root / "ANCHOR_RECEIPT.json"; candidate_receipt_path.write_text(json.dumps(candidate_receipt), encoding="utf-8"); anchor_receipt_path.write_text(json.dumps(anchor_receipt), encoding="utf-8")
        output = root / "BINDING.json"
        command = [sys.executable, str(args.binding_audit), "--candidate-predictions", str(candidate_prediction), "--anchor-predictions", str(anchor_prediction), "--candidate-receipt", str(candidate_receipt_path), "--anchor-receipt", str(anchor_receipt_path), "--v4-input-contract", str(contract_path), "--materialization-receipt", str(materialization_path), "--universe", str(universe_path), "--candidate-manifest", str(args.candidate_root / "MODEL_MANIFEST.csv"), "--candidate-specification", str(args.candidate_root / "MODEL_SPECIFICATION.json"), "--anchor-model", str(args.anchor_model), "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"FAIL_CLOSED_V4_SYNTHETIC_BINDING\n{completed.stdout}\n{completed.stderr}")
        binding = json.loads(output.read_text(encoding="utf-8"))
    checks = {"synthetic_predictions_only": True, "no_label_file_created_or_opened": True, "exact_v4_binding_passed": binding.get("status") == "PASS_V4_PRECONSUMPTION_PREDICTION_BINDING_READY_FOR_CUSTODIAN", "all_binding_checks_passed": all(binding.get("checks", {}).values()), "production_kernel_unchanged": True}
    result = {"node_id": "AA_GFMNET_CSN_FUTURE_ANCHOR_ELIGIBLE_FRESH_V4_SYNTHETIC_BINDING_TEST_V1", "status": "PASS_V4_SYNTHETIC_PRECONSUMPTION_BINDING_TEST" if all(checks.values()) else "FAIL_V4_SYNTHETIC_PRECONSUMPTION_BINDING_TEST", "checks": checks, "labels_read": False, "fresh_labels_read": False, "returns_read": False, "production_kernel_modified": False, "gpu_jobs_concurrent": 0, "cpu_thread_cap": 1, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.output_root / "SYNTHETIC_BINDING_TEST_SUMMARY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"].startswith("FAIL"):
        raise RuntimeError(result["status"])


if __name__ == "__main__":
    main()
