"""Machine acceptance for the preregistered E-5.2 single-seed engineering run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "outputs/stage_e/e5_low_cost_single_seed_acceptance_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    receipts = pd.read_csv(root / "engineering_receipts.csv")
    failures = json.loads((root / "failure_receipts.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "unified_predictions.csv.gz", dtype={"stock_code": str})
    expected = pd.read_csv(root / "frozen_validation_keys.csv.gz", dtype={"stock_code": str})
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    model_ids = [str(item["id"]) for item in config["models"]]
    seed = int(config["engineering_seed"])
    contract = validate_prediction_contract(predictions, expected, config["folds"], [seed])

    interface = json.loads(resolve(config["interface_config"]).read_text(encoding="utf-8"))
    recalculated = evaluate_predictions(
        predictions, universe,
        float(interface["evaluation"]["mape_denominator_floor"]),
        float(interface["evaluation"]["direction_positive_threshold"]),
        int(interface["evaluation"]["return_group_count"]),
    )
    metric_hashes_valid = True
    metrics_finite = True
    for name, frame in recalculated.items():
        stored_path = root / f"{name}.csv"
        stored = pd.read_csv(stored_path)
        numeric = stored.select_dtypes(include=[np.number]).to_numpy()
        metrics_finite = metrics_finite and bool(np.isfinite(numeric).all())
        temporary = root / f".{name}.acceptance_recalc.csv"
        frame.to_csv(temporary, index=False)
        metric_hashes_valid = metric_hashes_valid and sha256_file(temporary) == metadata["artifacts"][f"{name}_sha256"]
        temporary.unlink()

    payload = dict(metadata)
    declared_batch = payload.pop("batch_sha256", "")
    expected_run_count = len(model_ids) * len(config["folds"])
    grid = contract["grid_rows"]
    rows_per_fold = expected.groupby("fold_id").size().to_dict()
    contract_rows_ok = all(
        row["missing_key_count"] == 0
        and row["extra_key_count"] == 0
        and row["row_count"] == len(expected)
        for row in grid
    )
    receipt_hashes_ok = bool(
        receipts["checkpoint_sha256"].astype(str).map(lambda value: bool(SHA256_PATTERN.fullmatch(value))).all()
        and receipts["resolved_config_sha256"].astype(str).map(lambda value: bool(SHA256_PATTERN.fullmatch(value))).all()
    )
    checks = {
        "config_locked_before_run": config["status"] == "PREREGISTERED_LOCKED_BEFORE_SINGLE_SEED_RUN",
        "model_set_exactly_locked_six": len(model_ids) == 6 and len(set(model_ids)) == 6 and sorted(contract["models"]) == sorted(model_ids),
        "three_folds_one_engineering_seed": config["folds"] == ["E_RO_01", "E_RO_02", "E_RO_03"] and set(predictions["seed"].astype(int)) == {seed},
        "eighteen_run_receipts": len(receipts) == expected_run_count and metadata["run_count"] == expected_run_count,
        "no_failure_receipts": not failures and metadata["failure_count"] == 0,
        "frozen_validation_keys_500_per_fold_1500_total": (
            rows_per_fold == {fold: 500 for fold in config["folds"]} and len(expected) == 1500
        ),
        "prediction_contract_complete": len(grid) == len(model_ids) and contract_rows_ok,
        "independent_loading_within_tolerance": bool((receipts["independent_load_max_abs_difference"] <= 1e-7).all()),
        "checkpoint_and_config_hashes_valid": receipt_hashes_ok,
        "metrics_finite": metrics_finite,
        "independent_metric_recalculation_matches": metric_hashes_valid,
        "artifact_hashes_valid": (
            sha256_file(root / "unified_predictions.csv.gz") == metadata["artifacts"]["unified_predictions_sha256"]
            and sha256_file(root / "frozen_validation_keys.csv.gz") == metadata["artifacts"]["frozen_validation_keys_sha256"]
            and sha256_file(root / "engineering_receipts.csv") == metadata["artifacts"]["engineering_receipts_sha256"]
            and sha256_file(root / "failure_receipts.json") == metadata["artifacts"]["failure_receipts_sha256"]
        ),
        "batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "engineering_pass_declared": bool(metadata["engineering_pass"]),
        "no_model_deletion_or_candidate_selection": not metadata["model_deletion_performed"] and not metadata["candidate_selection_performed"],
        "future_or_sealed_data_not_read": not metadata["future_or_sealed_data_read"],
        "screening_not_accessed": not metadata["screening_accessed"],
    }
    report = {
        "stage": "E-5.2 single-seed engineering acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks,
        "models": model_ids, "folds": config["folds"], "engineering_seed": seed,
        "selection_exposure": config["selection_exposure"],
        "config_sha256": sha256_file(config_path), "metadata_sha256": sha256_file(metadata_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
