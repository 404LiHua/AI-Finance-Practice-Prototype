"""Machine acceptance for the frozen E-5.2 three-seed low-cost baseline review."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "outputs/stage_e/e5_low_cost_three_seed_acceptance_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    review = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = resolve(review["base_protocol_config"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    root = resolve(review["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "unified_predictions_three_seed.csv.gz", dtype={"stock_code": str})
    expected = pd.read_csv(root / "frozen_validation_keys.csv.gz", dtype={"stock_code": str})
    receipts = pd.read_csv(root / "engineering_receipts_three_seed.csv")
    failures = json.loads((root / "failure_receipts_three_seed.json").read_text(encoding="utf-8"))
    universe = pd.read_csv(resolve(base["paths"]["universe_path"]), dtype={"stock_code": str})
    interface = json.loads(resolve(base["interface_config"]).read_text(encoding="utf-8"))
    contract = validate_prediction_contract(predictions, expected, review["folds"], review["seeds"])
    recalculated = evaluate_predictions(
        predictions, universe, float(interface["evaluation"]["mape_denominator_floor"]),
        float(interface["evaluation"]["direction_positive_threshold"]),
        int(interface["evaluation"]["return_group_count"]),
    )
    metric_hashes_valid = True
    metrics_finite = True
    for name, frame in recalculated.items():
        stored = pd.read_csv(root / f"{name}.csv")
        metrics_finite = metrics_finite and bool(np.isfinite(stored.select_dtypes(include=[np.number]).to_numpy()).all())
        rendered_hash = __import__("hashlib").sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
        metric_hashes_valid = metric_hashes_valid and rendered_hash == metadata["artifacts"][f"{name}_sha256"]
    payload = dict(metadata)
    declared_batch = payload.pop("batch_sha256", "")
    model_ids = [str(item["id"]) for item in base["models"]]
    expected_runs = len(model_ids) * len(review["folds"]) * len(review["seeds"])
    grid = contract["grid_rows"]
    checks = {
        "review_locked_before_remaining_runs": review["status"] == "PREREGISTERED_LOCKED_BEFORE_REMAINING_SEED_RUNS",
        "base_protocol_hash_unchanged": sha256_file(base_path) == review["base_protocol_config_sha256"] == metadata["base_protocol_config_sha256"],
        "model_set_unchanged": metadata["models"] == model_ids and sorted(contract["models"]) == sorted(model_ids),
        "folds_unchanged": review["folds"] == base["folds"] == metadata["folds"],
        "three_preregistered_seeds": review["seeds"] == base["future_three_seeds"] == metadata["seeds"],
        "only_two_remaining_seeds_newly_trained": review["new_training_seeds"] == [20260723, 20260724] and metadata["new_training_seeds"] == [20260723, 20260724],
        "engineering_seed_reused": review["reused_engineering_seed"] == 20260725 and any(row["seed"] == 20260725 and row["reused"] for row in metadata["source_batches"]),
        "fifty_four_run_receipts": len(receipts) == expected_runs == 54 and metadata["run_count"] == 54,
        "no_failures": not failures and metadata["failure_count"] == 0,
        "all_models_cover_all_4500_keys": len(grid) == len(model_ids) * 3 and all(row["row_count"] == 1500 and row["missing_key_count"] == 0 and row["extra_key_count"] == 0 for row in grid),
        "independent_loading_within_tolerance": bool((receipts["independent_load_max_abs_difference"] <= 1e-7).all()),
        "metrics_finite": metrics_finite,
        "three_seed_pairwise_stability_present": len(recalculated["pairwise_seed_stability"]) == len(model_ids) * 3,
        "independent_metric_recalculation_matches": metric_hashes_valid,
        "artifact_hashes_valid": (
            sha256_file(root / "unified_predictions_three_seed.csv.gz") == metadata["artifacts"]["unified_predictions_three_seed_sha256"]
            and sha256_file(root / "frozen_validation_keys.csv.gz") == metadata["artifacts"]["frozen_validation_keys_sha256"]
            and sha256_file(root / "engineering_receipts_three_seed.csv") == metadata["artifacts"]["engineering_receipts_three_seed_sha256"]
            and sha256_file(root / "failure_receipts_three_seed.json") == metadata["artifacts"]["failure_receipts_three_seed_sha256"]
        ),
        "batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "review_pass_declared": bool(metadata["three_seed_review_pass"]),
        "no_selection_deletion_or_promotion": not metadata["candidate_selection_performed"] and not metadata["model_deletion_performed"],
        "future_and_screening_not_accessed": not metadata["future_or_sealed_data_read"] and not metadata["screening_accessed"],
    }
    report = {
        "stage": "E-5.2 three-seed frozen review acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(checks.values()),
        "checks": checks, "models": model_ids, "folds": review["folds"], "seeds": review["seeds"],
        "selection_exposure": review["selection_exposure"], "review_config_sha256": sha256_file(config_path),
        "base_protocol_config_sha256": sha256_file(base_path), "metadata_sha256": sha256_file(metadata_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
