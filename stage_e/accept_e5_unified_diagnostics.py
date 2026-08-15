"""Machine acceptance for the frozen E-5.4 ten-model unified diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.e5.diagnostics import (
    engineering_cost_summary, fold_and_worst_fold, model_disagreement,
    overall_pooled_metrics, seed_prediction_dispersion,
)
from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "outputs/stage_e/e5_unified_diagnostics_acceptance_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "unified_predictions_ten_models.csv.gz", dtype={"stock_code": str})
    receipts = pd.read_csv(root / "engineering_receipts_ten_models.csv")
    expected = pd.read_csv(root / "frozen_validation_keys.csv.gz", dtype={"stock_code": str})
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    interface = json.loads(resolve(config["interface_config"]).read_text(encoding="utf-8"))
    model_ids = [str(item["id"]) for item in config["models"]]
    component_map = {str(item["id"]): str(item["component_family"]) for item in config["models"]}
    contract = validate_prediction_contract(predictions, expected, config["folds"], config["seeds"])
    mape_floor = float(interface["evaluation"]["mape_denominator_floor"])
    threshold = float(config["diagnostic_rules"]["direction_positive_threshold"])
    recalculated = dict(evaluate_predictions(
        predictions, universe, mape_floor, threshold, int(config["diagnostic_rules"]["return_group_count"]),
    ))
    recalculated["overall_pooled_metrics"] = overall_pooled_metrics(predictions, mape_floor, threshold)
    recalculated["fold_pooled_metrics"], recalculated["worst_fold_summary"] = fold_and_worst_fold(
        predictions, mape_floor, threshold,
    )
    recalculated["engineering_cost_summary"] = engineering_cost_summary(receipts)
    recalculated["seed_prediction_dispersion"] = seed_prediction_dispersion(predictions)
    recalculated["model_disagreement"], recalculated["component_disagreement"] = model_disagreement(
        predictions, component_map, threshold,
    )

    output_hashes_valid = True
    diagnostics_finite = True
    for name, frame in recalculated.items():
        stored = pd.read_csv(root / f"{name}.csv")
        diagnostics_finite = diagnostics_finite and bool(
            np.isfinite(stored.select_dtypes(include=[np.number]).to_numpy()).all()
        )
        rendered = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
        output_hashes_valid = output_hashes_valid and rendered == metadata["artifacts"][f"{name}_sha256"]
    payload = dict(metadata)
    declared_batch = payload.pop("batch_sha256", "")
    costs = recalculated["engineering_cost_summary"]
    worst = recalculated["worst_fold_summary"]
    disagreement = recalculated["model_disagreement"]
    checks = {
        "rules_frozen_before_cross_batch_read": config["status"] == "DIAGNOSTIC_RULES_FROZEN_BEFORE_CROSS_BATCH_READ",
        "all_mutation_permissions_false": not any(config["restrictions"].values()),
        "source_acceptances_and_hashes_valid": all(
            json.loads(resolve(source["acceptance"]).read_text(encoding="utf-8"))["passed"]
            and sha256_file(resolve(source["metadata"])) == source["metadata_sha256"]
            and sha256_file(resolve(source["predictions"])) == source["predictions_sha256"]
            and sha256_file(resolve(source["receipts"])) == source["receipts_sha256"]
            and sha256_file(resolve(source["acceptance"])) == source["acceptance_sha256"]
            for source in config["sources"]
        ),
        "exactly_ten_frozen_models": len(model_ids) == 10 and sorted(contract["models"]) == sorted(model_ids),
        "three_folds_three_seeds_complete": len(contract["grid_rows"]) == 30 and all(row["row_count"] == 1500 and row["missing_key_count"] == 0 and row["extra_key_count"] == 0 for row in contract["grid_rows"]),
        "prediction_row_count_45000": len(predictions) == 45000 and metadata["prediction_rows"] == 45000,
        "receipt_row_count_90": len(receipts) == 90 and metadata["receipt_rows"] == 90,
        "engineering_cost_complete": len(costs) == 10 and bool((costs["run_count"] == 9).all()) and bool(np.isfinite(costs.select_dtypes(include=[np.number]).to_numpy()).all()),
        "worst_fold_complete": len(worst) == 10 and bool((worst["fold_count"] == 3).all()),
        "per_stock_complete": len(recalculated["diagnostics_per_stock"]) == 1000,
        "industry_market_cap_return_groups_complete": (
            recalculated["diagnostics_industry"]["model_id"].nunique() == 10
            and recalculated["diagnostics_market_cap"]["model_id"].nunique() == 10
            and len(recalculated["diagnostics_return_decile"]) == 100
        ),
        "seed_stability_complete": len(recalculated["seed_summary"]) == 10 and len(recalculated["pairwise_seed_stability"]) == 30,
        "seed_dispersion_complete": len(recalculated["seed_prediction_dispersion"]) == 10,
        "model_disagreement_has_45_pairs": len(disagreement) == 45 and disagreement[["model_a", "model_b"]].drop_duplicates().shape[0] == 45,
        "component_disagreement_complete": len(recalculated["component_disagreement"]) == 45,
        "diagnostics_finite": diagnostics_finite and metadata["diagnostics_finite"],
        "independent_diagnostic_recalculation_matches": output_hashes_valid,
        "core_artifact_hashes_valid": (
            sha256_file(root / "unified_predictions_ten_models.csv.gz") == metadata["artifacts"]["unified_predictions_ten_models_sha256"]
            and sha256_file(root / "engineering_receipts_ten_models.csv") == metadata["artifacts"]["engineering_receipts_ten_models_sha256"]
            and sha256_file(root / "frozen_validation_keys.csv.gz") == metadata["artifacts"]["frozen_validation_keys_sha256"]
        ),
        "diagnostic_rule_hash_valid": metadata["diagnostic_rule_sha256"] == stable_json_sha256(config["diagnostic_rules"]),
        "batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "no_training_or_checkpoint_write": not metadata["training_executed"] and not metadata["checkpoint_written"],
        "no_ranking_selection_deletion_or_promotion": (
            not metadata["candidate_ranking_performed"] and not metadata["candidate_selection_performed"]
            and not metadata["model_deletion_performed"]
        ),
        "future_and_screening_not_accessed": not metadata["future_or_sealed_data_read"] and not metadata["screening_accessed"],
    }
    report = {
        "stage": "E-5.4 ten-model unified diagnostics acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(checks.values()),
        "checks": checks, "models": model_ids, "folds": config["folds"], "seeds": config["seeds"],
        "selection_exposure": config["selection_exposure"], "config_sha256": sha256_file(config_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
