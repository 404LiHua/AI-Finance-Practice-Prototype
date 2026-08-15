"""Machine acceptance for the Stage E E-5.1 unified interface."""

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
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e5_interface_acceptance_v1.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "unified_predictions.csv.gz", dtype={"stock_code": str})
    expected = pd.read_csv(root / "frozen_validation_keys.csv.gz", dtype={"stock_code": str})
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    contract = validate_prediction_contract(predictions, expected, config["folds"], config["seeds"])
    recalculated = evaluate_predictions(
        predictions, universe, float(config["evaluation"]["mape_denominator_floor"]),
        float(config["evaluation"]["direction_positive_threshold"]), int(config["evaluation"]["return_group_count"]),
    )
    payload = dict(metadata)
    declared_batch = payload.pop("batch_sha256", "")
    metric_hashes_valid = True
    finite = True
    for name, frame in recalculated.items():
        stored = pd.read_csv(root / f"{name}.csv")
        finite = finite and bool(np.isfinite(stored.select_dtypes(include=[np.number]).to_numpy()).all())
        temporary = root / f".{name}.recalc.csv"
        frame.to_csv(temporary, index=False)
        metric_hashes_valid = metric_hashes_valid and sha256_file(temporary) == metadata["artifacts"][f"{name}_sha256"]
        temporary.unlink()
    checks = {
        "interface_status_frozen": config["status"] == "INTERFACE_CONTRACT_FROZEN_BEFORE_E5_BASELINE_TRAINING",
        "no_baseline_training_executed": not metadata["baseline_training_executed"] and not config["restrictions"]["baseline_training_allowed"],
        "three_folds_three_seeds": metadata["folds"] == config["folds"] and metadata["seeds"] == config["seeds"],
        "all_text_views_share_frozen_keys": len(metadata["text_view_hashes"]) == len(config["folds"]) * len(config["required_text_views"]),
        "prediction_contract_valid": len(contract["models"]) == 2 and all(row["missing_key_count"] == 0 and row["extra_key_count"] == 0 for row in contract["grid_rows"]),
        "metrics_finite": finite,
        "independent_metric_recalculation_matches": metric_hashes_valid,
        "prediction_hash_valid": sha256_file(root / "unified_predictions.csv.gz") == metadata["artifacts"]["unified_predictions_sha256"],
        "frozen_key_hash_valid": sha256_file(root / "frozen_validation_keys.csv.gz") == metadata["artifacts"]["frozen_validation_keys_sha256"],
        "batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "future_or_sealed_data_not_read": not metadata["future_or_sealed_data_read"],
        "screening_not_accessed": not metadata["screening_accessed"],
    }
    report = {
        "stage": "E-5.1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks, "models_in_fixture": metadata["models"],
        "config_sha256": metadata["config_sha256"], "metadata_sha256": sha256_file(metadata_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
