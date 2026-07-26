"""Machine acceptance for frozen-protocol E-4S.2 control stability runs."""

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

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e4s2_control_stability_acceptance_v2.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    results_path = root / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    protocol_path = resolve(config["paths"]["protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fold = pd.read_csv(root / "fold_results.csv")
    curves = pd.read_csv(root / "training_curves.csv")
    predictions = pd.read_csv(root / "predictions.csv.gz")
    payload = dict(results)
    declared_batch = payload.pop("batch_sha256", "")
    expected_runs = len(protocol["allowed_stage_e4s2_controls"]) * len(protocol["folds"]) * len(protocol["seeds"])
    checks = {
        "frozen_protocol_hash_unchanged": sha256_file(protocol_path) == config["expected_protocol_sha256"] == results["protocol_sha256"],
        "only_two_frozen_controls": results["controls"] == protocol["allowed_stage_e4s2_controls"],
        "original_three_seeds_and_folds": results["seeds"] == protocol["seeds"] and results["folds"] == protocol["folds"],
        "all_18_runs_complete": len(fold) == expected_runs,
        "all_train_cross_sections_used": bool((fold["train_cross_section_count"] > 48).all()),
        "fixed_40_epochs": bool((fold["epochs_completed"] == protocol["training"]["epochs"]).all()) and int(curves["epoch"].max()) == protocol["training"]["epochs"],
        "no_validation_checkpoint_selection": bool((~fold["validation_used_for_checkpoint_selection"].astype(bool)).all()) and set(fold["checkpoint_selection"].astype(str)) == {"final_epoch_ema"},
        "metrics_finite": bool(np.isfinite(fold[["mae", "mse", "rmse", "direction_accuracy", "final_train_loss"]].to_numpy()).all()),
        "adjacency_valid": bool(fold["adjacency_finite"].astype(bool).all() and fold["adjacency_row_stochastic"].astype(bool).all()),
        "prediction_keys_complete": len(predictions) > 0 and set(predictions["variant"].astype(str)) == set(protocol["allowed_stage_e4s2_controls"]),
        "artifact_hashes_valid": sha256_file(root / "fold_results.csv") == results["artifacts"]["fold_results_sha256"] and sha256_file(root / "predictions.csv.gz") == results["artifacts"]["predictions_sha256"] and sha256_file(root / "training_curves.csv") == results["artifacts"]["training_curves_sha256"],
        "batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "decision_consistent": bool(results["stable_controls"]) == bool(results["gate_a_pass"]) == bool(results["allow_stage_e4s3"]),
        "future_or_sealed_data_not_read": not results["future_or_sealed_data_read"],
        "screening_not_accessed": not results["screening_accessed"],
        "three_hundred_still_disabled": not results["allow_300_stock_expansion"],
    }
    acceptance = {
        "stage": "E-4S.2", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks,
        "gate_a_pass": results["gate_a_pass"], "stable_controls": results["stable_controls"],
        "allow_stage_e4s3": results["allow_stage_e4s3"], "allow_300_stock_expansion": False,
        "protocol_sha256": results["protocol_sha256"], "results_sha256": sha256_file(results_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    raise SystemExit(0 if acceptance["passed"] else 2)


if __name__ == "__main__":
    main()
