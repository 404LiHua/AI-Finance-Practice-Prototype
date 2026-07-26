"""Machine acceptance for preregistered E-4.4 bounded ablations."""

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
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e4_ablation_acceptance_100stocks_v1.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    fold_path = root / "fold_results.csv"
    summary_path = root / "summary.csv"
    predictions_path = root / "validation_predictions.csv.gz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fold = pd.read_csv(fold_path)
    summary = pd.read_csv(summary_path)
    predictions = pd.read_csv(predictions_path)
    expected_variants = [item["id"] for item in config["variants"]]
    expected_pairs = {(variant, fold_id) for variant in expected_variants for fold_id in ("E_RO_01", "E_RO_02", "E_RO_03")}
    actual_pairs = set(zip(fold["variant"].astype(str), fold["fold_id"].astype(str)))
    payload = dict(metadata)
    declared_batch = payload.pop("batch_sha256", "")
    checks = {
        "variant_set_frozen": set(summary["variant"].astype(str)) == set(expected_variants),
        "all_three_folds_run": actual_pairs == expected_pairs and len(fold) == 30,
        "metrics_finite": bool(np.isfinite(fold[["mae", "mse", "rmse", "direction_accuracy"]].to_numpy()).all()),
        "all_adjacencies_finite": bool(fold["adjacency_finite"].astype(bool).all()),
        "all_adjacencies_row_stochastic": bool(fold["adjacency_row_stochastic"].astype(bool).all()),
        "validation_predictions_present": len(predictions) > 0 and set(predictions["variant"].astype(str)) == set(expected_variants),
        "fold_results_hash_valid": sha256_file(fold_path) == metadata["artifacts"]["fold_results_sha256"],
        "summary_hash_valid": sha256_file(summary_path) == metadata["artifacts"]["summary_sha256"],
        "predictions_hash_valid": sha256_file(predictions_path) == metadata["artifacts"]["predictions_sha256"],
        "metadata_batch_sha_valid": declared_batch == stable_json_sha256(payload),
        "single_frozen_seed": metadata.get("seed") == config["seed"],
        "selection_exposure_declared": "development ablation" in metadata.get("selection_exposure", ""),
        "future_or_sealed_data_not_read": not metadata.get("future_or_sealed_data_read", True),
    }
    report = {
        "stage": "E-4.4", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks,
        "best_mean_mae_variant": str(summary.sort_values("mean_mae", kind="stable").iloc[0]["variant"]),
        "best_mean_mae": float(summary["mean_mae"].min()),
        "run_count": len(fold), "metadata_sha256": sha256_file(metadata_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
