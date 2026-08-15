"""Machine acceptance for E-4 fixed-graph stabilization."""

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
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e4_fixed_graph_stabilization_acceptance_100stocks_v1.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    report_path = root / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    development = pd.read_csv(root / "development_fold_results.csv")
    summary = pd.read_csv(root / "development_summary.csv")
    stability = pd.read_csv(root / "stability_fold_results.csv")
    original = json.loads(resolve(config["paths"]["original_training_checks_config"]).read_text(encoding="utf-8"))["stability"]
    payload = dict(report)
    declared_batch = payload.pop("batch_sha256", "")
    thresholds_equal = all(
        float(config["stability"][key]) == float(original[key])
        for key in ("minimum_pairwise_edge_jaccard", "minimum_pairwise_prediction_correlation", "maximum_mae_coefficient_of_variation")
    ) and [int(seed) for seed in config["stability"]["seeds"]] == [int(seed) for seed in original["seeds"]]
    expected_variants = {item["id"] for item in config["variants"]}
    checks = {
        "six_preregistered_variants_only": len(expected_variants) == 6 and set(summary["variant"].astype(str)) == expected_variants,
        "development_three_folds_complete": len(development) == 18 and set(development["fold_id"].astype(str)) == {"E_RO_01", "E_RO_02", "E_RO_03"},
        "unique_selected_variant": report["selected_variant"] in expected_variants,
        "original_seeds_and_thresholds_unchanged": thresholds_equal,
        "selected_variant_three_seed_three_fold_complete": len(stability) == 9 and set(stability["seed"].astype(int)) == set(config["stability"]["seeds"]),
        "metrics_finite": bool(np.isfinite(development[["mae", "rmse", "mse"]].to_numpy()).all() and np.isfinite(stability[["mae", "rmse", "mse"]].to_numpy()).all()),
        "adjacency_valid": bool(development["adjacency_finite"].astype(bool).all() and development["adjacency_row_stochastic"].astype(bool).all() and stability["adjacency_finite"].astype(bool).all() and stability["adjacency_row_stochastic"].astype(bool).all()),
        "artifact_hashes_valid": all(sha256_file(root / name) == report["artifacts"][key] for name, key in (
            ("development_fold_results.csv", "development_fold_results_sha256"),
            ("development_summary.csv", "development_summary_sha256"),
            ("development_predictions.csv.gz", "development_predictions_sha256"),
            ("stability_fold_results.csv", "stability_fold_results_sha256"),
            ("stability_predictions.csv.gz", "stability_predictions_sha256")
        )),
        "batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "decision_consistent": bool(report["allow_300_stock_expansion"]) == bool(report["stability_pass"]),
        "future_or_sealed_data_not_read": not report.get("future_or_sealed_data_read", True),
        "selection_exposure_declared": "fixed-graph stabilization" in report.get("selection_exposure", "")
    }
    acceptance = {
        "stage": "E-4.4 fixed-graph stabilization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "selected_variant": report["selected_variant"],
        "stability_pass": report["stability_pass"],
        "allow_300_stock_expansion": report["allow_300_stock_expansion"],
        "results_sha256": sha256_file(report_path)
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    raise SystemExit(0 if acceptance["passed"] else 2)


if __name__ == "__main__":
    main()
