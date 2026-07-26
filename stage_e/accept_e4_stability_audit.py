"""Machine acceptance for the no-training E-4S.1 stability audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e4_stability_audit_acceptance_v1.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    results_path = root / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    payload = dict(results)
    declared_batch = payload.pop("batch_sha256", "")
    artifact_files = {
        "learned_graph_recomputed_predictions_sha256": "learned_graph_recomputed_predictions.csv.gz",
        "sample_key_audit_sha256": "sample_key_audit.csv",
        "prediction_variance_by_fold_sha256": "prediction_variance_by_fold.csv",
        "pairwise_overall_sha256": "pairwise_overall.csv",
        "pairwise_by_fold_sha256": "pairwise_by_fold.csv",
        "pairwise_per_stock_sha256": "pairwise_per_stock.csv",
        "cross_seed_dispersion_sha256": "cross_seed_dispersion.csv",
    }
    keys = pd.read_csv(root / "sample_key_audit.csv")
    overall = pd.read_csv(root / "pairwise_overall.csv")
    checks = {
        "training_prohibited_by_config": not config["restrictions"]["training_allowed"],
        "no_training_executed": results["no_training_executed"],
        "expected_three_seeds_present": set(keys["seed"].astype(int)) == set(config["expected_seeds"]),
        "all_four_structures_audited": len(set(keys["variant"].astype(str))) == 4,
        "sample_keys_equal_across_seeds": results["sample_keys_equal_across_seeds"],
        "learned_metrics_independently_reproduced": results["reported_learned_graph_metrics_independently_reproduced"],
        "predictions_not_constant": results["zero_prediction_variance_rows"] == 0,
        "pairwise_metrics_finite": bool(overall[["pearson", "spearman", "prediction_sign_agreement"]].notna().all().all()),
        "training_protocol_v2_frozen": sha256_file(resolve(config["paths"]["training_protocol_v2"])) == results["training_protocol_v2_sha256"],
        "artifact_hashes_valid": all(sha256_file(root / name) == results["artifacts"][key] for key, name in artifact_files.items()),
        "batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "future_or_sealed_data_not_read": not results["future_or_sealed_data_read"],
        "screening_not_accessed": not results["screening_accessed"],
        "unique_audit_conclusion_present": results["unique_conclusion"] == "TRAINING_CONVERGENCE_AND_CHECKPOINT_SELECTION_INSTABILITY",
    }
    acceptance = {
        "stage": "E-4S.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "conclusion": results["unique_conclusion"],
        "training_protocol_v2_sha256": results["training_protocol_v2_sha256"],
        "results_sha256": sha256_file(results_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    raise SystemExit(0 if acceptance["passed"] else 2)


if __name__ == "__main__":
    main()
