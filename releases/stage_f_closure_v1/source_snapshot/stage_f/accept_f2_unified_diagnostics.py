"""Independently accept F-2.4 and enforce non-compensable stability failures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stage_e.hashing import sha256_file, stable_json_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matrix = pd.read_csv(root / "candidate_hard_gate_matrix.csv")
    summary = pd.read_csv(root / "candidate_hard_gate_summary.csv")
    gan_id = config["gan_candidate_id"]
    gan_matrix = matrix.loc[matrix["candidate_id"] == gan_id].set_index("gate_id")["passed"]
    stability_gate_ids = [
        "stability_seed_mae_cv", "stability_pairwise_pearson", "stability_pairwise_spearman",
        "stability_prediction_std_mean", "stability_prediction_std_p95",
    ]
    failed_stability = [gate for gate in stability_gate_ids if not bool(gan_matrix.loc[gate])]
    eligible = summary.loc[summary["all_hard_gates_pass"].astype(bool), "candidate_id"].astype(str).tolist()
    checks = {
        "metadata_pass": metadata["status"] == "PASS",
        "seven_models_retained": len(metadata["models_in_frozen_order"]) == 7,
        "four_candidates_retained": len(metadata["candidate_models"]) == 4,
        "all_twenty_gates_present": bool(matrix.groupby("candidate_id").size().eq(20).all()),
        "gan_stability_failure_retained": metadata["gan_stability_hard_failure_retained"] and len(failed_stability) > 0,
        "gan_known_four_stability_failures_retained": set(failed_stability) == {
            "stability_pairwise_pearson", "stability_pairwise_spearman",
            "stability_prediction_std_mean", "stability_prediction_std_p95",
        },
        "gan_not_eligible": gan_id not in eligible,
        "no_compensation": not metadata["stability_failure_compensation_allowed"],
        "no_new_training_or_inference": not metadata["new_training_performed"]
        and not metadata["new_model_inference_performed"],
        "no_ranking_or_deletion": not metadata["ranking_performed"]
        and not metadata["candidate_deletion_performed"],
        "future_data_closed": not metadata["screening_accessed"] and not metadata["final_accessed"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    acceptance = {
        "stage": "F-2.4 independent unified robustness acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "checks": checks, "passed_checks": sum(checks.values()), "required_checks": len(checks),
        "gan_candidate_id": gan_id, "gan_failed_stability_gates": failed_stability,
        "eligible_candidates": eligible,
        "eligibility_conclusion": metadata["eligibility_conclusion"],
        "stability_failures_non_compensable": True,
        "config_sha256": sha256_file(config_path), "metadata_sha256": sha256_file(metadata_path),
        "screening_authorized": False, "final_authorized": False,
    }
    acceptance["acceptance_sha256"] = stable_json_sha256(acceptance)
    output = resolve(config["paths"]["acceptance"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise RuntimeError("F-2.4 independent acceptance failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(path))


if __name__ == "__main__":
    main()
