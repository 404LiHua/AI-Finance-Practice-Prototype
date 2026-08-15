"""Independently accept F-2.3 engineering and retain the frozen stability result."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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
    stability = metadata["stability_result"]
    checks = {
        "engineering_metadata_pass": metadata["status"] == "PASS",
        "authorized_additional_seeds_only": metadata["additional_seeds"] == [20260723, 20260724],
        "nine_runs_complete": metadata["additional_run_count"] == 6 and metadata["all_three_seed_run_count"] == 9,
        "no_failures": metadata["failure_count"] == 0,
        "prediction_contract": metadata["prediction_contract_pass"] and metadata["prediction_rows"] == 4500,
        "losses_and_collapse": metadata["all_losses_finite"] and metadata["all_collapse_conditions_pass"],
        "independent_loading": metadata["all_independent_loads_pass"],
        "stress_entries": metadata["three_seed_stress_nonempty"],
        "cost": metadata["cost_limit_pass"],
        "stability_computed_without_ranking": len(stability["fold_seed_pairs"]) == 9
        and not metadata["ranking_performed"] and not metadata["promotion_recommendation_formed"],
        "future_data_closed": not metadata["screening_accessed"] and not metadata["final_accessed"],
    }
    engineering_pass = all(checks.values())
    stability_pass = bool(stability["all_stability_gates_pass"])
    status = "ENGINEERING_PASS_STABILITY_PASS" if engineering_pass and stability_pass else (
        "ENGINEERING_PASS_STABILITY_HARD_GATE_FAIL" if engineering_pass else "FAIL"
    )
    acceptance = {
        "stage": "F-2.3 three-seed GAN engineering and stability acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "engineering_checks": checks,
        "engineering_passed_checks": sum(checks.values()), "engineering_required_checks": len(checks),
        "candidate_id": config["candidate_id"], "seeds": config["all_seeds"], "folds": config["folds"],
        "stability_result": stability,
        "all_stability_gates_pass": stability_pass,
        "ranking_performed": False, "candidate_deletion_performed": False,
        "promotion_recommendation_formed": False,
        "config_sha256": sha256_file(config_path), "metadata_sha256": sha256_file(metadata_path),
        "screening_authorized": False, "final_authorized": False,
        "next_action": config["next_action_if_complete"],
    }
    acceptance["acceptance_sha256"] = stable_json_sha256(acceptance)
    output = resolve(config["paths"]["acceptance"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if not engineering_pass:
        raise RuntimeError("F-2.3 independent engineering acceptance failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path))


if __name__ == "__main__":
    main()
