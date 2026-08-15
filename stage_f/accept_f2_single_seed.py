"""Independently accept F-2.2 single-seed three-fold engineering artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_f.run_f2_single_seed import load_effective_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def run(config_path: Path) -> Path:
    config, correction = load_effective_config(config_path)
    root = resolve(config["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    receipts_path = root / "engineering_receipts.json"
    failures_path = root / "failure_receipts.json"
    predictions_path = root / "unified_predictions.csv.gz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    checks = {
        "metadata_pass": metadata["status"] == "PASS",
        "three_fold_receipts": len(receipts) == 3 and {r["fold_id"] for r in receipts} == set(config["folds"]),
        "no_failure_receipts": failures == [] and metadata["failure_count"] == 0,
        "losses_and_collapse_pass": metadata["all_gan_losses_finite"]
        and metadata["all_forecaster_losses_finite"] and metadata["all_collapse_conditions_pass"],
        "independent_loading_pass": metadata["all_independent_loads_pass"],
        "frozen_keys_pass": metadata["all_frozen_validation_keys_pass"]
        and all(r["validation_rows"] == 500 for r in receipts),
        "normal_and_stress_pass": metadata["all_normal_and_stress_entries_pass"]
        and all(r["stress_inference"]["scenario_count"] == 9 for r in receipts),
        "cost_pass": metadata["cost_limit_pass"],
        "artifact_hashes_match": metadata["artifacts"]["engineering_receipts_sha256"] == sha256_file(receipts_path)
        and metadata["artifacts"]["failure_receipts_sha256"] == sha256_file(failures_path)
        and metadata["artifacts"]["unified_predictions_sha256"] == sha256_file(predictions_path),
        "scope_closed": not metadata["ranking_performed"] and not metadata["additional_seed_executed"]
        and not metadata["screening_accessed"] and not metadata["final_accessed"],
    }
    acceptance = {
        "stage": "F-2.2 single-seed three-fold GAN engineering independent acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "passed_checks": sum(checks.values()), "required_checks": len(checks),
        "candidate_id": config["candidate_id"], "seed": config["seed"],
        "config_sha256": sha256_file(config_path), "metadata_sha256": sha256_file(metadata_path),
        "additional_seed_training_authorized": False, "screening_authorized": False, "final_authorized": False,
        "adapter_correction": correction,
        "next_action": "request separate authorization before F-2.3 seeds 20260723 and 20260724",
    }
    acceptance["acceptance_sha256"] = stable_json_sha256(acceptance)
    output = resolve(config["paths"]["acceptance"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if acceptance["status"] != "PASS":
        raise RuntimeError("F-2.2 independent acceptance failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path))


if __name__ == "__main__":
    main()
