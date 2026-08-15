"""Independently accept the frozen F-2 GAN addendum without training."""

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
    freeze = json.loads(resolve(config["paths"]["freeze_receipt"]).read_text(encoding="utf-8"))
    checks = {
        "freeze_receipt_pass": freeze["status"] == "PASS" and freeze["passed_checks"] == freeze["required_checks"],
        "config_hash_matches": freeze["config_sha256"] == sha256_file(config_path),
        "unique_candidate": config["gan_candidate_id"] == "stock_node_gwnet_bounded_cwgan_gp_l8",
        "fixed_architecture": config["architecture"]["candidate_count"] == 1,
        "fixed_losses": set(config["losses"]) == {"critic", "generator", "forecaster_retraining"},
        "fixed_pretraining": config["pretraining"]["checkpoint_selection"] == "final_epoch_only_if_all_health_checks_pass",
        "gradient_penalty_present": config["losses"]["critic"]["gradient_penalty_weight"] == 10.0,
        "normalization_present": "GroupNorm" in config["normalization"]["generator"]
        and "spectral_norm" in config["normalization"]["critic"],
        "collapse_is_hard_failure": config["collapse_policy"] == "any_condition_is_immediate_hard_failure_keep_receipt",
        "cost_is_bounded": config["cost_limits"]["maximum_total_nine_run_seconds"] == 360.0,
        "authorization_is_closed": not config["authorization"]["gan_training_authorized"],
        "future_data_is_closed": not config["restrictions"]["screening_allowed"]
        and not config["restrictions"]["final_allowed"],
    }
    acceptance = {
        "stage": "F-2.0 bounded GAN addendum independent acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "required_checks": len(checks),
        "gan_training_authorized": False,
        "gan_training_executed": False,
        "optimizer_step_executed": False,
        "screening_accessed": False,
        "final_accessed": False,
        "config_sha256": sha256_file(config_path),
        "freeze_receipt_sha256": sha256_file(resolve(config["paths"]["freeze_receipt"])),
        "next_action": "await explicit user authorization before F-2.1 minimal GAN training checks",
    }
    acceptance["acceptance_sha256"] = stable_json_sha256(acceptance)
    output = resolve(config["paths"]["acceptance_receipt"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if acceptance["status"] != "PASS":
        raise RuntimeError("F-2 GAN addendum acceptance failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(path))


if __name__ == "__main__":
    main()
