"""Freeze the bounded F-2 GAN addendum without executing GAN training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_f.gan import BoundedConditionalGenerator, SpectralTemporalCritic, parameter_count


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_hashes_valid = all(
        sha256_file(resolve(item["path"])) == item["sha256"]
        for item in config["source_registry"].values()
    )
    generator = BoundedConditionalGenerator(**config["architecture"]["generator"]["constructor"])
    critic = SpectralTemporalCritic(**config["architecture"]["critic"]["constructor"])
    upstream_path = resolve(config["upstream_f1_conclusion"]["path"])
    f1_acceptance = json.loads(upstream_path.read_text(encoding="utf-8"))
    checks = {
        "source_hashes_valid": source_hashes_valid,
        "upstream_f1_hash_valid": sha256_file(upstream_path)
        == config["upstream_f1_conclusion"]["sha256"],
        "f1_formal_no_candidate_retained": f1_acceptance["eligibility_conclusion"]
        == config["upstream_f1_conclusion"]["required_conclusion"],
        "exactly_one_generator_and_critic": config["architecture"]["candidate_count"] == 1,
        "generator_parameter_bound": parameter_count(generator)
        <= config["cost_limits"]["maximum_generator_parameters"],
        "critic_parameter_bound": parameter_count(critic)
        <= config["cost_limits"]["maximum_critic_parameters"],
        "bounded_residual_generator": config["architecture"]["generator"]["constructor"]["max_delta"] == 0.05,
        "wgan_gp_loss_frozen": config["losses"]["critic"]["gradient_penalty_weight"] == 10.0,
        "pretraining_schedule_frozen": config["pretraining"]["critic_warmup_epochs"] == 5
        and config["pretraining"]["joint_epochs"] == 20
        and config["pretraining"]["critic_steps_per_generator_step"] == 3,
        "normalization_frozen": config["normalization"]["generator"] == "GroupNorm(1,32); no BatchNorm"
        and config["normalization"]["critic"] == "spectral_norm on every Conv1d and Linear; no BatchNorm",
        "collapse_conditions_complete": len(config["collapse_failure_conditions"]) >= 7,
        "cost_limits_complete": config["cost_limits"]["maximum_total_nine_run_seconds"] <= 360.0,
        "training_not_authorized": not config["authorization"]["gan_training_authorized"]
        and config["authorization"]["explicit_training_authorization_required"],
        "future_data_forbidden": not config["restrictions"]["screening_allowed"]
        and not config["restrictions"]["final_allowed"],
    }
    receipt = {
        "stage": "F-2.0 bounded GAN addendum freeze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "required_checks": len(checks),
        "generator_parameter_count": parameter_count(generator),
        "critic_parameter_count": parameter_count(critic),
        "combined_gan_parameter_count": parameter_count(generator) + parameter_count(critic),
        "gan_training_executed": False,
        "optimizer_step_executed": False,
        "candidate_metrics_read": False,
        "screening_accessed": False,
        "final_accessed": False,
        "config_sha256": sha256_file(config_path),
        "hard_gates_sha256": config["frozen_evaluation"]["f0_hard_gates_sha256"],
        "next_action": "request explicit GAN training authorization only when ready to enter F-2.1",
    }
    receipt["receipt_sha256"] = stable_json_sha256(receipt)
    output = resolve(config["paths"]["freeze_receipt"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if receipt["status"] != "PASS":
        raise RuntimeError("F-2 GAN addendum freeze failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(path))


if __name__ == "__main__":
    main()
