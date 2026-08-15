"""Run the explicitly authorized F-2.1 synthetic minimal GAN training health check."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path

import torch

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_f.custody import StageFDataCustodyGuard, StageFDataCustodyViolation
from stage_f.gan import (
    BoundedConditionalGenerator,
    SpectralTemporalCritic,
    critic_wgan_gp_loss,
    deterministic_noise,
    generator_adversarial_loss,
    gradient_penalty,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def state_digest(module: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def parameter_l2_change(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    total = 0.0
    current = module.state_dict()
    for name, value in before.items():
        if value.is_floating_point() and name in current:
            difference = current[name].detach().cpu().to(torch.float64) - value.to(torch.float64)
            total += float((difference * difference).sum())
    return math.sqrt(total)


def train_once(config: dict[str, object]) -> dict[str, object]:
    seed = int(config["seed"])
    batch_config = config["synthetic_batch"]
    schedule = config["training_schedule"]
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = BoundedConditionalGenerator()
    critic = SpectralTemporalCritic()
    initial_generator = {name: value.detach().cpu().clone() for name, value in generator.state_dict().items()}
    initial_critic = {name: value.detach().cpu().clone() for name, value in critic.state_dict().items()}
    data_rng = torch.Generator(device="cpu").manual_seed(seed + 101)
    batch = int(batch_config["batch_size"])
    length = int(batch_config["sequence_length"])
    stocks = int(batch_config["stock_count"])
    features = int(batch_config["feature_count"])
    values = 0.35 * torch.randn((batch, length, stocks, features), generator=data_rng)
    node_available = torch.ones((batch, stocks), dtype=torch.bool)
    unavailable = int(batch_config["unavailable_stock_count"])
    node_available[:, -unavailable:] = False
    time_pattern = torch.linspace(-1.0, 1.0, length).view(1, length, 1, 1)
    feature_pattern = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25]).view(1, 1, 1, features)
    real_delta = 0.04 * torch.tanh(2.0 * time_pattern * feature_pattern)
    real_delta = real_delta.expand(batch, length, stocks, features).clone()
    real_delta = real_delta * node_available[:, None, :, None]
    real_values = values + real_delta
    noise = deterministic_noise((batch, length, stocks, 8), seed + 202)
    interpolation_rng = torch.Generator(device="cpu").manual_seed(seed + 303)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=float(schedule["learning_rate"]),
        betas=tuple(float(value) for value in schedule["betas"]),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=float(schedule["learning_rate"]),
        betas=tuple(float(value) for value in schedule["betas"]),
    )

    critic_losses: list[float] = []
    generator_losses: list[float] = []
    gradient_penalties: list[float] = []
    gradient_norms: list[float] = []

    def critic_step() -> None:
        critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_values, _ = generator(values, noise, node_available)
        real_score = critic(real_values, node_available)
        fake_score = critic(fake_values, node_available)
        interpolation = torch.rand((batch, 1, 1, 1), generator=interpolation_rng)
        penalty = gradient_penalty(critic, real_values, fake_values, interpolation, node_available)
        loss = critic_wgan_gp_loss(real_score, fake_score, penalty)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)
        critic_optimizer.step()
        critic_losses.append(float(loss.detach()))
        gradient_penalties.append(float(penalty.detach()))
        gradient_norms.append(float(norm.detach()))

    for _ in range(int(schedule["critic_warmup_epochs"])):
        critic_step()

    for _ in range(int(schedule["joint_epochs"])):
        for _ in range(int(schedule["critic_steps_per_generator_step"])):
            critic_step()
        generator_optimizer.zero_grad(set_to_none=True)
        fake_values, delta = generator(values, noise, node_available)
        fake_score = critic(fake_values, node_available)
        reference = values[:, -1, :, 0]
        real_forecaster_loss = ((values[:, -1, :, 0] - reference) ** 2).mean(dim=1)
        fake_forecaster_loss = ((fake_values[:, -1, :, 0] - reference) ** 2).mean(dim=1)
        detail = generator_adversarial_loss(
            fake_score,
            real_forecaster_loss,
            fake_forecaster_loss,
            delta,
        )
        detail.total.backward()
        generator_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=100.0)
        generator_optimizer.step()
        generator_losses.append(float(detail.total.detach()))
        gradient_norms.append(float(generator_norm.detach()))

    generator.eval()
    critic.eval()
    with torch.no_grad():
        fake_values, delta = generator(values, noise, node_available)
        final_gap = float((critic(real_values, node_available) - critic(fake_values, node_available)).mean())
    finite_values = critic_losses + generator_losses + gradient_penalties + gradient_norms + [final_gap]
    first_window = sum(critic_losses[:5]) / 5.0
    last_window = sum(critic_losses[-5:]) / 5.0
    relative_improvement = (first_window - last_window) / max(abs(first_window), 1e-12)
    return {
        "critic_loss_first_window_mean": first_window,
        "critic_loss_last_window_mean": last_window,
        "critic_loss_relative_improvement": relative_improvement,
        "generator_loss_initial": generator_losses[0],
        "generator_loss_final": generator_losses[-1],
        "gradient_penalty_max": max(gradient_penalties),
        "gradient_norm_max": max(gradient_norms),
        "final_critic_gap": final_gap,
        "maximum_absolute_delta": float(delta.abs().max()),
        "mean_absolute_delta": float(delta.abs().mean()),
        "unavailable_node_maximum_absolute_delta": float(delta[:, :, -unavailable:].abs().max()),
        "fake_shape": list(fake_values.shape),
        "all_training_values_finite": all(math.isfinite(value) for value in finite_values),
        "generator_parameter_l2_change": parameter_l2_change(initial_generator, generator),
        "critic_parameter_l2_change": parameter_l2_change(initial_critic, critic),
        "generator_state_sha256": state_digest(generator),
        "critic_state_sha256": state_digest(critic),
        "critic_optimizer_steps": len(critic_losses),
        "generator_optimizer_steps": len(generator_losses),
    }


def run(config_path: Path) -> Path:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen_path = resolve(config["frozen_addendum"]["path"])
    acceptance_path = resolve(config["frozen_addendum_acceptance"]["path"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    source_hashes_valid = all(
        sha256_file(resolve(item["path"])) == item["sha256"]
        for item in config["source_registry"].values()
    )
    thresholds = config["minimum_health_thresholds"]
    guard = StageFDataCustodyGuard.from_config(REPO_ROOT / "stage_f/configs/f0_data_custody_v1.json", REPO_ROOT)
    blocked_paths = [
        REPO_ROOT / "outputs/stage_c/stage_c_recommended_v2_c4_20230609_20240607/predictions.csv",
        REPO_ROOT / "outputs/stage_d/d5_screening_20240614_20250613/predictions.csv",
        REPO_ROOT / "data/screening/panel.parquet",
        REPO_ROOT / "data/final/panel.parquet",
    ]
    boundary_rejections = 0
    for path in blocked_paths:
        try:
            guard.assert_path_allowed(path, "f2_1_minimal_health")
        except StageFDataCustodyViolation:
            boundary_rejections += 1
    try:
        guard.assert_development_dates(["2023-06-09"], "trade_date")
    except StageFDataCustodyViolation:
        boundary_rejections += 1

    left = train_once(config)
    right = train_once(config)
    elapsed = time.perf_counter() - started
    metric_keys = [
        key for key, value in left.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    maximum_repeat_difference = max(abs(float(left[key]) - float(right[key])) for key in metric_keys)
    determinism = (
        maximum_repeat_difference <= float(thresholds["maximum_repeat_metric_difference"])
        and left["generator_state_sha256"] == right["generator_state_sha256"]
        and left["critic_state_sha256"] == right["critic_state_sha256"]
    )
    expected_shape = [
        int(config["synthetic_batch"]["batch_size"]),
        int(config["synthetic_batch"]["sequence_length"]),
        int(config["synthetic_batch"]["stock_count"]),
        int(config["synthetic_batch"]["feature_count"]),
    ]
    shapes_and_finite = (
        left["fake_shape"] == expected_shape
        and left["all_training_values_finite"]
        and left["maximum_absolute_delta"] <= float(thresholds["maximum_generated_absolute_delta"]) + 1e-7
        and left["unavailable_node_maximum_absolute_delta"] == 0.0
        and left["gradient_penalty_max"] <= 100.0
        and left["gradient_norm_max"] <= 100.0 + 1e-6
    )
    minimum_overfit = (
        left["critic_loss_relative_improvement"]
        >= float(thresholds["minimum_critic_loss_relative_improvement"])
        and left["generator_parameter_l2_change"]
        >= float(thresholds["minimum_generator_parameter_l2_change"])
        and left["critic_parameter_l2_change"]
        >= float(thresholds["minimum_critic_parameter_l2_change"])
    )
    boundaries = boundary_rejections == 5
    protocol_checks = {
        "source_hashes_valid": source_hashes_valid,
        "frozen_addendum_hash": sha256_file(frozen_path) == config["frozen_addendum"]["sha256"],
        "frozen_acceptance_hash": sha256_file(acceptance_path)
        == config["frozen_addendum_acceptance"]["sha256"],
        "frozen_acceptance_pass": acceptance["status"] == "PASS",
        "unique_candidate_unchanged": config["candidate_id"] == frozen["gan_candidate_id"],
        "schedule_unchanged": config["training_schedule"]["critic_warmup_epochs"]
        == frozen["pretraining"]["critic_warmup_epochs"]
        and config["training_schedule"]["joint_epochs"] == frozen["pretraining"]["joint_epochs"]
        and config["training_schedule"]["critic_steps_per_generator_step"]
        == frozen["pretraining"]["critic_steps_per_generator_step"],
        "minimal_scope_only": config["authorization"]["optimizer_steps_authorized"]
        and config["authorization"]["synthetic_minimal_training_authorized"]
        and not config["authorization"]["formal_fold_training_authorized"]
        and not config["authorization"]["additional_seed_training_authorized"],
        "future_data_closed": not config["authorization"]["screening_authorized"]
        and not config["authorization"]["final_authorized"],
        "cost_within_limit": elapsed <= float(thresholds["maximum_total_health_check_seconds"]),
    }
    checks = {
        "determinism": determinism,
        "tensor_shapes_and_finite_training": shapes_and_finite,
        "minimum_overfit_and_parameter_updates": minimum_overfit,
        "data_boundaries": boundaries,
    }
    status = "PASS" if all(protocol_checks.values()) and all(checks.values()) else "FAIL"
    receipt = {
        "stage": "F-2.1 minimal GAN training health check",
        "generated_at": "2026-07-28",
        "status": status,
        "candidate_id": config["candidate_id"],
        "seed": config["seed"],
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "required_checks": len(checks),
        "protocol_checks": protocol_checks,
        "passed_protocol_checks": sum(protocol_checks.values()),
        "required_protocol_checks": len(protocol_checks),
        "left_run": left,
        "repeat_run": right,
        "maximum_repeat_metric_difference": maximum_repeat_difference,
        "boundary_rejections": boundary_rejections,
        "elapsed_seconds": elapsed,
        "optimizer_steps_executed": True,
        "synthetic_minimal_training_executed": True,
        "formal_fold_training_executed": False,
        "additional_seed_training_executed": False,
        "candidate_metrics_read": False,
        "candidate_ranking_performed": False,
        "screening_accessed": False,
        "final_accessed": False,
        "config_sha256": sha256_file(config_path),
        "next_action": config["next_action_if_pass"] if status == "PASS" else config["next_action_if_fail"],
    }
    receipt["receipt_sha256"] = stable_json_sha256(receipt)
    output = resolve(config["paths"]["receipt"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise RuntimeError("F-2.1 minimal GAN training health check failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path))


if __name__ == "__main__":
    main()
