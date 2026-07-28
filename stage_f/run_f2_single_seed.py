"""Run F-2.2 seed-20260725 three-fold GAN engineering receipts without ranking."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage_e.e5.interface import E5FoldView, load_fold_view, validation_key_frame
from stage_e.e5.neural_graph import FixedIndustryGraphWaveNet, fixed_industry_adjacency
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import set_seed
from stage_f.custody import StageFDataCustodyGuard
from stage_f.gan import (
    BoundedConditionalGenerator,
    SpectralTemporalCritic,
    critic_wgan_gp_loss,
    generator_adversarial_loss,
    gradient_penalty,
    parameter_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_effective_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if "extends_config" not in raw:
        return raw, {"correction_applied": False}
    base_path = resolve(raw["extends_config"]["path"])
    if sha256_file(base_path) != raw["extends_config"]["sha256"]:
        raise RuntimeError("F-2.2 correction base config hash mismatch")
    config, prior_correction = load_effective_config(base_path)
    config["status"] = raw["status"]
    config["source_registry"]["runner"] = dict(raw["runner_override"])
    config["source_registry"]["independent_acceptance"] = dict(raw["acceptance_override"])
    config["paths"]["output_root"] = raw["output_root"]
    config["paths"]["acceptance"] = raw["acceptance_output"]
    return config, {
        "correction_applied": True,
        "correction_id": raw["correction_id"],
        "correction_scope": raw["correction_scope"],
        "failed_attempt": raw["failed_attempt"],
        "base_config_sha256": raw["extends_config"]["sha256"],
        "prior_correction": prior_correction,
    }


def state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def masked_per_sample_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.huber_loss(prediction, target, reduction="none") * mask.to(prediction.dtype)
    return loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(prediction.dtype)


def load_frozen_forecaster(path: Path) -> tuple[FixedIndustryGraphWaveNet, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["family"] != "stock_node_gwnet_fixed_industry":
        raise ValueError("F-2.2 frozen forecaster family changed")
    model = FixedIndustryGraphWaveNet(
        int(payload["input_size"]), np.asarray(payload["adjacency"], dtype=np.float32), dict(payload["parameters"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def delta_similarity_fraction(delta: torch.Tensor, available: torch.Tensor) -> float:
    rows = delta.detach().reshape(delta.shape[0], -1)
    if len(rows) < 2:
        stock_rows = delta.detach().permute(0, 2, 1, 3).reshape(-1, delta.shape[1] * delta.shape[3])
        stock_mask = available.detach().reshape(-1)
        rows = stock_rows[stock_mask][:32]
    if len(rows) < 2:
        return 0.0
    rows = nn.functional.normalize(rows, dim=1, eps=1e-12)
    similarities = rows @ rows.T
    upper = similarities[torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)]
    return float((upper > 0.995).to(torch.float32).mean()) if upper.numel() else 0.0


class GanCollapse(RuntimeError):
    def __init__(self, conditions: list[str], history: list[dict[str, Any]]) -> None:
        super().__init__("; ".join(conditions))
        self.conditions = conditions
        self.history = history


def train_gan(
    view: E5FoldView,
    frozen_forecaster: FixedIndustryGraphWaveNet,
    config: dict[str, Any],
    seed: int,
    log_path: Path,
) -> tuple[BoundedConditionalGenerator, SpectralTemporalCritic, dict[str, Any]]:
    set_seed(seed)
    params = config["gan_parameters"]
    thresholds = config["collapse_thresholds"]
    sequence_length = int(params["sequence_length"])
    train = view.split_indices("train")
    values = torch.from_numpy(view.numeric_values[train, -sequence_length:].astype(np.float32))
    targets = torch.from_numpy(view.target_scaled[train].astype(np.float32))
    masks = torch.from_numpy(view.sample_mask[train].astype(bool))
    available = torch.from_numpy(view.node_available[train].astype(bool))
    dataset = torch.utils.data.TensorDataset(values, targets, masks, available)
    loader_rng = torch.Generator(device="cpu").manual_seed(seed + 101)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(params["batch_size"]), shuffle=True, generator=loader_rng,
    )
    noise_rng = torch.Generator(device="cpu").manual_seed(seed + 202)
    interpolation_rng = torch.Generator(device="cpu").manual_seed(seed + 303)
    generator = BoundedConditionalGenerator(feature_count=values.shape[-1])
    critic = SpectralTemporalCritic(feature_count=values.shape[-1])
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=float(params["learning_rate"]),
        betas=tuple(float(value) for value in params["betas"]),
    )
    optimizer_c = torch.optim.Adam(
        critic.parameters(), lr=float(params["learning_rate"]),
        betas=tuple(float(value) for value in params["betas"]),
    )
    history: list[dict[str, Any]] = []
    low_gap_epochs = 0
    low_delta_epochs = 0
    bad_critic_gradient_epochs = 0
    bad_generator_gradient_epochs = 0
    started = time.perf_counter()
    total_epochs = int(params["critic_warmup_epochs"]) + int(params["joint_epochs"])
    for epoch in range(1, total_epochs + 1):
        joint = epoch > int(params["critic_warmup_epochs"])
        critic_losses: list[float] = []
        generator_losses: list[float] = []
        penalties: list[float] = []
        gaps: list[float] = []
        critic_gradients: list[float] = []
        generator_gradients: list[float] = []
        delta_means: list[float] = []
        last_delta: torch.Tensor | None = None
        last_available: torch.Tensor | None = None
        for batch_values, batch_targets, batch_masks, batch_available in loader:
            critic_steps = int(params["critic_steps_per_generator_step"]) if joint else 1
            for _ in range(critic_steps):
                optimizer_c.zero_grad(set_to_none=True)
                noise = torch.randn((*batch_values.shape[:-1], 8), generator=noise_rng)
                with torch.no_grad():
                    fake_values, _ = generator(batch_values, noise, batch_available)
                real_score = critic(batch_values, batch_available)
                fake_score = critic(fake_values, batch_available)
                interpolation = torch.rand((len(batch_values), 1, 1, 1), generator=interpolation_rng)
                penalty = gradient_penalty(critic, batch_values, fake_values, interpolation, batch_available)
                loss_c = critic_wgan_gp_loss(
                    real_score, fake_score, penalty,
                    gradient_penalty_weight=float(params["gradient_penalty_weight"]),
                    drift_weight=float(params["critic_drift_weight"]),
                )
                loss_c.backward()
                norm_c = torch.nn.utils.clip_grad_norm_(critic.parameters(), float(params["gradient_clip"]))
                optimizer_c.step()
                critic_losses.append(float(loss_c.detach()))
                penalties.append(float(penalty.detach()))
                gaps.append(float((real_score.mean() - fake_score.mean()).detach()))
                critic_gradients.append(float(norm_c.detach()))
            if joint:
                optimizer_g.zero_grad(set_to_none=True)
                noise = torch.randn((*batch_values.shape[:-1], 8), generator=noise_rng)
                fake_values, delta = generator(batch_values, noise, batch_available)
                fake_score = critic(fake_values, batch_available)
                with torch.no_grad():
                    real_forecaster_loss = masked_per_sample_huber(
                        frozen_forecaster(batch_values), batch_targets, batch_masks,
                    )
                fake_forecaster_loss = masked_per_sample_huber(
                    frozen_forecaster(fake_values), batch_targets, batch_masks,
                )
                detail = generator_adversarial_loss(
                    fake_score, real_forecaster_loss, fake_forecaster_loss, delta,
                    hardness_weight=float(params["hardness_weight"]),
                    energy_weight=float(params["energy_weight"]),
                    target_mean_absolute_delta=float(params["target_mean_absolute_delta"]),
                    maximum_hardness_gain=float(params["maximum_hardness_gain"]),
                )
                detail.total.backward()
                norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), float(params["gradient_clip"]))
                optimizer_g.step()
                generator_losses.append(float(detail.total.detach()))
                generator_gradients.append(float(norm_g.detach()))
                delta_means.append(float(delta.detach().abs().mean()))
                last_delta = delta.detach()
                last_available = batch_available.detach()
        gap = float(np.mean(gaps))
        mean_delta = float(np.mean(delta_means)) if delta_means else 0.0
        critic_gradient = float(np.mean(critic_gradients))
        generator_gradient = float(np.mean(generator_gradients)) if generator_gradients else None
        low_gap_epochs = low_gap_epochs + 1 if abs(gap) < float(thresholds["minimum_absolute_critic_gap"]) else 0
        if joint:
            low_delta_epochs = low_delta_epochs + 1 if mean_delta < float(thresholds["minimum_mean_absolute_delta"]) else 0
            bad_generator_gradient_epochs = (
                bad_generator_gradient_epochs + 1
                if generator_gradient is None
                or generator_gradient < float(thresholds["minimum_gradient_norm"])
                or generator_gradient > float(thresholds["maximum_gradient_norm"])
                else 0
            )
        bad_critic_gradient_epochs = (
            bad_critic_gradient_epochs + 1
            if critic_gradient < float(thresholds["minimum_gradient_norm"])
            or critic_gradient > float(thresholds["maximum_gradient_norm"])
            else 0
        )
        feature_stds: list[float] = []
        high_similarity_fraction = 0.0
        maximum_delta = 0.0
        unavailable_delta = 0.0
        if last_delta is not None and last_available is not None:
            maximum_delta = float(last_delta.abs().max())
            unavailable_delta = float((last_delta * (~last_available)[:, None, :, None]).abs().max())
            feature_stds = [float(last_delta[..., index].std()) for index in range(last_delta.shape[-1])]
            high_similarity_fraction = delta_similarity_fraction(last_delta, last_available)
        row = {
            "epoch": epoch,
            "phase": "joint" if joint else "critic_warmup",
            "critic_loss": float(np.mean(critic_losses)),
            "generator_loss": float(np.mean(generator_losses)) if generator_losses else None,
            "gradient_penalty_max": max(penalties),
            "critic_gap": gap,
            "critic_gradient_norm": critic_gradient,
            "generator_gradient_norm": generator_gradient,
            "mean_absolute_delta": mean_delta if joint else None,
            "maximum_absolute_delta": maximum_delta if joint else None,
            "unavailable_node_maximum_absolute_delta": unavailable_delta if joint else None,
            "per_feature_delta_std": feature_stds,
            "high_similarity_pair_fraction": high_similarity_fraction if joint else None,
        }
        history.append(row)
        values_to_check = [row["critic_loss"], row["gradient_penalty_max"], gap, critic_gradient]
        values_to_check.extend(value for value in (row["generator_loss"], generator_gradient, mean_delta) if value is not None)
        failures: list[str] = []
        if not all(math.isfinite(float(value)) for value in values_to_check):
            failures.append("non_finite_loss_gradient_or_statistic")
        if row["gradient_penalty_max"] > float(thresholds["maximum_gradient_penalty"]):
            failures.append("gradient_penalty_above_100")
        if low_gap_epochs >= int(thresholds["critic_gap_patience_epochs"]):
            failures.append("critic_gap_below_1e-4_for_three_epochs")
        if abs(gap) > float(thresholds["maximum_absolute_critic_gap"]):
            failures.append("critic_gap_above_20")
        if joint and low_delta_epochs >= int(thresholds["delta_collapse_patience_epochs"]):
            failures.append("mean_absolute_delta_below_0.001_for_three_epochs")
        if joint and mean_delta > float(thresholds["maximum_mean_absolute_delta"]):
            failures.append("mean_absolute_delta_above_0.045")
        if joint and sum(std < float(thresholds["minimum_per_feature_delta_std"]) for std in feature_stds) > int(
            thresholds["maximum_collapsed_feature_count"]
        ):
            failures.append("at_least_half_features_have_collapsed_delta_std")
        if joint and high_similarity_fraction >= float(thresholds["maximum_high_similarity_pair_fraction"]):
            failures.append("generated_delta_pairwise_similarity_collapse")
        if bad_critic_gradient_epochs >= int(thresholds["gradient_patience_epochs"]):
            failures.append("critic_gradient_outside_bounds_for_three_epochs")
        if joint and bad_generator_gradient_epochs >= int(thresholds["gradient_patience_epochs"]):
            failures.append("generator_gradient_outside_bounds_for_three_epochs")
        if joint and (maximum_delta > 0.0500001 or unavailable_delta != 0.0):
            failures.append("generated_delta_shape_mask_or_bound_violation")
        if failures:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise GanCollapse(failures, history)
    duration = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    detail = {
        "gan_training_seconds": duration,
        "gan_epochs_completed": len(history),
        "critic_optimizer_steps": sum(
            len(loader) * (int(params["critic_steps_per_generator_step"]) if row["phase"] == "joint" else 1)
            for row in history
        ),
        "generator_optimizer_steps": len(loader) * int(params["joint_epochs"]),
        "all_gan_losses_finite": True,
        "collapse_conditions_pass": True,
        "generator_parameter_count": parameter_count(generator),
        "critic_parameter_count": parameter_count(critic),
        "final_epoch": history[-1],
    }
    return generator, critic, detail


def train_augmented_forecaster(
    view: E5FoldView,
    adjacency: np.ndarray,
    generator: BoundedConditionalGenerator,
    config: dict[str, Any],
    seed: int,
    checkpoint: Path,
    log_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = config["forecaster_parameters"]
    set_seed(seed)
    sequence_length = int(params["sequence_length"])
    train = view.split_indices("train")
    validation = view.split_indices("validation")
    train_x = torch.from_numpy(view.numeric_values[train, -sequence_length:].astype(np.float32))
    train_y = torch.from_numpy(view.target_scaled[train].astype(np.float32))
    train_mask = torch.from_numpy(view.sample_mask[train].astype(bool))
    train_available = torch.from_numpy(view.node_available[train].astype(bool))
    dataset = torch.utils.data.TensorDataset(train_x, train_y, train_mask, train_available)
    loader_rng = torch.Generator(device="cpu").manual_seed(seed + 404)
    noise_rng = torch.Generator(device="cpu").manual_seed(seed + 505)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(params["batch_size"]), shuffle=True, generator=loader_rng,
    )
    model = FixedIndustryGraphWaveNet(train_x.shape[-1], adjacency, params)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]),
    )
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    validation_x = torch.from_numpy(view.numeric_values[validation, -sequence_length:].astype(np.float32))
    validation_y = torch.from_numpy(view.target_scaled[validation].astype(np.float32))
    validation_mask = torch.from_numpy(view.sample_mask[validation].astype(bool))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    augmented_samples = 0
    original_samples = 0
    started = time.perf_counter()
    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        losses = []
        epoch_augmented = 0
        epoch_original = 0
        for batch_x, batch_y, batch_mask, batch_available in loader:
            add_count = int(math.floor(len(batch_x) * float(params["train_augmentation_ratio"])))
            if add_count:
                with torch.no_grad():
                    noise = torch.randn((*batch_x[:add_count].shape[:-1], 8), generator=noise_rng)
                    augmented_x, _ = generator(batch_x[:add_count], noise, batch_available[:add_count])
            else:
                augmented_x = batch_x[:0]
            combined_x = torch.cat([batch_x, augmented_x], dim=0) if add_count else batch_x
            combined_y = torch.cat([batch_y, batch_y[:add_count]], dim=0) if add_count else batch_y
            combined_mask = torch.cat([batch_mask, batch_mask[:add_count]], dim=0) if add_count else batch_mask
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.huber_loss(model(combined_x)[combined_mask], combined_y[combined_mask])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite F-2.2 forecaster training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(params["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            epoch_original += len(batch_x)
            epoch_augmented += add_count
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.huber_loss(
                model(validation_x)[validation_mask], validation_y[validation_mask],
            ))
        train_loss = float(np.mean(losses))
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise RuntimeError("non-finite F-2.2 forecaster train or validation loss")
        history.append({
            "epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
            "original_train_windows": epoch_original, "augmented_train_windows": epoch_augmented,
        })
        augmented_samples += epoch_augmented
        original_samples += epoch_original
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = state_copy(model)
            stale = 0
        else:
            stale += 1
        if stale >= int(params["patience"]):
            break
    if best_state is None:
        raise RuntimeError("F-2.2 forecaster did not produce checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scaled = model(validation_x).numpy()
    prediction = scaled.reshape(-1) * float(view.target_std_train) + float(view.target_mean_train)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "candidate_id": config["candidate_id"],
        "family": "stock_node_gwnet_fixed_industry",
        "state_dict": model.state_dict(),
        "parameters": dict(params),
        "input_size": int(train_x.shape[-1]),
        "adjacency": adjacency.astype(np.float32),
        "target_mean": float(view.target_mean_train),
        "target_std": float(view.target_std_train),
        "fold_id": view.fold_id,
        "seed": seed,
        "gan_generator_state_dict": generator.state_dict(),
    }, checkpoint)
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prediction.astype(float), {
        "forecaster_training_seconds": time.perf_counter() - started,
        "forecaster_epochs_completed": len(history),
        "first_train_loss": history[0]["train_loss"],
        "last_train_loss": history[-1]["train_loss"],
        "best_validation_loss": best_loss,
        "all_forecaster_losses_finite": True,
        "realized_augmentation_ratio": augmented_samples / max(original_samples, 1),
        "forecaster_parameter_count": parameter_count(model),
    }


def load_prediction_model(checkpoint: Path) -> tuple[FixedIndustryGraphWaveNet, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = FixedIndustryGraphWaveNet(
        int(payload["input_size"]), np.asarray(payload["adjacency"], dtype=np.float32), dict(payload["parameters"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict_loaded_model(model: FixedIndustryGraphWaveNet, payload: dict[str, Any], values: np.ndarray) -> np.ndarray:
    sequence_length = int(payload["parameters"]["sequence_length"])
    tensor = torch.from_numpy(values[:, -sequence_length:].astype(np.float32))
    with torch.no_grad():
        scaled = model(tensor).numpy()
    return scaled.astype(float) * float(payload["target_std"]) + float(payload["target_mean"])


def load_predict_values(checkpoint: Path, values: np.ndarray) -> np.ndarray:
    model, payload = load_prediction_model(checkpoint)
    return predict_loaded_model(model, payload, values)


def fit_observed_stress(view: E5FoldView, feature_order: list[str]) -> dict[str, float]:
    train = view.split_indices("train")
    index = {name: position for position, name in enumerate(feature_order)}
    target = view.target_raw[train]
    valid = view.sample_mask[train]
    latest = view.numeric_values[train, -1]
    drawdown = view.numeric_values[train, -4:, :, index["return_1w"]].sum(axis=1)
    return {
        "negative_return_tail_q10": float(np.quantile(target[valid], 0.10)),
        "positive_return_tail_q90": float(np.quantile(target[valid], 0.90)),
        "high_volatility_q90": float(np.quantile(latest[:, :, index["return_vol_12"]][valid], 0.90)),
        "low_liquidity_q10": float(np.quantile(latest[:, :, index["model_volume_hands"]][valid], 0.10)),
        "four_week_drawdown_q10": float(np.quantile(drawdown[valid], 0.10)),
    }


def stress_inference(
    checkpoint: Path, view: E5FoldView, seed: int, fold_index: int,
    feature_order: list[str], output_path: Path,
) -> dict[str, Any]:
    validation = view.split_indices("validation")
    values = view.numeric_values[validation].astype(np.float32)
    target = view.target_raw[validation].astype(np.float32)
    valid = view.sample_mask[validation].astype(bool)
    feature_index = {name: index for index, name in enumerate(feature_order)}
    thresholds = fit_observed_stress(view, feature_order)
    model, payload = load_prediction_model(checkpoint)
    normal = predict_loaded_model(model, payload, values)
    noise_rng = np.random.default_rng(seed + 1101 + fold_index)
    noisy = values + noise_rng.normal(0.0, 0.05, size=values.shape).astype(np.float32)
    noise_prediction = predict_loaded_model(model, payload, noisy)
    node_rng = np.random.default_rng(seed + 2201 + fold_index)
    node_indices = np.sort(node_rng.choice(values.shape[2], size=max(1, round(values.shape[2] * 0.10)), replace=False))
    node_masked = values.copy()
    node_masked[:, :, node_indices, :] = 0.0
    node_prediction = predict_loaded_model(model, payload, node_masked)
    latest_masked = values.copy()
    latest_masked[:, -1, :, :] = 0.0
    latest_prediction = predict_loaded_model(model, payload, latest_masked)
    latest = values[:, -1]
    drawdown = values[:, -4:, :, feature_index["return_1w"]].sum(axis=1)
    masks = {
        "normal_unperturbed": valid,
        "negative_return_tail_q10": valid & (target <= thresholds["negative_return_tail_q10"]),
        "positive_return_tail_q90": valid & (target >= thresholds["positive_return_tail_q90"]),
        "high_volatility_q90": valid & (latest[:, :, feature_index["return_vol_12"]] >= thresholds["high_volatility_q90"]),
        "low_liquidity_q10": valid & (latest[:, :, feature_index["model_volume_hands"]] <= thresholds["low_liquidity_q10"]),
        "four_week_drawdown_q10": valid & (drawdown <= thresholds["four_week_drawdown_q10"]),
        "feature_noise_sigma_005": valid,
        "node_mask_10pct": valid,
        "latest_week_feature_mask": valid,
    }
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    arrays: dict[str, np.ndarray] = {
        "normal_prediction": normal, "feature_noise_prediction": noise_prediction,
        "node_mask_prediction": node_prediction, "latest_week_mask_prediction": latest_prediction,
        "target_raw": target, "sample_mask": valid, "masked_node_indices": node_indices.astype(np.int64),
    }
    for name, mask in masks.items():
        arrays[f"scenario_mask__{name}"] = mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    finite = np.concatenate([
        normal.reshape(-1), noise_prediction.reshape(-1), node_prediction.reshape(-1), latest_prediction.reshape(-1),
    ])
    if not np.isfinite(finite).all():
        raise RuntimeError("non-finite F-2.2 normal or stress prediction")
    return {
        "status": "PASS", "scenario_count": len(masks), "scenario_valid_sample_counts": counts,
        "train_only_observed_thresholds": thresholds,
        "prediction_shapes": {"normal": list(normal.shape), "feature_noise": list(noise_prediction.shape),
                              "node_mask": list(node_prediction.shape), "latest_week_mask": list(latest_prediction.shape)},
        "artifact_sha256": sha256_file(output_path),
    }


def assert_frozen_keys(actual: pd.DataFrame, expected: pd.DataFrame, fold_id: str) -> str:
    columns = ["fold_id", "sample_row_id", "trade_date", "target_date", "stock_code", "sample_valid"]
    left = actual[columns].copy().astype({"stock_code": str}).sort_values(columns[:-1]).reset_index(drop=True)
    right = expected.loc[expected["fold_id"].astype(str) == fold_id, columns].copy().astype(
        {"stock_code": str}
    ).sort_values(columns[:-1]).reset_index(drop=True)
    if len(left) != 500 or len(right) != 500 or not left.equals(right):
        raise RuntimeError(f"{fold_id} validation sample keys differ from frozen E-5 keys")
    return stable_json_sha256(left.to_dict(orient="records"))


def run(config_path: Path, overwrite: bool = False) -> Path:
    config, correction = load_effective_config(config_path)
    if config["seed"] != 20260725 or config["folds"] != ["E_RO_01", "E_RO_02", "E_RO_03"]:
        raise ValueError("F-2.2 seed or folds changed")
    for source in config["source_registry"].values():
        if sha256_file(resolve(source["path"])) != source["sha256"]:
            raise RuntimeError(f"F-2.2 source hash mismatch: {source['path']}")
    for fold_id, item in config["frozen_forecaster_checkpoints"].items():
        if sha256_file(resolve(item["path"])) != item["sha256"]:
            raise RuntimeError(f"F-2.2 frozen forecaster hash mismatch: {fold_id}")
    guard = StageFDataCustodyGuard.from_config(resolve(config["paths"]["custody_config"]), REPO_ROOT)
    guarded = [resolve(config["paths"][key]) for key in ("adapter_root", "universe_path", "frozen_validation_keys")]
    guard.assert_paths_allowed(guarded, "f2_2_single_seed_engineering")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    expected_keys = pd.read_csv(resolve(config["paths"]["frozen_validation_keys"]), dtype={"stock_code": str})
    receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    total_started = time.perf_counter()
    for fold_index, fold_id in enumerate(config["folds"]):
        run_dir = output_root / "runs" / f"{fold_id}__{config['candidate_id']}__seed{config['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        fold_started = time.perf_counter()
        try:
            view = load_fold_view(resolve(config["paths"]["adapter_root"]), fold_id, "no_text")
            guard.assert_development_dates(view.trade_date, f"{fold_id} trade_date")
            guard.assert_development_dates(view.target_date.reshape(-1), f"{fold_id} target_date")
            key_frame = validation_key_frame(view)
            key_sha = assert_frozen_keys(key_frame, expected_keys, fold_id)
            adjacency, industries = fixed_industry_adjacency(view.stock_code, universe)
            frozen_path = resolve(config["frozen_forecaster_checkpoints"][fold_id]["path"])
            frozen_forecaster, _ = load_frozen_forecaster(frozen_path)
            generator, critic, gan_detail = train_gan(
                view, frozen_forecaster, config, int(config["seed"]), run_dir / "gan_training_log.json",
            )
            gan_checkpoint = run_dir / "gan_final_epoch.pt"
            torch.save({
                "candidate_id": config["candidate_id"], "fold_id": fold_id, "seed": config["seed"],
                "generator_state_dict": generator.state_dict(), "critic_state_dict": critic.state_dict(),
                "generator_constructor": {"feature_count": 6, "noise_dim": 8, "hidden_channels": 32, "max_delta": 0.05},
                "critic_constructor": {"feature_count": 6, "hidden_channels": 32},
            }, gan_checkpoint)
            forecaster_checkpoint = run_dir / "forecaster.pt"
            prediction, forecaster_detail = train_augmented_forecaster(
                view, adjacency, generator, config, int(config["seed"]), forecaster_checkpoint,
                run_dir / "forecaster_training_log.json",
            )
            loaded = load_predict_values(
                forecaster_checkpoint, view.numeric_values[view.split_indices("validation")],
            ).reshape(-1)
            load_difference = float(np.max(np.abs(prediction - loaded)))
            if load_difference > float(config["engineering_gates"]["independent_loading_max_abs_difference"]):
                raise RuntimeError(f"independent load mismatch: {load_difference}")
            stress = stress_inference(
                forecaster_checkpoint, view, int(config["seed"]), fold_index, list(config["feature_order"]),
                run_dir / "normal_and_stress_predictions.npz",
            )
            duration = time.perf_counter() - fold_started
            if duration > float(config["engineering_gates"]["maximum_fold_duration_seconds"]):
                raise RuntimeError(f"F-2.2 fold duration cost hard failure: {duration}")
            rows = key_frame.copy()
            rows.insert(0, "seed", config["seed"])
            rows.insert(0, "candidate_id", config["candidate_id"])
            rows["prediction"] = prediction
            predictions.append(rows)
            receipt = {
                "candidate_id": config["candidate_id"], "fold_id": fold_id, "seed": config["seed"], "status": "PASS",
                "validation_rows": len(rows), "validation_sample_key_sha256": key_sha,
                "stock_order_sha256": stable_json_sha256(view.stock_code.astype(str).tolist()),
                "adjacency_sha256": stable_json_sha256(adjacency.tolist()), "industry_count": len(set(industries)),
                "frozen_forecaster_sha256": sha256_file(frozen_path),
                "gan_checkpoint_sha256": sha256_file(gan_checkpoint),
                "forecaster_checkpoint_sha256": sha256_file(forecaster_checkpoint),
                "independent_load_max_abs_difference": load_difference,
                "stress_inference": stress, "duration_seconds": duration, **gan_detail, **forecaster_detail,
            }
            receipts.append(receipt)
            (run_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{fold_id} PASS duration={duration:.3f}s load_diff={load_difference:.3g}", flush=True)
        except Exception as exc:
            failure = {
                "candidate_id": config["candidate_id"], "fold_id": fold_id, "seed": config["seed"], "status": "FAIL",
                "error_type": type(exc).__name__, "error": str(exc),
                "collapse_conditions": exc.conditions if isinstance(exc, GanCollapse) else [],
                "duration_seconds": time.perf_counter() - fold_started,
            }
            failures.append(failure)
            (run_dir / "failure_receipt.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            print(f"{fold_id} FAIL {type(exc).__name__}: {exc}", flush=True)
    receipts_path = output_root / "engineering_receipts.json"
    failures_path = output_root / "failure_receipts.json"
    receipts_path.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions_path = output_root / "unified_predictions.csv.gz"
    frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not frame.empty:
        frame.to_csv(predictions_path, index=False, compression={"method": "gzip", "mtime": 0})
    scenario_names = sorted({
        name for receipt in receipts for name in receipt["stress_inference"]["scenario_valid_sample_counts"]
    })
    pooled_counts = {
        name: sum(int(receipt["stress_inference"]["scenario_valid_sample_counts"].get(name, 0)) for receipt in receipts)
        for name in scenario_names
    }
    total_duration = time.perf_counter() - total_started
    pooled_stress_pass = len(pooled_counts) == 9 and all(count > 0 for count in pooled_counts.values())
    passed = (
        len(receipts) == int(config["engineering_gates"]["required_fold_runs"])
        and not failures and pooled_stress_pass
        and all(receipt["collapse_conditions_pass"] for receipt in receipts)
        and total_duration <= float(config["engineering_gates"]["maximum_total_three_fold_duration_seconds"])
    )
    metadata = {
        "stage": "F-2.2 single-seed three-fold GAN engineering receipts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "candidate_id": config["candidate_id"], "seed": config["seed"], "folds": config["folds"],
        "expected_run_count": 3, "completed_run_count": len(receipts), "failure_count": len(failures),
        "all_gan_losses_finite": bool(receipts) and all(r["all_gan_losses_finite"] for r in receipts),
        "all_forecaster_losses_finite": bool(receipts) and all(r["all_forecaster_losses_finite"] for r in receipts),
        "all_collapse_conditions_pass": bool(receipts) and all(r["collapse_conditions_pass"] for r in receipts),
        "all_independent_loads_pass": bool(receipts) and all(
            r["independent_load_max_abs_difference"] <= config["engineering_gates"]["independent_loading_max_abs_difference"]
            for r in receipts
        ),
        "all_frozen_validation_keys_pass": bool(receipts) and all(r["validation_rows"] == 500 for r in receipts),
        "all_normal_and_stress_entries_pass": pooled_stress_pass,
        "pooled_three_fold_stress_counts": pooled_counts,
        "total_duration_seconds": total_duration,
        "cost_limit_pass": total_duration <= config["engineering_gates"]["maximum_total_three_fold_duration_seconds"],
        "ranking_performed": False, "candidate_deletion_performed": False,
        "promotion_recommendation_formed": False, "additional_seed_executed": False,
        "screening_accessed": False, "final_accessed": False,
        "adapter_correction": correction,
        "config_sha256": sha256_file(config_path),
        "artifacts": {
            "engineering_receipts_sha256": sha256_file(receipts_path),
            "failure_receipts_sha256": sha256_file(failures_path),
            "unified_predictions_sha256": sha256_file(predictions_path) if predictions_path.is_file() else None,
        },
        "next_action": config["next_action_if_pass"] if passed else config["next_action_if_fail"],
    }
    metadata["batch_sha256"] = stable_json_sha256(metadata)
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("F-2.2 single-seed engineering did not pass all frozen checks")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
