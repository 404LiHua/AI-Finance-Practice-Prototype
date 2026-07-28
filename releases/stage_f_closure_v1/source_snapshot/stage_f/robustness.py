from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from stage_e.e5.neural_graph import build_model


TAIL_WEIGHTED_ID = "stock_node_gwnet_tail_weighted_l8"
NOISE_AUGMENTED_ID = "stock_node_gwnet_noise_aug_l8"
FEATURE_MASKED_ID = "stock_node_gwnet_feature_mask_l8"
F1_CANDIDATE_IDS = (TAIL_WEIGHTED_ID, NOISE_AUGMENTED_ID, FEATURE_MASKED_ID)

TAIL_QUANTILE = 0.90
TAIL_WEIGHT = 2.0
NOISE_SIGMA = 0.03
FEATURE_MASK_PROBABILITY = 0.05


def validate_candidate_id(candidate_id: str) -> str:
    value = str(candidate_id)
    if value not in F1_CANDIDATE_IDS:
        raise ValueError(f"candidate is outside the frozen F-1 set: {value}")
    return value


def fit_train_tail_threshold(
    target_raw: np.ndarray | torch.Tensor,
    sample_mask: np.ndarray | torch.Tensor,
    split_name: str,
) -> float:
    """Fit the frozen absolute-target Q90 threshold using TRAIN observations only."""
    if str(split_name).casefold() != "train":
        raise ValueError("tail threshold fitting is restricted to TRAIN")
    target = torch.as_tensor(target_raw, dtype=torch.float64)
    mask = torch.as_tensor(sample_mask, dtype=torch.bool)
    if target.shape != mask.shape:
        raise ValueError("target_raw and sample_mask must share shape")
    valid = target[mask]
    if valid.numel() == 0 or not torch.isfinite(valid).all():
        raise ValueError("TRAIN tail threshold requires finite valid targets")
    return float(torch.quantile(valid.abs(), TAIL_QUANTILE))


def weighted_masked_huber(
    prediction: torch.Tensor,
    target_scaled: torch.Tensor,
    target_raw: torch.Tensor,
    sample_mask: torch.Tensor,
    tail_threshold_raw: float,
) -> torch.Tensor:
    if prediction.shape != target_scaled.shape or prediction.shape != target_raw.shape:
        raise ValueError("prediction, target_scaled and target_raw must share shape")
    if prediction.shape != sample_mask.shape:
        raise ValueError("sample_mask must share the prediction shape")
    mask = sample_mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("masked Huber loss requires at least one valid observation")
    elementwise = nn.functional.huber_loss(prediction, target_scaled, reduction="none")
    tail = target_raw.abs() >= float(tail_threshold_raw)
    weights = torch.where(tail, torch.full_like(elementwise, TAIL_WEIGHT), torch.ones_like(elementwise))
    return (elementwise[mask] * weights[mask]).sum() / weights[mask].sum()


@dataclass
class F1CandidateStrategy:
    """The only three training-time changes authorized by the frozen F-0 protocol."""

    candidate_id: str
    seed: int
    tail_threshold_raw: float | None = None

    def __post_init__(self) -> None:
        self.candidate_id = validate_candidate_id(self.candidate_id)
        self.seed = int(self.seed)
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def fit(self, target_raw: np.ndarray | torch.Tensor, sample_mask: np.ndarray | torch.Tensor, split_name: str) -> None:
        if self.candidate_id == TAIL_WEIGHTED_ID:
            self.tail_threshold_raw = fit_train_tail_threshold(target_raw, sample_mask, split_name)
        elif str(split_name).casefold() != "train":
            raise ValueError("F-1 strategy fitting is restricted to TRAIN")

    def transform(self, values: torch.Tensor, training: bool) -> torch.Tensor:
        if not torch.is_floating_point(values):
            raise TypeError("numeric values must be a floating-point tensor")
        if not training or self.candidate_id == TAIL_WEIGHTED_ID:
            return values
        if self.candidate_id == NOISE_AUGMENTED_ID:
            noise = torch.randn(values.shape, generator=self._generator, dtype=values.dtype, device="cpu")
            return values + noise.to(values.device) * NOISE_SIGMA
        keep = torch.rand(values.shape, generator=self._generator, dtype=values.dtype, device="cpu")
        return values * (keep.to(values.device) >= FEATURE_MASK_PROBABILITY).to(values.dtype)

    def loss(
        self,
        prediction: torch.Tensor,
        target_scaled: torch.Tensor,
        target_raw: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.candidate_id == TAIL_WEIGHTED_ID:
            if self.tail_threshold_raw is None:
                raise RuntimeError("TRAIN tail threshold must be fitted before computing loss")
            return weighted_masked_huber(
                prediction, target_scaled, target_raw, sample_mask, self.tail_threshold_raw,
            )
        mask = sample_mask.to(dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("masked Huber loss requires at least one valid observation")
        return nn.functional.huber_loss(prediction[mask], target_scaled[mask])


def build_f1_candidate_model(
    candidate_id: str,
    input_size: int,
    adjacency: np.ndarray,
    base_parameters: dict[str, Any],
) -> nn.Module:
    validate_candidate_id(candidate_id)
    if int(base_parameters.get("sequence_length", -1)) != 8:
        raise ValueError("F-1 candidates must retain the frozen L8 base architecture")
    return build_model(
        "stock_node_gwnet_fixed_industry",
        int(input_size),
        int(base_parameters["sequence_length"]),
        dict(base_parameters),
        adjacency,
    )


def candidate_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    strategy: F1CandidateStrategy,
    values: torch.Tensor,
    target_scaled: torch.Tensor,
    target_raw: torch.Tensor,
    sample_mask: torch.Tensor,
    gradient_clip: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    transformed = strategy.transform(values, training=True)
    prediction = model(transformed)
    if prediction.shape != target_scaled.shape:
        raise ValueError("candidate prediction shape differs from the frozen target shape")
    loss = strategy.loss(prediction, target_scaled, target_raw, sample_mask)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite F-1 candidate loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
    optimizer.step()
    return float(loss.detach())
