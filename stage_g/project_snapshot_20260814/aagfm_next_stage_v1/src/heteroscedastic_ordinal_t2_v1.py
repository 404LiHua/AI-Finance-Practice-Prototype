from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RobustScaler:
    median: np.ndarray
    scale: np.ndarray
    clip: float = 8.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.median.size:
            raise ValueError("feature shape mismatch")
        transformed = (array - self.median) / self.scale
        transformed[~np.isfinite(transformed)] = 0.0
        return np.clip(transformed, -self.clip, self.clip)


def fit_robust_scaler(values: np.ndarray, clip: float = 8.0) -> RobustScaler:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("values must be nonempty 2d")
    median = np.nanmedian(array, axis=0)
    scale = np.nanquantile(array, 0.75, axis=0) - np.nanquantile(array, 0.25, axis=0)
    median[~np.isfinite(median)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return RobustScaler(median=median, scale=scale, clip=float(clip))


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-5, 1.0 - 1e-5))
    return float(np.log(value / (1.0 - value)))


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(max(float(value), 1e-6))))


class HeteroscedasticProportionalOdds(nn.Module):
    def __init__(self, location_features: int, scale_features: int, initial_thresholds: tuple[float, float]) -> None:
        super().__init__()
        if location_features <= 0 or scale_features <= 0 or initial_thresholds[1] <= initial_thresholds[0]:
            raise ValueError("invalid model configuration")
        self.beta = nn.Parameter(torch.zeros(location_features, dtype=torch.float64))
        self.gamma = nn.Parameter(torch.zeros(scale_features, dtype=torch.float64))
        self.threshold0 = nn.Parameter(torch.tensor(initial_thresholds[0], dtype=torch.float64))
        self.raw_gap = nn.Parameter(torch.tensor(_inverse_softplus(initial_thresholds[1] - initial_thresholds[0] - 1e-3), dtype=torch.float64))

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        first = self.threshold0
        return first, first + F.softplus(self.raw_gap) + 1e-3

    def forward(self, location: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if location.ndim != 2 or scale.ndim != 2 or location.shape[0] != scale.shape[0]:
            raise ValueError("row alignment failure")
        eta = location.to(torch.float64) @ self.beta
        # tanh imposes the frozen closed interval [-1, +1] before exponentiation.
        log_scale = torch.tanh(scale.to(torch.float64) @ self.gamma)
        sigma = torch.exp(log_scale)
        first, second = self.thresholds()
        cumulative0 = torch.sigmoid((first - eta) / sigma)
        cumulative1 = torch.sigmoid((second - eta) / sigma)
        probability = torch.stack([cumulative0, cumulative1 - cumulative0, 1.0 - cumulative1], dim=1)
        probability = probability.clamp_min(1e-12)
        return probability / probability.sum(dim=1, keepdim=True)


@dataclass(frozen=True)
class FittedHeteroscedasticProportionalOdds:
    beta: np.ndarray
    gamma: np.ndarray
    thresholds: tuple[float, float]
    location_scaler: RobustScaler
    scale_scaler: RobustScaler

    def predict_proba(self, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
        x = self.location_scaler.transform(location)
        z = self.scale_scaler.transform(scale)
        eta = x @ self.beta
        sigma = np.exp(np.tanh(z @ self.gamma))
        c0 = 1.0 / (1.0 + np.exp(-(self.thresholds[0] - eta) / sigma))
        c1 = 1.0 / (1.0 + np.exp(-(self.thresholds[1] - eta) / sigma))
        probability = np.column_stack([c0, c1 - c0, 1.0 - c1])
        probability = np.clip(probability, 1e-12, None)
        return probability / probability.sum(axis=1, keepdims=True)


def fit_heteroscedastic_proportional_odds(
    location: np.ndarray,
    scale: np.ndarray,
    target: np.ndarray,
    location_l2: float = 1e-3,
    scale_l2: float = 1e-2,
    max_iter: int = 200,
) -> FittedHeteroscedasticProportionalOdds:
    y = np.asarray(target, dtype=np.int64).reshape(-1)
    x_raw = np.asarray(location, dtype=np.float64)
    z_raw = np.asarray(scale, dtype=np.float64)
    if x_raw.ndim != 2 or z_raw.ndim != 2 or len(y) != len(x_raw) or len(y) != len(z_raw) or len(y) == 0:
        raise ValueError("location, scale and target must align")
    if np.any((y < 0) | (y > 2)):
        raise ValueError("target classes must be 0, 1, 2")
    location_scaler = fit_robust_scaler(x_raw)
    scale_scaler = fit_robust_scaler(z_raw)
    x = torch.as_tensor(location_scaler.transform(x_raw), dtype=torch.float64)
    z = torch.as_tensor(scale_scaler.transform(z_raw), dtype=torch.float64)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    count = np.bincount(y, minlength=3).astype(float) + 1.0
    prior = count / count.sum()
    model = HeteroscedasticProportionalOdds(x.shape[1], z.shape[1], (_logit(prior[0]), _logit(prior[0] + prior[1])))
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=int(max_iter), tolerance_grad=1e-8, tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        probability = model(x, z)
        nll = -torch.log(probability[torch.arange(len(y)), y_tensor]).mean()
        loss = nll + float(location_l2) * model.beta.square().mean() + float(scale_l2) * model.gamma.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    first, second = model.thresholds()
    return FittedHeteroscedasticProportionalOdds(
        beta=model.beta.detach().cpu().numpy().copy(),
        gamma=model.gamma.detach().cpu().numpy().copy(),
        thresholds=(float(first.detach()), float(second.detach())),
        location_scaler=location_scaler,
        scale_scaler=scale_scaler,
    )


