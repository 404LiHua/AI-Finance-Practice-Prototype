from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import spectral_norm


def deterministic_noise(shape: tuple[int, ...], seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=dtype, device="cpu")


class BoundedConditionalGenerator(nn.Module):
    """Shared temporal generator producing bounded residuals for real stock-node windows."""

    def __init__(self, feature_count: int = 6, noise_dim: int = 8, hidden_channels: int = 32, max_delta: float = 0.05) -> None:
        super().__init__()
        self.feature_count = int(feature_count)
        self.noise_dim = int(noise_dim)
        self.max_delta = float(max_delta)
        self.input_projection = nn.Conv1d(self.feature_count + self.noise_dim, hidden_channels, 1)
        self.norm1 = nn.GroupNorm(1, hidden_channels)
        self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.output_projection = nn.Conv1d(hidden_channels, self.feature_count, 1)

    def forward(
        self,
        values: torch.Tensor,
        noise: torch.Tensor,
        node_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 4:
            raise ValueError("generator values must be [batch,time,stock,feature]")
        if noise.shape != (*values.shape[:-1], self.noise_dim):
            raise ValueError("generator noise shape differs from frozen contract")
        if values.shape[-1] != self.feature_count:
            raise ValueError("generator feature count differs from frozen contract")
        batch, length, stocks, _ = values.shape
        joined = torch.cat([values, noise.to(values.device, values.dtype)], dim=-1)
        channels = joined.permute(0, 2, 3, 1).reshape(batch * stocks, -1, length)
        hidden = torch.nn.functional.silu(self.norm1(self.input_projection(channels)))
        residual = torch.nn.functional.silu(self.norm2(self.conv1(hidden)))
        hidden = hidden + self.conv2(residual)
        delta = torch.tanh(self.output_projection(torch.nn.functional.silu(hidden))) * self.max_delta
        delta = delta.reshape(batch, stocks, self.feature_count, length).permute(0, 3, 1, 2)
        if node_available is not None:
            if node_available.shape == (batch, stocks):
                availability = node_available[:, None, :, None]
            elif node_available.shape == (batch, length, stocks):
                availability = node_available[:, :, :, None]
            else:
                raise ValueError("node availability must be [batch,stock] or [batch,time,stock]")
            delta = delta * availability.to(delta.device, delta.dtype)
        return values + delta, delta


class SpectralTemporalCritic(nn.Module):
    """Spectrally normalized critic with shared temporal encoding and masked stock pooling."""

    def __init__(self, feature_count: int = 6, hidden_channels: int = 32) -> None:
        super().__init__()
        self.feature_count = int(feature_count)
        self.input_projection = spectral_norm(nn.Conv1d(self.feature_count, hidden_channels, 1))
        self.conv1 = spectral_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1))
        self.head = spectral_norm(nn.Linear(hidden_channels, 1))

    def forward(self, values: torch.Tensor, node_available: torch.Tensor | None = None) -> torch.Tensor:
        if values.ndim != 4 or values.shape[-1] != self.feature_count:
            raise ValueError("critic values must follow [batch,time,stock,feature]")
        batch, length, stocks, features = values.shape
        channels = values.permute(0, 2, 3, 1).reshape(batch * stocks, features, length)
        hidden = torch.nn.functional.leaky_relu(self.input_projection(channels), negative_slope=0.2)
        hidden = torch.nn.functional.leaky_relu(self.conv1(hidden), negative_slope=0.2)
        hidden = torch.nn.functional.leaky_relu(self.conv2(hidden) + hidden, negative_slope=0.2)
        node_hidden = hidden.mean(dim=-1).reshape(batch, stocks, -1)
        if node_available is None:
            pooled = node_hidden.mean(dim=1)
        else:
            availability = node_available[:, -1] if node_available.ndim == 3 else node_available
            if availability.shape != (batch, stocks):
                raise ValueError("critic node availability has invalid shape")
            weights = availability.to(node_hidden.device, node_hidden.dtype).unsqueeze(-1)
            pooled = (node_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


def gradient_penalty(
    critic: SpectralTemporalCritic,
    real_values: torch.Tensor,
    fake_values: torch.Tensor,
    interpolation: torch.Tensor,
    node_available: torch.Tensor | None = None,
) -> torch.Tensor:
    if interpolation.shape != (real_values.shape[0], 1, 1, 1):
        raise ValueError("gradient-penalty interpolation must be [batch,1,1,1]")
    mixed = interpolation.to(real_values.device, real_values.dtype) * real_values
    mixed = mixed + (1.0 - interpolation.to(real_values.device, real_values.dtype)) * fake_values
    mixed.requires_grad_(True)
    score = critic(mixed, node_available)
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm = gradients.reshape(len(gradients), -1).norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


def critic_wgan_gp_loss(
    real_score: torch.Tensor,
    fake_score: torch.Tensor,
    penalty: torch.Tensor,
    gradient_penalty_weight: float = 10.0,
    drift_weight: float = 0.001,
) -> torch.Tensor:
    return fake_score.mean() - real_score.mean() + float(gradient_penalty_weight) * penalty + float(drift_weight) * (real_score ** 2).mean()


@dataclass(frozen=True)
class GeneratorLossDetail:
    total: torch.Tensor
    realism: torch.Tensor
    hardness: torch.Tensor
    energy: torch.Tensor


def generator_adversarial_loss(
    fake_score: torch.Tensor,
    real_forecaster_loss: torch.Tensor,
    fake_forecaster_loss: torch.Tensor,
    delta: torch.Tensor,
    hardness_weight: float = 0.5,
    energy_weight: float = 2.0,
    target_mean_absolute_delta: float = 0.02,
    maximum_hardness_gain: float = 0.05,
) -> GeneratorLossDetail:
    realism = -fake_score.mean()
    gain = (fake_forecaster_loss - real_forecaster_loss).clamp(min=-maximum_hardness_gain, max=maximum_hardness_gain)
    hardness = -gain.mean()
    energy = (delta.abs().mean() - float(target_mean_absolute_delta)) ** 2
    total = realism + float(hardness_weight) * hardness + float(energy_weight) * energy
    return GeneratorLossDetail(total=total, realism=realism, hardness=hardness, energy=energy)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
