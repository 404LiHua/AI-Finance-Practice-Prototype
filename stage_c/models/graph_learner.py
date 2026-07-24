from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DynamicGraphLearner(nn.Module):
    """Learn a sparse row-stochastic graph over the sequence positions."""

    def __init__(
        self,
        hidden_dim: int,
        top_k: int = 2,
        temperature: float = 0.8,
        keep_self_loops: bool = True,
        sampling_mode: str = "gumbel",
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sampling_mode not in {"gumbel", "deterministic"}:
            raise ValueError(f"unsupported sampling_mode: {sampling_mode}")
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.keep_self_loops = bool(keep_self_loops)
        self.sampling_mode = sampling_mode
        self.scale = math.sqrt(hidden_dim)

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, sequence, hidden]")
        sequence_length = values.shape[1]
        maximum_edges = sequence_length if self.keep_self_loops else sequence_length - 1
        if self.top_k > maximum_edges:
            raise ValueError("top_k exceeds the number of available neighbours")

        logits = torch.matmul(self.query(values), self.key(values).transpose(-1, -2)) / self.scale
        if not self.keep_self_loops:
            diagonal = torch.eye(sequence_length, device=values.device, dtype=torch.bool)
            logits = logits.masked_fill(diagonal.unsqueeze(0), torch.finfo(logits.dtype).min)

        if self.training and self.sampling_mode == "gumbel":
            probabilities = F.gumbel_softmax(
                logits, tau=self.temperature, hard=False, dim=-1,
            )
        else:
            probabilities = torch.softmax(logits / self.temperature, dim=-1)

        top_values, top_indices = torch.topk(probabilities, k=self.top_k, dim=-1)
        sparse = torch.zeros_like(probabilities).scatter(-1, top_indices, top_values)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
