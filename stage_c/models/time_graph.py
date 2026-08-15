from __future__ import annotations

import torch
from torch import nn


class TimeGraphBlock(nn.Module):
    """Deterministic graph message passing directly in the time domain."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.matmul(adjacency, values)
        propagated = self.dropout(self.activation(self.message(propagated)))
        return self.norm(propagated)

