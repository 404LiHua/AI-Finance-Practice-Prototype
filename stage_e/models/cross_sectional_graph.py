from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossSectionalGraphLearner(nn.Module):
    """Sparse graph learner over stock nodes with output shape [batch, stock, stock]."""

    def __init__(
        self,
        hidden_dim: int,
        top_k: int = 8,
        temperature: float = 1.0,
        keep_self_loops: bool = True,
        sampling_mode: str = "deterministic",
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or top_k < 1 or temperature <= 0:
            raise ValueError("hidden_dim, top_k and temperature must be positive")
        if sampling_mode not in {"deterministic", "gumbel"}:
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

    def forward(self, values: torch.Tensor, node_available: torch.Tensor | None = None) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, stock, hidden]")
        batch_size, stock_count, _ = values.shape
        maximum = stock_count - 1
        if self.top_k > maximum:
            raise ValueError("top_k exceeds available stock neighbours")
        if node_available is None:
            node_available = torch.ones((batch_size, stock_count), dtype=torch.bool, device=values.device)
        if node_available.shape != (batch_size, stock_count):
            raise ValueError("node_available must have shape [batch, stock]")
        node_available = node_available.to(dtype=torch.bool, device=values.device)
        logits = torch.matmul(self.query(values), self.key(values).transpose(-1, -2)) / self.scale
        key_mask = node_available.unsqueeze(1).expand(-1, stock_count, -1)
        logits = logits.masked_fill(~key_mask, torch.finfo(logits.dtype).min)
        diagonal = torch.eye(stock_count, dtype=torch.bool, device=values.device).unsqueeze(0)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        if self.training and self.sampling_mode == "gumbel":
            probabilities = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        else:
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
        probabilities = probabilities * key_mask.to(probabilities.dtype)
        top_values, top_indices = torch.topk(probabilities, k=self.top_k, dim=-1)
        sparse = torch.zeros_like(probabilities).scatter(-1, top_indices, top_values)
        sparse = sparse * node_available.unsqueeze(-1).to(sparse.dtype)
        if self.keep_self_loops:
            sparse = sparse + diagonal.to(sparse.dtype) * node_available.unsqueeze(-1).to(sparse.dtype)
        row_sums = sparse.sum(dim=-1, keepdim=True)
        sparse = sparse / row_sums.clamp_min(1e-8)
        unavailable = ~node_available
        if unavailable.any():
            identity = torch.eye(stock_count, dtype=sparse.dtype, device=sparse.device).unsqueeze(0)
            sparse = torch.where(unavailable.unsqueeze(-1), identity.expand(batch_size, -1, -1), sparse)
        return sparse


class CrossSectionalGraphBlock(nn.Module):
    """Residual message passing over stock nodes."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or adjacency.ndim != 3:
            raise ValueError("expected values [B,N,D] and adjacency [B,N,N]")
        if values.shape[:2] != adjacency.shape[:2] or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("stock dimensions do not match")
        propagated = torch.matmul(adjacency, values)
        update = self.dropout(self.activation(self.message(propagated)))
        return self.norm(values + update)
