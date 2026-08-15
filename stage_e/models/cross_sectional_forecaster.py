from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stage_e.models.cross_sectional_graph import CrossSectionalGraphBlock, CrossSectionalGraphLearner


class CrossSectionalTemporalForecaster(nn.Module):
    """Per-stock temporal encoder followed by a stock cross-sectional graph."""

    def __init__(
        self,
        input_dim: int,
        stock_count: int,
        hidden_dim: int = 48,
        top_k: int = 8,
        dropout: float = 0.0,
        sampling_mode: str = "deterministic",
    ) -> None:
        super().__init__()
        self.stock_count = int(stock_count)
        self.temporal_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.stock_embedding = nn.Parameter(torch.zeros(1, stock_count, hidden_dim))
        self.pre_graph_norm = nn.LayerNorm(hidden_dim)
        self.graph_learner = CrossSectionalGraphLearner(
            hidden_dim=hidden_dim, top_k=top_k, keep_self_loops=True, sampling_mode=sampling_mode,
        )
        self.graph_block = CrossSectionalGraphBlock(hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.stock_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        node_available: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> Any:
        if values.ndim != 4:
            raise ValueError("values must have shape [batch, time, stock, feature]")
        batch_size, time_steps, stock_count, feature_count = values.shape
        if stock_count != self.stock_count:
            raise ValueError("stock dimension differs from the frozen model stock count")
        sequences = values.permute(0, 2, 1, 3).reshape(batch_size * stock_count, time_steps, feature_count)
        _, hidden = self.temporal_encoder(sequences)
        nodes = hidden[-1].reshape(batch_size, stock_count, -1)
        nodes = self.pre_graph_norm(nodes + self.stock_embedding)
        adjacency = self.graph_learner(nodes, node_available=node_available)
        graph_nodes = self.graph_block(nodes, adjacency)
        prediction = self.head(graph_nodes).squeeze(-1)
        if return_details:
            return {"prediction": prediction, "adjacency": adjacency, "temporal_nodes": nodes, "graph_nodes": graph_nodes}
        return prediction
