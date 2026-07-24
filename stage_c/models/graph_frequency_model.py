from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stage_c.models.frequency_graph import FrequencyGraphBlock
from stage_c.models.graph_learner import DynamicGraphLearner
from stage_c.models.time_graph import TimeGraphBlock


class GraphFrequencyModel(nn.Module):
    """Minimal Stage C model: temporal encoder + learned sparse graph + FFT branch."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 32,
        nhead: int = 4,
        num_temporal_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        top_k: int = 2,
        gumbel_temperature: float = 0.8,
        keep_self_loops: bool = True,
        graph_mode: str = "learned",
        use_frequency: bool = True,
        fusion_mode: str = "gated",
        sampling_mode: str = "gumbel",
        use_time_graph: bool = False,
    ) -> None:
        super().__init__()
        if graph_mode not in {"learned", "identity", "temporal_neighbor"}:
            raise ValueError(f"unsupported graph_mode: {graph_mode}")
        if fusion_mode not in {"gated", "mean"}:
            raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
        self.graph_mode = graph_mode
        self.use_frequency = bool(use_frequency)
        self.use_time_graph = bool(use_time_graph)
        if self.use_frequency and self.use_time_graph:
            raise ValueError("frequency and time-graph propagation are mutually exclusive in v1/v2")
        self.fusion_mode = fusion_mode
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=num_temporal_layers, enable_nested_tensor=False,
        )
        self.graph_learner = None
        if graph_mode == "learned" and (self.use_frequency or self.use_time_graph):
            self.graph_learner = DynamicGraphLearner(
                hidden_dim=hidden_dim,
                top_k=top_k,
                temperature=gumbel_temperature,
                keep_self_loops=keep_self_loops,
                sampling_mode=sampling_mode,
            )
        self.frequency_graph = FrequencyGraphBlock(hidden_dim, dropout=dropout) if self.use_frequency else None
        self.time_graph = TimeGraphBlock(hidden_dim, dropout=dropout) if self.use_time_graph else None
        self.gate = (
            nn.Linear(hidden_dim * 2, hidden_dim)
            if (self.use_frequency or self.use_time_graph) and fusion_mode == "gated" else None
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def _fixed_adjacency(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = values.shape[:2]
        adjacency = torch.eye(sequence_length, device=values.device, dtype=values.dtype)
        if self.graph_mode == "temporal_neighbor" and sequence_length > 1:
            rows = torch.arange(1, sequence_length, device=values.device)
            adjacency[rows, rows - 1] = 1.0
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True)
        return adjacency.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, values: torch.Tensor, return_details: bool = False) -> Any:
        projected = self.input_projection(values) + self.position[:, : values.shape[1]]
        temporal = self.temporal_encoder(projected)
        if self.graph_learner is None:
            adjacency = self._fixed_adjacency(temporal)
        else:
            adjacency = self.graph_learner(temporal)
        temporal_last = temporal[:, -1]
        if self.time_graph is not None:
            graph_values = self.time_graph(temporal, adjacency)
            graph_last = graph_values[:, -1]
            if self.gate is None:
                gate = torch.full_like(temporal_last, 0.5)
            else:
                gate = torch.sigmoid(self.gate(torch.cat([temporal_last, graph_last], dim=-1)))
            fused = gate * temporal_last + (1.0 - gate) * graph_last
            frequency = graph_values
        elif self.frequency_graph is None:
            frequency = temporal
            gate = torch.ones_like(temporal_last)
            fused = temporal_last
        else:
            frequency = self.frequency_graph(temporal, adjacency)
            frequency_last = frequency[:, -1]
            if self.gate is None:
                gate = torch.full_like(temporal_last, 0.5)
            else:
                gate = torch.sigmoid(self.gate(torch.cat([temporal_last, frequency_last], dim=-1)))
            fused = gate * temporal_last + (1.0 - gate) * frequency_last
        prediction = self.head(self.output_norm(fused)).squeeze(-1)
        if return_details:
            return {
                "prediction": prediction,
                "adjacency": adjacency,
                "gate": gate,
                "frequency": frequency,
            }
        return prediction
