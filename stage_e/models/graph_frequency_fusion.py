from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stage_e.models.cross_sectional_graph import CrossSectionalGraphLearner


class CrossSectionalTimeGraphBlock(nn.Module):
    """Stock-node message passing at every temporal position."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or adjacency.ndim != 3:
            raise ValueError("expected values [B,T,N,D] and adjacency [B,N,N]")
        if values.shape[0] != adjacency.shape[0] or values.shape[2] != adjacency.shape[1] or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("batch or stock dimensions do not match")
        propagated = torch.einsum("bij,btjd->btid", adjacency, values)
        update = self.dropout(self.activation(self.message(propagated)))
        return self.norm(values + update)


class CrossSectionalFrequencyGraphBlock(nn.Module):
    """FFT over time, graph propagation over stocks, then inverse FFT."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.real_linear = nn.Linear(hidden_dim, hidden_dim)
        self.imag_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or adjacency.ndim != 3:
            raise ValueError("expected values [B,T,N,D] and adjacency [B,N,N]")
        if values.shape[0] != adjacency.shape[0] or values.shape[2] != adjacency.shape[1] or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("batch or stock dimensions do not match")
        time_steps = values.shape[1]
        spectrum = torch.fft.rfft(values, dim=1, norm="ortho")
        real = torch.einsum("bij,bfjd->bfid", adjacency, spectrum.real)
        imag = torch.einsum("bij,bfjd->bfid", adjacency, spectrum.imag)
        real_output = self.real_linear(real) - self.imag_linear(imag)
        imag_output = self.imag_linear(real) + self.real_linear(imag)
        reconstructed = torch.fft.irfft(torch.complex(real_output, imag_output), n=time_steps, dim=1, norm="ortho")
        return self.norm(values + self.dropout(reconstructed))


class GraphFrequencyFusionModel(nn.Module):
    """Temporal stock encoder with cross-sectional time/frequency graph branches."""

    def __init__(
        self,
        input_dim: int,
        stock_count: int,
        hidden_dim: int = 64,
        top_k: int = 8,
        dropout: float = 0.0,
        graph_mode: str = "learned_deterministic",
        branch_mode: str = "dual_branch",
        fusion_mode: str = "fixed_mean",
        text_dim: int = 0,
        text_fusion: str = "none",
    ) -> None:
        super().__init__()
        if graph_mode not in {"no_graph", "identity", "provided", "learned_deterministic", "learned_gumbel"}:
            raise ValueError(f"unsupported graph_mode: {graph_mode}")
        if branch_mode not in {"temporal_only", "time_graph", "frequency_graph", "dual_branch"}:
            raise ValueError(f"unsupported branch_mode: {branch_mode}")
        if fusion_mode not in {"concat", "fixed_mean", "gated", "residual"}:
            raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
        if text_fusion not in {"none", "early", "mid"}:
            raise ValueError(f"unsupported text_fusion: {text_fusion}")
        if text_fusion != "none" and text_dim < 1:
            raise ValueError("text_dim must be positive when text fusion is enabled")
        self.stock_count = int(stock_count)
        self.graph_mode = graph_mode
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        self.text_fusion = text_fusion
        self.temporal_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.stock_embedding = nn.Parameter(torch.zeros(1, stock_count, hidden_dim))
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim) if text_dim > 0 else None
        self.early_projection = nn.Linear(hidden_dim * 2, hidden_dim) if text_fusion == "early" else None
        self.mid_projection = nn.Linear(hidden_dim * 2, hidden_dim) if text_fusion == "mid" else None
        sampling = "gumbel" if graph_mode == "learned_gumbel" else "deterministic"
        self.graph_learner = CrossSectionalGraphLearner(hidden_dim, top_k=top_k, sampling_mode=sampling) if graph_mode.startswith("learned_") else None
        self.time_graph = CrossSectionalTimeGraphBlock(hidden_dim, dropout=dropout)
        self.frequency_graph = CrossSectionalFrequencyGraphBlock(hidden_dim, dropout=dropout)
        self.concat_projection = nn.Linear(hidden_dim * 2, hidden_dim) if fusion_mode == "concat" else None
        self.gate_projection = nn.Linear(hidden_dim * 2, hidden_dim) if fusion_mode == "gated" else None
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.stock_embedding, mean=0.0, std=0.02)

    def _encode(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, stock_count, feature_count = values.shape
        sequences = values.permute(0, 2, 1, 3).reshape(batch_size * stock_count, time_steps, feature_count)
        encoded, _ = self.temporal_encoder(sequences)
        encoded = encoded.reshape(batch_size, stock_count, time_steps, -1).permute(0, 2, 1, 3)
        return self.temporal_norm(encoded + self.stock_embedding.unsqueeze(1))

    def _adjacency(self, nodes: torch.Tensor, node_available: torch.Tensor | None, provided: torch.Tensor | None) -> torch.Tensor:
        batch_size, stock_count = nodes.shape[:2]
        identity = torch.eye(stock_count, device=nodes.device, dtype=nodes.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        if self.graph_mode in {"no_graph", "identity"}:
            return identity
        if self.graph_mode == "provided":
            if provided is None or provided.shape != identity.shape:
                raise ValueError("provided adjacency must have shape [B,N,N]")
            return provided
        assert self.graph_learner is not None
        return self.graph_learner(nodes, node_available=node_available)

    def forward(
        self,
        values: torch.Tensor,
        node_available: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
        text_features: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> Any:
        if values.ndim != 4 or values.shape[2] != self.stock_count:
            raise ValueError("values must have shape [B,T,frozen_stock_count,F]")
        temporal = self._encode(values)
        text_hidden = None
        if self.text_fusion != "none":
            if text_features is None or text_features.shape[:2] != (values.shape[0], values.shape[2]):
                raise ValueError("text_features must have shape [B,N,text_dim]")
            assert self.text_projection is not None
            text_hidden = self.text_projection(text_features)
        if self.text_fusion == "early":
            repeated = text_hidden.unsqueeze(1).expand(-1, temporal.shape[1], -1, -1)
            temporal = self.early_projection(torch.cat([temporal, repeated], dim=-1))
        graph = self._adjacency(temporal[:, -1], node_available, adjacency)
        if self.branch_mode == "temporal_only" or self.graph_mode == "no_graph":
            fused = temporal[:, -1]
            time_values = temporal
            frequency_values = temporal
            gate = None
        else:
            time_values = self.time_graph(temporal, graph)
            frequency_values = self.frequency_graph(temporal, graph)
            if self.branch_mode == "time_graph":
                fused, gate = time_values[:, -1], None
            elif self.branch_mode == "frequency_graph":
                fused, gate = frequency_values[:, -1], None
            else:
                time_last, frequency_last = time_values[:, -1], frequency_values[:, -1]
                if self.fusion_mode == "concat":
                    fused, gate = self.concat_projection(torch.cat([time_last, frequency_last], dim=-1)), None
                elif self.fusion_mode == "fixed_mean":
                    fused, gate = 0.5 * time_last + 0.5 * frequency_last, None
                elif self.fusion_mode == "gated":
                    gate = torch.sigmoid(self.gate_projection(torch.cat([time_last, frequency_last], dim=-1)))
                    fused = gate * time_last + (1.0 - gate) * frequency_last
                else:
                    gate = None
                    base = temporal[:, -1]
                    fused = base + 0.5 * (time_last - base) + 0.5 * (frequency_last - base)
        if self.text_fusion == "mid":
            fused = self.mid_projection(torch.cat([fused, text_hidden], dim=-1))
        prediction = self.head(self.output_norm(fused)).squeeze(-1)
        if return_details:
            return {
                "prediction": prediction, "adjacency": graph, "temporal": temporal,
                "time_branch": time_values, "frequency_branch": frequency_values, "gate": gate,
            }
        return prediction
