from __future__ import annotations

import torch
from torch import nn


class FixedEqualEnsemble(nn.Module):
    """Parameter-free average of the stable temporal and fixed-graph branches."""

    def forward(self, temporal_prediction: torch.Tensor, fixed_graph_prediction: torch.Tensor) -> torch.Tensor:
        if temporal_prediction.shape != fixed_graph_prediction.shape:
            raise ValueError("ensemble component predictions must have identical shapes")
        return 0.5 * temporal_prediction + 0.5 * fixed_graph_prediction

