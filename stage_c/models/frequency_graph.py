from __future__ import annotations

import torch
from torch import nn


class FrequencyGraphBlock(nn.Module):
    """Complex-valued graph propagation in the temporal Fourier domain."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.real_linear = nn.Linear(hidden_dim, hidden_dim)
        self.imag_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or adjacency.ndim != 3:
            raise ValueError("expected values [B,T,D] and adjacency [B,T,T]")
        spectrum = torch.fft.fft(values, dim=1, norm="ortho")
        real = torch.matmul(adjacency, spectrum.real)
        imag = torch.matmul(adjacency, spectrum.imag)
        real_output = self.real_linear(real) - self.imag_linear(imag)
        imag_output = self.imag_linear(real) + self.real_linear(imag)
        propagated = torch.complex(real_output, imag_output)
        reconstructed = torch.fft.ifft(propagated, dim=1, norm="ortho").real
        return self.norm(values + self.dropout(reconstructed))

