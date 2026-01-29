"""Temporal encoding layers.

This file keeps temporal encoders decoupled from specific models so that
you can swap GRU/TCN/Transformer easily.
"""

import torch
import torch.nn as nn


class TemporalGRU(nn.Module):
    """A thin wrapper around nn.GRU for (B, L, N, F) inputs.

    It reshapes per-node sequences into a batch of size (B*N), runs a GRU,
    and reshapes back.

    Args:
        in_dim: feature dimension per timestep.
        hidden_dim: GRU hidden size.
        num_layers: number of GRU layers.
        dropout: dropout between GRU layers (ignored if num_layers=1).
        bidirectional: whether to use bidirectional GRU (default False).

    Shapes:
        x_seq: (B, L, N, F)
        returns:
            H: (B, N, hidden_dim * (2 if bidirectional else 1))
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, L, N, F)
        B, L, N, F = x_seq.shape
        x = x_seq.permute(0, 2, 1, 3).contiguous()   # (B, N, L, F)
        x = x.view(B * N, L, F)                      # (B*N, L, F)
        out, _ = self.gru(x)                         # (B*N, L, H*)
        last = out[:, -1, :]                         # (B*N, H*)
        H = last.view(B, N, -1)                      # (B, N, H*)
        return H
