# layers/temporal.py
# -*- coding: utf-8 -*-
"""
Temporal encoders (node-wise and sequence-wise).

Provides:
- NodeWiseGRUEncoder: encode each node's time series independently (shared GRU across nodes)
- NodeWiseLSTMEncoder: optional LSTM variant
- TemporalMLP: a lightweight temporal mixer (baseline / ablation friendly)

Conventions:
Input to node-wise encoders:
  H_seq: (B, L, N, F)  where
    B=batch, L=window length, N=#nodes, F=feature dim after spatial mixing

Output:
  h_last: (B, N, H)   last hidden state per node
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_4d(x: torch.Tensor, name: str) -> Tuple[int, int, int, int]:
    if x.dim() != 4:
        raise ValueError(f"{name} must be (B,L,N,F), got {tuple(x.shape)}")
    return x.shape  # type: ignore[return-value]


class NodeWiseGRUEncoder(nn.Module):
    """
    Shared GRU over time for each node, implemented by flattening (B,N) -> (B*N).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        # PyTorch GRU dropout works only when num_layers > 1
        gru_dropout = float(dropout) if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=self.bidirectional,
        )

        out_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.out_dim = out_dim

    def forward(
        self,
        H_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        H_seq: (B, L, N, F)
        h0:    optional initial hidden (num_layers * num_directions, B*N, hidden_dim)
        return:
          h_last: (B, N, out_dim)
        """
        B, L, N, F_in = _check_4d(H_seq, "H_seq")
        if F_in != self.in_dim:
            raise ValueError(f"Expected F={self.in_dim}, got F={F_in}")

        # (B, L, N, F) -> (B, N, L, F) -> (B*N, L, F)
        x = H_seq.permute(0, 2, 1, 3).contiguous().view(B * N, L, F_in)

        out, hN = self.gru(x, h0)  # hN: (num_layers*num_dir, B*N, hidden_dim)

        # Take last layer hidden
        h_last_layer = hN[-(2 if self.bidirectional else 1) :]  # (num_dir, B*N, hidden_dim) if bi else (1, B*N, hidden_dim)
        if self.bidirectional:
            # concat forward/backward hidden
            h_last = torch.cat([h_last_layer[0], h_last_layer[1]], dim=-1)  # (B*N, 2*hidden_dim)
        else:
            h_last = h_last_layer[0]  # (B*N, hidden_dim)

        h_last = h_last.view(B, N, self.out_dim)
        return h_last


class NodeWiseLSTMEncoder(nn.Module):
    """
    Shared LSTM over time for each node (optional alternative).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional,
        )

        out_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.out_dim = out_dim

    def forward(
        self,
        H_seq: torch.Tensor,
        state0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        H_seq: (B, L, N, F)
        state0: (h0, c0) where each is (num_layers*num_dir, B*N, hidden_dim)
        return:
          h_last: (B, N, out_dim)
        """
        B, L, N, F_in = _check_4d(H_seq, "H_seq")
        if F_in != self.in_dim:
            raise ValueError(f"Expected F={self.in_dim}, got F={F_in}")

        x = H_seq.permute(0, 2, 1, 3).contiguous().view(B * N, L, F_in)
        out, (hN, cN) = self.lstm(x, state0)

        h_last_layer = hN[-(2 if self.bidirectional else 1) :]
        if self.bidirectional:
            h_last = torch.cat([h_last_layer[0], h_last_layer[1]], dim=-1)
        else:
            h_last = h_last_layer[0]

        h_last = h_last.view(B, N, self.out_dim)
        return h_last


class TemporalMLP(nn.Module):
    """
    Lightweight temporal mixer:
      - flattens time dimension and applies an MLP per node.

    Useful as a fast ablation baseline when you don't want recurrent models.

    Input:  (B, L, N, F)
    Output: (B, N, out_dim)
    """

    def __init__(
        self,
        window_len: int,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "tanh"] = "gelu",
    ):
        super().__init__()
        self.window_len = int(window_len)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.activation = activation

        self.fc1 = nn.Linear(self.window_len * self.in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, H_seq: torch.Tensor) -> torch.Tensor:
        B, L, N, F_in = _check_4d(H_seq, "H_seq")
        if L != self.window_len:
            raise ValueError(f"Expected L={self.window_len}, got L={L}")
        if F_in != self.in_dim:
            raise ValueError(f"Expected F={self.in_dim}, got F={F_in}")

        # (B, L, N, F) -> (B, N, L, F) -> (B, N, L*F)
        x = H_seq.permute(0, 2, 1, 3).contiguous().view(B, N, L * F_in)

        x = self.fc1(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x

