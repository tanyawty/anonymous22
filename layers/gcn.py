# layers/gcn.py
# -*- coding: utf-8 -*-
"""
GCN / message passing layers.

Provides:
- GraphConv: simple A @ XW (optionally with bias) + activation
- GCNLayer: a more stable layer with optional residual, dropout, and normalization

Shapes:
- X: (B, N, F_in)
- A: (B, N, N) or (N, N) (broadcasted to batch)

Notes:
- We do NOT normalize A inside by default. Keep normalization consistent in graph.py (mech)
  and LearnedGraphAttn (row-softmax) to avoid double-normalization.
"""

from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_batched_adjacency(A: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Accept A as (N,N) or (B,N,N). Return (B,N,N).
    """
    if A.dim() == 2:
        return A.unsqueeze(0).expand(batch_size, -1, -1)
    if A.dim() == 3:
        return A
    raise ValueError(f"A must be (N,N) or (B,N,N), got {tuple(A.shape)}")


class GraphConv(nn.Module):
    """
    Minimal graph convolution:
      X' = act( A @ (X W + b) )

    This mirrors your current implementation but lives as a reusable layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Optional[str] = "relu",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = float(dropout)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, F_in)
        A: (B, N, N) or (N, N)
        """
        if X.dim() != 3:
            raise ValueError(f"X must be (B,N,F), got {tuple(X.shape)}")
        B, N, _ = X.shape
        A_b = _ensure_batched_adjacency(A, B)

        Xw = self.lin(X)                  # (B, N, out_dim)
        Xw = F.dropout(Xw, p=self.dropout, training=self.training)
        AX = torch.bmm(A_b, Xw)           # (B, N, out_dim)

        if self.activation is None or self.activation == "none":
            return AX
        if self.activation == "relu":
            return F.relu(AX)
        if self.activation == "gelu":
            return F.gelu(AX)
        if self.activation == "tanh":
            return torch.tanh(AX)
        raise ValueError(f"Unknown activation: {self.activation}")


class GCNLayer(nn.Module):
    """
    A more stable GCN block:
      H = A @ (X W)
      H = dropout(H)
      H = H + residual (optional, with projection if needed)
      H = norm(H) (optional)
      H = act(H)

    This is generally more robust across different A distributions (mech DAD vs learned softmax).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: Literal["relu", "gelu", "tanh", "none"] = "relu",
        dropout: float = 0.0,
        residual: bool = True,
        norm: Literal["none", "layernorm"] = "layernorm",
        bias: bool = True,
    ):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

        self.activation = activation
        self.dropout = float(dropout)
        self.residual = bool(residual)

        if self.residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_proj = None

        if norm == "layernorm":
            self.norm = nn.LayerNorm(out_dim)
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError(f"Unknown norm: {norm}")

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, F_in)
        A: (B, N, N) or (N, N)
        """
        if X.dim() != 3:
            raise ValueError(f"X must be (B,N,F), got {tuple(X.shape)}")
        B, _, _ = X.shape
        A_b = _ensure_batched_adjacency(A, B)

        H = self.lin(X)                    # (B, N, out_dim)
        H = torch.bmm(A_b, H)              # (B, N, out_dim)
        H = F.dropout(H, p=self.dropout, training=self.training)

        if self.residual:
            res = self.res_proj(X) if self.res_proj is not None else X
            H = H + res

        if self.norm is not None:
            # LayerNorm expects last dim = features
            H = self.norm(H)

        if self.activation == "none" or self.activation is None:
            return H
        if self.activation == "relu":
            return F.relu(H)
        if self.activation == "gelu":
            return F.gelu(H)
        if self.activation == "tanh":
            return torch.tanh(H)
        raise ValueError(f"Unknown activation: {self.activation}")

