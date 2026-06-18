# models/forecastgrapher.py
# -*- coding: utf-8 -*-
"""
ForecastGrapher: Adaptive graph learning + spatial-temporal modeling.
Inspired by: "ForecastGrapher: Redefining Multivariate Time Series Forecasting
with Graph Neural Networks" (2024).

Core idea:
  1. Project node features (B,L,N,F) -> (B,L,N,d_model)
  2. Learn adaptive graph from mean-pooled node embeddings -> (B,N,N)
  3. Apply GCN propagation at every time step (spatial mixing)
  4. Apply temporal self-attention per node (temporal mixing)
  5. Linear output head -> (B,N,H)

Interface (matches existing run_baselines.py panel convention):
  forward(x_seq, A_mech=None)
    x_seq : (B, L, N, F)
    returns: (y_seq: (B, N, H), A_learn: (B, N, N), gamma: tensor(0.))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph import LearnedGraphAttn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding along the time (L) axis."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, L, d_model)
        return self.drop(x + self.pe[:, : x.size(-2), :])


class _GCNLayer(nn.Module):
    """
    One GCN layer:  H_out = ReLU( A * H_in * W )
    Works with batched adjacency: A (B, N, N), H (B, N, d).
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # H: (B, N, d_in)   A: (B, N, N) row-normalized
        agg = torch.bmm(A, H)          # (B, N, d_in)
        out = self.fc(agg)             # (B, N, d_out)
        out = F.gelu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.norm(out)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ForecastGrapher(nn.Module):
    """
    ForecastGrapher baseline.

    Args:
        num_nodes  : N (number of assets)
        in_dim     : F (feature dimension per node per timestep)
        horizon    : H (prediction horizon)
        window_len : L (look-back window)
        d_model    : embedding dimension
        n_heads    : number of attention heads in temporal transformer
        n_layers   : number of transformer encoder layers
        gcn_steps  : number of stacked GCN layers
        graph_hidden: hidden dim for graph attention learner
        dropout    : dropout probability
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        horizon: int,
        window_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        gcn_steps: int = 2,
        graph_hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes  = int(num_nodes)
        self.in_dim     = int(in_dim)
        self.horizon    = int(horizon)
        self.window_len = int(window_len)
        self.d_model    = int(d_model)

        # 1. Input projection: (B, L, N, F) -> (B, L, N, d_model)
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, max_len=max(window_len + 10, 512), dropout=dropout)

        # 2. Adaptive graph learner (queries mean-pooled node embeddings)
        self.graph_learner = LearnedGraphAttn(
            in_dim=d_model,
            hidden_dim=graph_hidden,
            symmetric=True,
            dropout=dropout,
        )

        # 3. Stacked GCN for spatial propagation
        self.gcn_layers = nn.ModuleList([
            _GCNLayer(d_model, d_model, dropout=dropout)
            for _ in range(gcn_steps)
        ])

        # 4. Temporal transformer per node (shared weights)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # pre-LN (more stable)
        )
        self.temporal_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 5. Output head
        self.norm_out = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, horizon)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        """
        x_seq : (B, L, N, F)
        A_mech: ignored (graph is fully learned from data)
        Returns
        -------
        y_seq  : (B, N, H)
        A_learn: (B, N, N)
        gamma  : scalar tensor 0.
        """
        B, L, N, F = x_seq.shape

        # --- 1. Feature projection ---
        h = self.input_proj(x_seq)               # (B, L, N, d)

        # Positional encoding along L: reshape to (B*N, L, d)
        h_bn = h.permute(0, 2, 1, 3).contiguous().view(B * N, L, self.d_model)
        h_bn = self.pos_enc(h_bn)                # (B*N, L, d)
        h    = h_bn.view(B, N, L, self.d_model).permute(0, 2, 1, 3)  # (B, L, N, d)

        # --- 2. Learn adaptive graph ---
        H_mean   = h.mean(dim=1)                 # (B, N, d) — mean over time
        A_learn  = self.graph_learner(H_mean)    # (B, N, N)

        # --- 3. GCN spatial propagation (apply per time step via batch trick) ---
        # (B, L, N, d) -> (B*L, N, d), tile A to (B*L, N, N)
        h_bl  = h.contiguous().view(B * L, N, self.d_model)
        A_exp = A_learn.unsqueeze(1).expand(-1, L, -1, -1).contiguous().view(B * L, N, N)

        for gcn in self.gcn_layers:
            h_bl = gcn(h_bl, A_exp)              # (B*L, N, d)

        h = h_bl.view(B, L, N, self.d_model)    # (B, L, N, d)

        # --- 4. Temporal self-attention per node ---
        h_bn   = h.permute(0, 2, 1, 3).contiguous().view(B * N, L, self.d_model)
        h_bn   = self.temporal_enc(h_bn)         # (B*N, L, d)
        h_last = h_bn[:, -1, :]                  # take last timestep: (B*N, d)

        # --- 5. Predict ---
        h_last = self.norm_out(h_last)
        y_seq  = self.head(h_last).view(B, N, self.horizon)   # (B, N, H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y_seq, A_learn, gamma
