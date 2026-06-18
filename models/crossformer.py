# models/crossformer.py
# -*- coding: utf-8 -*-
"""
Crossformer: Transformer utilizing cross-dimension dependency for multivariate
time series forecasting.
Inspired by: "Crossformer: Transformer Utilizing Cross-Dimension Dependency for
Multivariate Time Series Forecasting" (ICLR 2023).

Core idea (adapted to this repo's (B, L, N, F) format):
  1. Segment-based embedding: divide L time steps into fixed segments per node
     -> (B, N, num_seg, d_model)
  2. Stack of two-stage attention layers:
       Stage 1 (temporal)   : each node attends over its own segments
       Stage 2 (cross-dim)  : a small set of router tokens gather info from all
                              nodes and broadcast back, modeling N-dim dependencies
  3. Flatten segment embeddings -> linear output head -> (B, N, H)

Interface:
  forward(x_seq, A_mech=None)
    x_seq : (B, L, N, F)
    returns: (y_seq: (B, N, H), None, gamma: tensor(0.))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Segment embedding
# ---------------------------------------------------------------------------

class _SegmentEmbedding(nn.Module):
    """
    Divide the L-length feature sequence of each node into non-overlapping
    segments of length `seg_len` and project each segment to d_model.

    Input : (batch, L, in_dim)       where batch = B*N
    Output: (batch, num_seg, d_model)
    """
    def __init__(self, seg_len: int, in_dim: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.seg_len = int(seg_len)
        self.proj    = nn.Linear(seg_len * in_dim, d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, F)
        BN, L, F = x.shape
        # Pad to make L divisible by seg_len
        pad = (self.seg_len - L % self.seg_len) % self.seg_len
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))           # (BN, L+pad, F)
        L_pad  = x.size(1)
        n_seg  = L_pad // self.seg_len
        x      = x.view(BN, n_seg, self.seg_len * F)
        return self.drop(self.proj(x))              # (BN, n_seg, d_model)


# ---------------------------------------------------------------------------
# Two-stage attention layer
# ---------------------------------------------------------------------------

class _TwoStageAttnLayer(nn.Module):
    """
    One Crossformer encoder layer.

    Stage 1 (temporal): shared self-attention over `num_seg` segments within
                        each node independently.
    Stage 2 (cross-dim): learnable router tokens first aggregate from all N
                         nodes (many-to-few attention), then broadcast back to
                         nodes (few-to-many attention).

    Input / Output: (B, N, num_seg, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_seg: int,
        n_routers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.num_seg   = num_seg
        self.n_routers = n_routers

        # ---- Stage 1: temporal self-attention ----
        self.t_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.t_norm1 = nn.LayerNorm(d_model)
        self.t_ff    = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.t_norm2 = nn.LayerNorm(d_model)
        self.t_drop  = nn.Dropout(dropout)

        # ---- Stage 2: router-based cross-dimension attention ----
        # Router tokens: shape (1, num_seg, n_routers, d_model) — one set per segment
        self.router = nn.Parameter(
            torch.randn(1, num_seg, n_routers, d_model) * 0.02
        )

        # Many-to-few: routers attend to node representations
        self.d_attn_q = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.d_norm_r = nn.LayerNorm(d_model)

        # Few-to-many: nodes attend back to (updated) routers
        self.d_attn_v = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.d_norm1  = nn.LayerNorm(d_model)
        self.d_ff     = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.d_norm2  = nn.LayerNorm(d_model)
        self.d_drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, num_seg, d_model)"""
        B, N, S, d = x.shape

        # ===== Stage 1: temporal attention =====
        # Flatten (B, N) -> process (B*N, S, d)
        x_bn = x.contiguous().view(B * N, S, d)

        attn_out, _ = self.t_attn(self.t_norm1(x_bn), self.t_norm1(x_bn), self.t_norm1(x_bn))
        x_bn = x_bn + self.t_drop(attn_out)
        x_bn = x_bn + self.t_drop(self.t_ff(self.t_norm2(x_bn)))

        x = x_bn.view(B, N, S, d)

        # ===== Stage 2: cross-dimension attention via routers =====
        # Reorganise to (B, S, N, d) to process per segment
        x_bs = x.permute(0, 2, 1, 3).contiguous()            # (B, S, N, d)
        x_seg = x_bs.view(B * S, N, d)                       # (B*S, N, d)

        # Router tokens: (1, S, n_routers, d) -> (B, S, n_routers, d) -> (B*S, n_routers, d)
        r = self.router.expand(B, -1, -1, -1).contiguous().view(B * S, self.n_routers, d)

        # Many-to-few: routers gather from node representations
        r_out, _ = self.d_attn_q(self.d_norm_r(r), x_seg, x_seg)   # (B*S, n_routers, d)
        r = r + self.d_drop(r_out)

        # Few-to-many: nodes gather from updated routers
        n_out, _ = self.d_attn_v(x_seg, r, r)                       # (B*S, N, d)
        x_seg = x_seg + self.d_drop(n_out)
        x_seg = x_seg + self.d_drop(self.d_ff(self.d_norm2(x_seg)))

        x_bs = x_seg.view(B, S, N, d)
        x    = x_bs.permute(0, 2, 1, 3).contiguous()                # (B, N, S, d)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Crossformer(nn.Module):
    """
    Crossformer baseline.

    Args:
        num_nodes  : N
        in_dim     : F
        horizon    : H
        window_len : L
        seg_len    : segment length for the segment embedding
        d_model    : embedding dimension
        n_heads    : number of attention heads
        n_layers   : number of stacked two-stage attention layers
        n_routers  : number of router tokens for cross-dim stage
        dropout    : dropout probability
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        horizon: int,
        window_len: int,
        seg_len: int   = 4,
        d_model: int   = 64,
        n_heads: int   = 4,
        n_layers: int  = 3,
        n_routers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes  = int(num_nodes)
        self.in_dim     = int(in_dim)
        self.horizon    = int(horizon)
        self.window_len = int(window_len)
        self.d_model    = int(d_model)

        # Pre-compute num_seg (after padding)
        pad = (seg_len - window_len % seg_len) % seg_len
        self.num_seg = (window_len + pad) // seg_len

        # 1. Segment embedding
        self.seg_emb = _SegmentEmbedding(
            seg_len=seg_len, in_dim=in_dim, d_model=d_model, dropout=dropout
        )
        # Learnable positional embeddings for segments
        self.seg_pos = nn.Parameter(torch.randn(1, self.num_seg, d_model) * 0.02)

        # 2. Stacked two-stage attention
        self.layers = nn.ModuleList([
            _TwoStageAttnLayer(
                d_model=d_model, n_heads=n_heads, num_seg=self.num_seg,
                n_routers=n_routers, dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # 3. Output head: flatten segments -> project to horizon
        self.norm_out = nn.LayerNorm(d_model)
        self.head     = nn.Linear(self.num_seg * d_model, horizon)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.seg_pos, std=0.02)
        nn.init.trunc_normal_(self.router if hasattr(self, "router") else self.seg_pos, std=0.02)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        """
        x_seq : (B, L, N, F)
        Returns
        -------
        y_seq  : (B, N, H)
        A_learn: None
        gamma  : scalar tensor 0.
        """
        B, L, N, F = x_seq.shape

        # 1. Segment embedding per node
        # (B, L, N, F) -> permute to (B, N, L, F) -> flatten to (B*N, L, F)
        x_flat = x_seq.permute(0, 2, 1, 3).contiguous().view(B * N, L, F)
        seg    = self.seg_emb(x_flat)           # (B*N, num_seg, d_model)
        seg    = seg + self.seg_pos             # add positional embeddings

        # Reshape back to (B, N, num_seg, d_model)
        seg = seg.view(B, N, self.num_seg, self.d_model)

        # 2. Stacked two-stage attention
        for layer in self.layers:
            seg = layer(seg)                    # (B, N, num_seg, d_model)

        # 3. Output head
        seg  = self.norm_out(seg)               # (B, N, num_seg, d_model)
        flat = seg.contiguous().view(B, N, self.num_seg * self.d_model)
        y_seq = self.head(flat)                 # (B, N, H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y_seq, None, gamma
