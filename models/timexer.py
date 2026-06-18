# models/timexer.py
# -*- coding: utf-8 -*-
"""
TimeXer: Transformer empowered with exogenous variable handling.
Inspired by: "TimeXer: Empowering Transformers for Time Series Forecasting
with Exogenous Variables" (NeurIPS 2024).

Adaptation for this repo's (B, L, N, F) format:
  - Endogenous series  : x_seq[:, :, :, 0]     -> (B, L, N)  raw log-returns per node
  - Exogenous features : x_seq[:, :, :, 1:]    -> (B, L, N, F-1)  derived + macro features
                         aggregated over nodes  -> (B, L, F-1) shared exogenous signal

Architecture per node:
  1. Patch the endogenous return series -> (B*N, num_patches, d_model)
  2. Encode exogenous features as temporal tokens -> (B*N, L, d_model)
  3. Alternating:
       (a) self-attention within endogenous patches
       (b) cross-attention: endo patches attend to exo tokens
  4. Flatten patches -> linear head -> (B, N, H)

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
# Sub-modules
# ---------------------------------------------------------------------------

class _PatchEmbedding(nn.Module):
    """
    Divide a 1-D time series into overlapping patches and project to d_model.
    Input : (batch, L)
    Output: (batch, num_patches, d_model)
    """
    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.proj      = nn.Linear(patch_len, d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L)
        x_unfold = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # -> (batch, num_patches, patch_len)
        return self.drop(self.proj(x_unfold))   # (batch, num_patches, d_model)


class _CrossAttentionBlock(nn.Module):
    """
    Endogenous tokens (Q) attend to exogenous tokens (K, V),
    followed by a feed-forward network.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q     = nn.LayerNorm(d_model)
        self.norm_kv    = nn.LayerNorm(d_model)
        self.norm_ff    = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q : (batch, Pq, d)   endogenous patch tokens
        kv: (batch, Le, d)   exogenous temporal tokens
        """
        attn_out, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(attn_out)
        q = q + self.drop(self.ff(self.norm_ff(q)))
        return q


class _SelfAttentionBlock(nn.Module):
    """Standard pre-LN transformer encoder block."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TimeXer(nn.Module):
    """
    TimeXer baseline.

    Args:
        num_nodes  : N
        in_dim     : F  (first channel = raw return; rest = exogenous)
        horizon    : H
        window_len : L
        patch_len  : length of each patch (endogenous patching)
        patch_stride: stride for patching
        d_model    : embedding dimension
        n_heads    : attention heads
        n_layers   : number of (self-attn + cross-attn) layer pairs
        dropout    : dropout
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        horizon: int,
        window_len: int,
        patch_len: int    = 4,
        patch_stride: int = 2,
        d_model: int      = 64,
        n_heads: int      = 4,
        n_layers: int     = 2,
        dropout: float    = 0.1,
    ):
        super().__init__()
        self.num_nodes  = int(num_nodes)
        self.in_dim     = int(in_dim)
        self.horizon    = int(horizon)
        self.window_len = int(window_len)
        self.d_model    = int(d_model)
        self.exo_dim    = max(1, in_dim - 1)   # features other than raw return

        # Number of patches after unfolding
        self.num_patches = max(1, (window_len - patch_len) // patch_stride + 1)

        # ---- Endogenous branch ----
        self.endo_patch = _PatchEmbedding(
            patch_len=patch_len, stride=patch_stride, d_model=d_model, dropout=dropout
        )
        # Learnable positional embeddings for patches
        self.endo_pos = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # ---- Exogenous branch ----
        # Mean-over-nodes exo features -> (B, L, exo_dim) -> project to d_model per timestep
        self.exo_proj = nn.Linear(self.exo_dim, d_model)
        # Positional encoding for the L exo tokens
        self.exo_pos  = nn.Parameter(torch.randn(1, window_len, d_model) * 0.02)

        # ---- Alternating attention layers ----
        self.self_blocks  = nn.ModuleList([
            _SelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.cross_blocks = nn.ModuleList([
            _CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        # ---- Output head ----
        self.norm_out  = nn.LayerNorm(d_model)
        self.head      = nn.Linear(self.num_patches * d_model, horizon)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.endo_pos, std=0.02)
        nn.init.trunc_normal_(self.exo_pos,  std=0.02)

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

        # ---- Split endo / exo ----
        endo = x_seq[:, :, :, 0]                              # (B, L, N)
        exo  = x_seq[:, :, :, 1:] if F > 1 else \
               torch.zeros(B, L, N, 1, device=x_seq.device)  # (B, L, N, exo_dim)

        # ---- Endogenous patch tokens: (B*N, num_patches, d_model) ----
        # Flatten nodes into batch dim
        endo_flat  = endo.permute(0, 2, 1).contiguous().view(B * N, L)   # (B*N, L)
        endo_tok   = self.endo_patch(endo_flat)    # (B*N, num_patches, d_model)
        endo_tok   = endo_tok + self.endo_pos      # positional

        # ---- Exogenous tokens: aggregate over nodes -> (B, L, exo_dim) ----
        exo_mean   = exo.mean(dim=2)               # (B, L, exo_dim)
        exo_tok_b  = self.exo_proj(exo_mean)       # (B, L, d_model)
        exo_tok_b  = exo_tok_b + self.exo_pos      # positional: (B, L, d_model)

        # Expand exo tokens to match B*N batch (each node shares the same exo signal)
        exo_tok    = exo_tok_b.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, L, d_model)
        exo_tok    = exo_tok.contiguous().view(B * N, L, self.d_model)

        # ---- Alternating self-attn + cross-attn ----
        for self_blk, cross_blk in zip(self.self_blocks, self.cross_blocks):
            endo_tok = self_blk(endo_tok)             # (B*N, P, d)
            endo_tok = cross_blk(endo_tok, exo_tok)   # (B*N, P, d)

        # ---- Output head ----
        endo_tok = self.norm_out(endo_tok)             # (B*N, P, d)
        flat     = endo_tok.contiguous().view(B * N, self.num_patches * self.d_model)
        y_seq    = self.head(flat).view(B, N, self.horizon)   # (B, N, H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y_seq, None, gamma
