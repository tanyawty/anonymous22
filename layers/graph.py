"""Graph structure learner layers."""
# layers/graph.py
# -*- coding: utf-8 -*-
"""
Graph-related layers & utilities:
- LearnedGraphAttn: learn a dense (or masked/sparsified) adjacency from node embeddings
- build_adjacency_from_edges: build a prior/mechanism adjacency from an edge list (CSV/DF)
- helpers: normalize adjacency, apply masks, top-k sparsify
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helpers
# -----------------------------
def _to_device_dtype(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x.device != device:
        x = x.to(device)
    if x.dtype != dtype:
        x = x.to(dtype=dtype)
    return x


def add_self_loops(A: np.ndarray, weight: float = 1.0) -> np.ndarray:
    A2 = A.copy()
    np.fill_diagonal(A2, np.diag(A2) + weight)
    return A2


def symmetrize(A: np.ndarray, method: str = "max") -> np.ndarray:
    """
    method:
      - "max": A <- max(A, A^T)
      - "mean": A <- (A + A^T)/2
      - "sum": A <- A + A^T
    """
    if method == "max":
        return np.maximum(A, A.T)
    if method == "mean":
        return 0.5 * (A + A.T)
    if method == "sum":
        return A + A.T
    raise ValueError(f"Unknown symmetrize method: {method}")


def normalize_adjacency(A: np.ndarray, kind: str = "sym") -> np.ndarray:
    """
    Normalize adjacency.
    kind:
      - "sym": D^{-1/2} A D^{-1/2}
      - "row": D^{-1} A
      - "none": no normalization
    """
    if kind == "none":
        return A

    eps = 1e-12
    deg = A.sum(axis=1)  # (N,)
    if kind == "row":
        inv = 1.0 / (deg + eps)
        return (inv[:, None] * A)

    if kind == "sym":
        inv_sqrt = 1.0 / np.sqrt(deg + eps)
        return (inv_sqrt[:, None] * A) * inv_sqrt[None, :]

    raise ValueError(f"Unknown normalize kind: {kind}")


def apply_edge_mask(A: torch.Tensor, mask: torch.Tensor, fill: float = -1e9) -> torch.Tensor:
    """
    Apply a boolean/0-1 mask to A.
    - If A is logits, use fill=-1e9 so softmax zeroes masked edges.
    - If A is already probabilities, use fill=0.0 and multiply.
    Shapes:
      A:   (B, N, N) or (N, N)
      mask:(N, N) or broadcastable to A
    """
    if mask.dtype != torch.bool:
        mask = mask > 0
    # broadcast mask to A
    while mask.dim() < A.dim():
        mask = mask.unsqueeze(0)
    return torch.where(mask, A, torch.full_like(A, fill))


def topk_sparsify(
    A: torch.Tensor,
    k: int,
    dim: int = -1,
    renorm: bool = True,
) -> torch.Tensor:
    """
    Keep only top-k entries along `dim` for each row (default last dim).
    A is assumed to be non-negative (probabilities) or logits (any real):
      - If probabilities: zeros out others; optional renorm to sum=1 along dim.
      - If logits: keeps logits for top-k, sets others to -inf; (then softmax later).
    """
    if k is None or k <= 0:
        return A

    # Work with last dim by default
    values, idx = torch.topk(A, k=k, dim=dim)
    sparse = torch.full_like(A, float("-inf") if A.dtype.is_floating_point else 0)

    sparse.scatter_(dim, idx, values)

    if renorm:
        # If logits, renorm happens after softmax in caller.
        # If already probs (non-negative & sum-ish), we can renorm here:
        if torch.isfinite(sparse).all():
            # looks like probs; avoid dividing by 0
            s = sparse.sum(dim=dim, keepdim=True).clamp_min(1e-12)
            sparse = sparse / s

    return sparse


# -----------------------------
# Learned graph layer
# -----------------------------
class LearnedGraphAttn(nn.Module):
    """
    Attention-style graph learner:
      input:  H_node (B, N, F_in)
      output: A (B, N, N) row-stochastic adjacency by default

    Options:
      - mask: restrict candidate edges (e.g., mech edges only)
      - topk: sparsify adjacency to top-k neighbors per node
      - symmetric: enforce A == A^T (after softmax)
      - temperature: softmax temperature on attention logits
      - return_logits: return logits before softmax (useful for debugging)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        symmetric: bool = False,
        temperature: float = 1.0,
        topk: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.symmetric = symmetric
        self.temperature = float(temperature)
        self.topk = topk
        self.dropout = float(dropout)

    def forward(
        self,
        H_node: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        H_node: (B, N, F)
        mask:   (N, N) or (B, N, N) boolean/0-1 tensor
        """
        if H_node.dim() != 3:
            raise ValueError(f"H_node should be (B,N,F), got {tuple(H_node.shape)}")

        B, N, _ = H_node.shape
        Z = self.proj(H_node)  # (B, N, d)
        Z = F.dropout(Z, p=self.dropout, training=self.training)

        # scaled dot-product
        d = Z.size(-1)
        logits = torch.bmm(Z, Z.transpose(1, 2)) / math.sqrt(max(d, 1))  # (B, N, N)
        if self.temperature != 1.0:
            logits = logits / max(self.temperature, 1e-8)

        if mask is not None:
            # If we use softmax, masked edges should get -inf so prob->0
            logits = apply_edge_mask(logits, mask, fill=-1e9)

        # Optional top-k sparsification on logits BEFORE softmax
        if self.topk is not None and self.topk > 0 and self.topk < N:
            logits = topk_sparsify(logits, k=self.topk, dim=-1, renorm=False)

        A = torch.softmax(logits, dim=-1)  # row-stochastic

        if self.symmetric:
            A = 0.5 * (A + A.transpose(1, 2))
            # re-normalize rows
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if return_logits:
            return A, logits
        return A


# -----------------------------
# Prior / mechanism adjacency builder
# -----------------------------
@dataclass
class EdgeBuildConfig:
    src_col: str = "src"
    dst_col: str = "dst"
    w_col: str = "w"

    # name mapping helpers
    allow_px_prefix_fix: bool = True  # map "wti" -> "px_wti" if node_list uses "px_*"

    # graph ops
    add_self_loop: bool = True
    self_loop_weight: float = 1.0
    make_symmetric: bool = True
    sym_method: str = "max"  # max/mean/sum

    # normalization
    normalize: str = "sym"  # sym/row/none

    # weights
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


def _resolve_node_name(name: str, node_list: Sequence[str], allow_px_prefix_fix: bool = True) -> Optional[str]:
    if name in node_list:
        return name
    if allow_px_prefix_fix:
        if ("px_" + name) in node_list:
            return "px_" + name
        # also allow reverse
        if name.startswith("px_"):
            raw = name[3:]
            if raw in node_list:
                return raw
    return None


def build_adjacency_from_edges(
    edges: Union[str, pd.DataFrame],
    node_list: Sequence[str],
    cfg: Optional[EdgeBuildConfig] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build (N,N) adjacency tensor from an edge list.

    edges:
      - path to CSV
      - or DataFrame with columns [src,dst,w] (configurable)
    node_list:
      list of node names in the order used by your dataset/model

    returns:
      A: torch.Tensor (N,N) normalized as cfg.normalize
    """
    if cfg is None:
        cfg = EdgeBuildConfig()

    if isinstance(edges, str):
        if not os.path.exists(edges):
            raise FileNotFoundError(f"Edge file not found: {edges}")
        df = pd.read_csv(edges)
    elif isinstance(edges, pd.DataFrame):
        df = edges.copy()
    else:
        raise TypeError("edges must be a CSV path or a pandas DataFrame")

    required = {cfg.src_col, cfg.dst_col, cfg.w_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Edges missing columns {missing}. Found columns: {list(df.columns)}")

    N = len(node_list)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    A = np.zeros((N, N), dtype=np.float64)

    for _, row in df.iterrows():
        src_raw = str(row[cfg.src_col]).strip()
        dst_raw = str(row[cfg.dst_col]).strip()
        w = float(row[cfg.w_col])

        src = _resolve_node_name(src_raw, node_list, allow_px_prefix_fix=cfg.allow_px_prefix_fix)
        dst = _resolve_node_name(dst_raw, node_list, allow_px_prefix_fix=cfg.allow_px_prefix_fix)
        if src is None or dst is None:
            # silently skip unknown nodes
            continue

        if cfg.clip_min is not None:
            w = max(w, cfg.clip_min)
        if cfg.clip_max is not None:
            w = min(w, cfg.clip_max)

        A[node_to_idx[src], node_to_idx[dst]] = w

    if cfg.add_self_loop:
        A = add_self_loops(A, weight=cfg.self_loop_weight)
    if cfg.make_symmetric:
        A = symmetrize(A, method=cfg.sym_method)

    A = normalize_adjacency(A, kind=cfg.normalize)

    A_t = torch.tensor(A, dtype=dtype)
    if device is not None:
        A_t = A_t.to(device)
    return A_t


def mech_edge_mask_from_adjacency(A_mech: torch.Tensor, include_self: bool = True) -> torch.Tensor:
    """
    Create a boolean mask from a prior adjacency (N,N):
      mask[i,j] = True if A_mech[i,j] != 0
    Useful to constrain LearnedGraphAttn to only mech candidate edges.
    """
    if A_mech.dim() != 2:
        raise ValueError(f"A_mech must be (N,N), got {tuple(A_mech.shape)}")
    mask = (A_mech != 0)
    if include_self:
        mask = mask | torch.eye(A_mech.size(0), device=A_mech.device, dtype=torch.bool)
    return mask
