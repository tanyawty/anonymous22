"""Mechanism graph construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def build_adjacency_from_edges(
    edges_path: str,
    node_list: list[str],
    weight_col: str = "w",
    default_weight: float = 1.0,
    self_loop: bool = True,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Build a normalized mechanism adjacency matrix A_mech from an edge CSV.

    The edge CSV is expected to have columns: source, target, and optionally `weight_col`.

    Name resolution:
    - If 'wti' is not in node_list but 'px_wti' is, it will auto-map.

    Normalization:
    - Optional symmetrization (A <- 0.5*(A + A^T))
    - Symmetric normalization D^{-1/2} A D^{-1/2}

    Returns
    -------
    A_norm : torch.Tensor
        Shape (N, N), float32.
    """
    edges = pd.read_csv(edges_path)
    nodes_idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    A = np.zeros((N, N), dtype=float)

    def resolve_name(raw: str):
        if raw in nodes_idx:
            return raw
        cand = f"px_{raw}"
        if cand in nodes_idx:
            return cand
        return None

    for _, r in edges.iterrows():
        src_raw, tgt_raw = r["source"], r["target"]
        src = resolve_name(src_raw)
        tgt = resolve_name(tgt_raw)
        if src is None or tgt is None:
            continue
        i, j = nodes_idx[src], nodes_idx[tgt]
        w = r.get(weight_col, np.nan)
        if pd.isna(w):
            w = default_weight
        A[i, j] += float(w)

    if self_loop:
        np.fill_diagonal(A, A.diagonal() + 1.0)

    if symmetrize:
        A = 0.5 * (A + A.T)

    deg = np.sum(A, axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    A_norm = np.nan_to_num(A_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.tensor(A_norm, dtype=torch.float32)
