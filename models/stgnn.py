# models/stgnn_learn.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn

from layers.graph import LearnedGraphAttn
from layers.gcn import GraphConv
from layers.temporal import NodeWiseGRUEncoder


class STGNN_LearnOnly(nn.Module):
    """
    Learn-only STGNN baseline (purely learned graph).

    Pipeline:
      1) Graph learner builds A_learn from a window summary (mean over time)
      2) Spatial message passing (GCN) per time step using A_learn
      3) Temporal encoder (node-wise GRU) aggregates across time for each node
      4) Linear head outputs horizon-step sequence for each node

    forward:
      x_seq: (B, L, N, F)
      A_mech: ignored (kept for unified interface)
    returns:
      y_seq:  (B, N, H)
      A_learn:(B, N, N)
      gamma:  tensor(0.0)   (for unified interface with mech/hybrid models)
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        horizon: int,
        # hidden sizes
        gcn_hidden: int = 32,
        gru_hidden: int = 64,
        graph_hidden: int = 64,
        # graph learner controls
        graph_symmetric: bool = True,
        graph_temperature: float = 1.0,
        graph_topk: int | None = None,
        # dropout (runner-friendly)
        dropout: float = 0.0,
        gcn_dropout: float | None = None,
        rnn_layers: int = 1,
        rnn_dropout: float | None = None,
        # optional: graph learner dropout (keep default 0.0 for stability)
        graph_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.horizon = int(horizon)

        # ---- dropout mapping (compat with run_baselines: dropout=...) ----
        # If user didn't explicitly provide layer-specific dropouts,
        # fall back to global `dropout`.
        if gcn_dropout is None:
            gcn_dropout = float(dropout)
        if rnn_dropout is None:
            rnn_dropout = float(dropout)

        # ---- Learn adjacency ----
        self.graph_learner = LearnedGraphAttn(
            in_dim=self.in_dim,
            hidden_dim=int(graph_hidden),
            symmetric=bool(graph_symmetric),
            temperature=float(graph_temperature),
            topk=graph_topk,
            dropout=float(graph_dropout),
        )

        # ---- Spatial message passing ----
        self.gcn = GraphConv(
            in_dim=self.in_dim,
            out_dim=int(gcn_hidden),
            activation="relu",
            dropout=float(gcn_dropout),
        )

        # ---- Temporal encoder (per node) ----
        self.temporal = NodeWiseGRUEncoder(
            in_dim=int(gcn_hidden),
            hidden_dim=int(gru_hidden),
            num_layers=int(rnn_layers),
            dropout=float(rnn_dropout),
            bidirectional=False,
        )

        # ---- Horizon head ----
        self.head = nn.Linear(self.temporal.out_dim, self.horizon)

        # ---- gamma (fixed 0 for learn-only) ----
        self.register_buffer("_gamma0", torch.tensor(0.0))

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        """
        Args:
            x_seq: (B, L, N, F)
            A_mech: ignored (for unified interface)
        Returns:
            y_seq: (B, N, H)
            A_learn: (B, N, N)
            gamma: tensor(0.0)
        """
        if x_seq.dim() != 4:
            raise ValueError(f"x_seq must be (B,L,N,F), got {tuple(x_seq.shape)}")

        B, L, N, F = x_seq.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")
        if F != self.in_dim:
            raise ValueError(f"F mismatch: got {F}, expected {self.in_dim}")

        # 1) Learn adjacency from window summary
        H_node = x_seq.mean(dim=1)           # (B, N, F)
        A_learn = self.graph_learner(H_node) # (B, N, N)

        # 2) Spatial propagation per time step
        H_list = []
        for t in range(L):
            Xt = x_seq[:, t, :, :]          # (B, N, F)
            Ht = self.gcn(Xt, A_learn)      # (B, N, gcn_hidden)
            H_list.append(Ht)
        H_seq = torch.stack(H_list, dim=1)  # (B, L, N, gcn_hidden)

        # 3) Temporal encoding -> last hidden per node
        h_last = self.temporal(H_seq)       # (B, N, gru_hidden)

        # 4) Horizon prediction
        y_seq = self.head(h_last)           # (B, N, H)

        return y_seq, A_learn, self._gamma0


# Optional compatibility alias (if some scripts import STGNN)
STGNN = STGNN_LearnOnly

