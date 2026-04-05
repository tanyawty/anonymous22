# models/gp_mech_stgnn_edge_gate.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from layers.graph import LearnedGraphAttn
from layers.gcn import GraphConv
from layers.temporal import NodeWiseGRUEncoder


class GPMechSTGNNEdgeGate(nn.Module):
    """
    Mechanism-aware STGNN with edge-wise fusion gate.

    mode:
      - "learn":          A = A_learn
      - "mech":           A = A_mech (prior-only)
      - "prior_residual": A_ij = gamma_ij * A_mech,ij + (1-gamma_ij) * A_learn,ij

    Here gamma_ij is a learnable edge-wise gate (one scalar per edge).

    Inputs:
      x_seq:  (B, L, N, F)
      A_mech: (N, N) or None

    Returns:
      y_seq:   (B, N, H)
      A_learn: (B, N, N)
      gamma:   (N,N) edge-wise gate after sigmoid
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        horizon: int = 5,
        mode: str = "prior_residual",
        graph_hidden: int = 32,
        gcn_hidden: int = 32,
        gru_hidden: int = 64,
        gcn_dropout: float = 0.0,
        rnn_layers: int = 1,
        rnn_dropout: float = 0.0,
        graph_topk: int | None = None,
        symmetric_gate: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.horizon = int(horizon)
        self.mode = str(mode)
        self.symmetric_gate = bool(symmetric_gate)

        # 1) Learned graph
        self.graph_learner = LearnedGraphAttn(
            in_dim=in_dim,
            hidden_dim=graph_hidden,
            symmetric=True,
            topk=graph_topk,
        )

        # 2) Spatial message passing
        self.gcn = GraphConv(
            in_dim=in_dim,
            out_dim=gcn_hidden,
            activation="relu",
            dropout=gcn_dropout,
        )

        # 3) Temporal encoder
        self.temporal = NodeWiseGRUEncoder(
            in_dim=gcn_hidden,
            hidden_dim=gru_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bidirectional=False,
        )

        # 4) Horizon head
        self.head = nn.Linear(self.temporal.out_dim, horizon)

        # edge-wise gate: one scalar per edge
        self.gamma_raw = nn.Parameter(torch.full((self.num_nodes, self.num_nodes), 0.5))

    def _select_adj(self, A_learn: torch.Tensor, A_mech: torch.Tensor | None, B: int):
        device = A_learn.device

        if self.mode == "learn" or A_mech is None:
            gamma = torch.zeros(self.num_nodes, self.num_nodes, device=device)
            return A_learn, gamma

        A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(device)

        if self.mode == "mech":
            gamma = torch.ones(self.num_nodes, self.num_nodes, device=device)
            return A_mech_b, gamma

        gamma = torch.sigmoid(self.gamma_raw).to(device)  # (N,N)
        if self.symmetric_gate:
            gamma = 0.5 * (gamma + gamma.transpose(0, 1))

        gamma_b = gamma.unsqueeze(0)                      # (1,N,N)
        A_dyn = gamma_b * A_mech_b + (1.0 - gamma_b) * A_learn
        return A_dyn, gamma

    def forward(self, x_seq: torch.Tensor, A_mech: torch.Tensor | None):
        if x_seq.dim() != 4:
            raise ValueError(f"x_seq must be (B,L,N,F), got {tuple(x_seq.shape)}")

        B, L, N, F = x_seq.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")
        if F != self.in_dim:
            raise ValueError(f"F mismatch: got {F}, expected {self.in_dim}")

        # learn adjacency
        H_node = x_seq.mean(dim=1)           # (B,N,F)
        A_learn = self.graph_learner(H_node) # (B,N,N)

        # choose adjacency by mode
        A_dyn, gamma = self._select_adj(A_learn, A_mech, B)

        # spatial per t
        H_list = []
        for t in range(L):
            Xt = x_seq[:, t, :, :]           # (B,N,F)
            Ht = self.gcn(Xt, A_dyn)         # (B,N,gcn_hidden)
            H_list.append(Ht)
        H_seq = torch.stack(H_list, dim=1)   # (B,L,N,gcn_hidden)

        # temporal per node
        h_last = self.temporal(H_seq)        # (B,N,gru_hidden)

        # horizon prediction
        y_seq = self.head(h_last)            # (B,N,H)
        return y_seq, A_learn, gamma


def seq_to_pf_ma_gap(y_seq: torch.Tensor):
    """
    y_seq: (B,N,H) future return sequence
    returns pf/ma/gap each (B,N)
    """
    pf = y_seq.sum(dim=-1)
    ma = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap
