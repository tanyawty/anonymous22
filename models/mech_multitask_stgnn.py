# models/gp_mech_stgnn.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from layers.graph import LearnedGraphAttn
from layers.gcn import GraphConv          # 或者换成 GCNLayer 也行
from layers.temporal import NodeWiseGRUEncoder


class GPMechSTGNN(nn.Module):
    """
    Mechanism-aware STGNN (prior / hybrid / learn).

    mode:
      - "learn":          A = A_learn
      - "mech":           A = A_mech (prior-only)
      - "prior_residual": A = gamma*A_mech + (1-gamma)*A_learn

    Inputs:
      x_seq: (B,L,N,F)
      A_mech: (N,N) or None (learn-only can pass None)

    Returns:
      y_seq: (B,N,H)
      A_learn: (B,N,N)
      gamma: scalar tensor
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
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.horizon = int(horizon)
        self.mode = str(mode)

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

        # 3) Temporal encoder (node-wise GRU)
        self.temporal = NodeWiseGRUEncoder(
            in_dim=gcn_hidden,
            hidden_dim=gru_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bidirectional=False,
        )

        # 4) Horizon head
        self.head = nn.Linear(self.temporal.out_dim, horizon)

        # gating for hybrid
        self.gamma_raw = nn.Parameter(torch.tensor(0.5))

    def _select_adj(self, A_learn: torch.Tensor, A_mech: torch.Tensor | None, B: int):
        device = A_learn.device

        if self.mode == "learn" or A_mech is None:
            gamma = torch.tensor(0.0, device=device)
            return A_learn, gamma

        # A_mech: (N,N) -> (B,N,N)
        A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(device)

        if self.mode == "mech":
            gamma = torch.tensor(1.0, device=device)
            return A_mech_b, gamma

        # prior_residual
        gamma = torch.sigmoid(self.gamma_raw)
        A_dyn = gamma * A_mech_b + (1.0 - gamma) * A_learn
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
        H_node = x_seq.mean(dim=1)                 # (B,N,F)
        A_learn = self.graph_learner(H_node)       # (B,N,N)

        # choose adjacency by mode
        A_dyn, gamma = self._select_adj(A_learn, A_mech, B)

        # spatial per t
        H_list = []
        for t in range(L):
            Xt = x_seq[:, t, :, :]                 # (B,N,F)
            Ht = self.gcn(Xt, A_dyn)               # (B,N,gcn_hidden)
            H_list.append(Ht)
        H_seq = torch.stack(H_list, dim=1)         # (B,L,N,gcn_hidden)

        # temporal per node
        h_last = self.temporal(H_seq)              # (B,N,gru_hidden)

        # horizon prediction
        y_seq = self.head(h_last)                  # (B,N,H)
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
