# models/stgnn_learn.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from layers.graph import LearnedGraphAttn
from layers.gcn import GraphConv  # 或者 GCNLayer（更稳）
from layers.temporal import NodeWiseGRUEncoder


class STGNN_LearnOnly(nn.Module):
    """
    Learn-only STGNN baseline:
      - Learn adjacency A_learn from window-averaged node features
      - Spatial message passing with A_learn
      - Node-wise GRU over time
      - Linear head predicts future H-step return sequence

    forward:
      x_seq: (B, L, N, F)
      A_mech: ignored
    returns:
      y_seq: (B, N, H)
      A_learn: (B, N, N)
      gamma: tensor(0.0)
    """

    def __init__(
    self,
    num_nodes: int,
    in_dim: int,
    horizon: int,
    gcn_hidden: int = 32,
    gru_hidden: int = 64,
    graph_hidden: int = 64,
    graph_symmetric: bool = True,
    graph_temperature: float = 1.0,
    graph_topk: int | None = None,
    dropout: float = 0.0,          
    gcn_dropout: float | None = None,
    rnn_layers: int = 1,
    rnn_dropout: float | None = None,
):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.horizon = horizon
    if gcn_dropout is None:
    gcn_dropout = dropout
if rnn_dropout is None:
    rnn_dropout = dropout

        # Learn adjacency
        self.graph_learner = LearnedGraphAttn(
            in_dim=in_dim,
            hidden_dim=graph_hidden,
            symmetric=graph_symmetric,
            temperature=graph_temperature,
            topk=graph_topk,
            dropout=0.0,
        )

        # Spatial message passing
        self.gcn = GraphConv(
            in_dim=in_dim,
            out_dim=gcn_hidden,
            activation="relu",
            dropout=gcn_dropout,
        )

        # Temporal encoder (per node)
        self.temporal = NodeWiseGRUEncoder(
            in_dim=gcn_hidden,
            hidden_dim=gru_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bidirectional=False,
        )

        # Horizon head: (B,N,gru_hidden) -> (B,N,H)
        self.head = nn.Linear(self.temporal.out_dim, horizon)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        """
        x_seq: (B, L, N, F)
        """
        if x_seq.dim() != 4:
            raise ValueError(f"x_seq must be (B,L,N,F), got {tuple(x_seq.shape)}")

        B, L, N, F = x_seq.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")
        if F != self.in_dim:
            raise ValueError(f"F mismatch: got {F}, expected {self.in_dim}")

        # 1) Learn adjacency from window summary
        H_node = x_seq.mean(dim=1)                # (B,N,F)
        A_learn = self.graph_learner(H_node)      # (B,N,N)

        # 2) Spatial propagation per time step
        H_list = []
        for t in range(L):
            Xt = x_seq[:, t, :, :]               # (B,N,F)
            Ht = self.gcn(Xt, A_learn)           # (B,N,gcn_hidden)
            H_list.append(Ht)
        H_seq = torch.stack(H_list, dim=1)        # (B,L,N,gcn_hidden)

        # 3) Temporal encoding
        h_last = self.temporal(H_seq)             # (B,N,gru_hidden)

        # 4) Predict future return sequence
        y_seq = self.head(h_last)                 # (B,N,H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y_seq, A_learn, gamma
