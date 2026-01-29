# models/classical_baselines.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph import LearnedGraphAttn
from layers.temporal import NodeWiseGRUEncoder, NodeWiseLSTMEncoder, TemporalMLP


class _BasePanelModel(nn.Module):
    """
    Base wrapper:
      - Optionally produce A_learn for logging
      - Always return (y_seq, A_learn, gamma)
    """
    def __init__(self, num_nodes: int, in_dim: int, horizon: int, use_graph_logging: bool = True, graph_hidden: int = 32):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.horizon = horizon

        self.use_graph_logging = use_graph_logging
        self.graph_learner = None
        if use_graph_logging:
            self.graph_learner = LearnedGraphAttn(in_dim=in_dim, hidden_dim=graph_hidden, symmetric=True)

    def _log_graph(self, x_seq: torch.Tensor):
        if self.graph_learner is None:
            return None
        H_node = x_seq.mean(dim=1)  # (B,N,F)
        return self.graph_learner(H_node)  # (B,N,N)


class GRU_Baseline(_BasePanelModel):
    """
    Temporal-only baseline (shared GRU per node):
      x_seq: (B,L,N,F) -> GRU -> (B,N,H)
    """
    def __init__(self, num_nodes, in_dim, horizon, hidden_dim=64, rnn_layers=1, rnn_dropout=0.0,
                 use_graph_logging=True, graph_hidden=32):
        super().__init__(num_nodes, in_dim, horizon, use_graph_logging, graph_hidden)
        self.temporal = NodeWiseGRUEncoder(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=rnn_layers, dropout=rnn_dropout)
        self.head = nn.Linear(self.temporal.out_dim, horizon)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        # x_seq: (B,L,N,F)
        B, L, N, Fdim = x_seq.shape
        if N != self.num_nodes or Fdim != self.in_dim:
            raise ValueError(f"Input shape mismatch, got {tuple(x_seq.shape)}, expect N={self.num_nodes}, F={self.in_dim}")

        A_learn = self._log_graph(x_seq)  # for logging only

        h_last = self.temporal(x_seq)     # (B,N,Hid)
        y_seq = self.head(h_last)         # (B,N,H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y_seq, A_learn, gamma


class LSTM_Baseline(_BasePanelModel):
    """
    Temporal-only baseline (shared LSTM per node).
    """
    def __init__(self, num_nodes, in_dim, horizon, hidden_dim=64, rnn_layers=1, rnn_dropout=0.0,
                 use_graph_logging=True, graph_hidden=32):
        super().__init__(num_nodes, in_dim, horizon, use_graph_logging, graph_hidden)
        self.temporal = NodeWiseLSTMEncoder(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=rnn_layers, dropout=rnn_dropout)
        self.head = nn.Linear(self.temporal.out_dim, horizon)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        B, L, N, Fdim = x_seq.shape
        if N != self.num_nodes or Fdim != self.in_dim:
            raise ValueError(f"Input shape mismatch, got {tuple(x_seq.shape)}, expect N={self.num_nodes}, F={self.in_dim}")

        A_learn = self._log_graph(x_seq)

        h_last = self.temporal(x_seq)     # (B,N,Hid)
        y_seq = self.head(h_last)         # (B,N,H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y_seq, A_learn, gamma


class MLP_Baseline(_BasePanelModel):
    """
    Very fast temporal baseline: flatten (L,F) -> MLP per node.
    """
    def __init__(self, num_nodes, in_dim, horizon, window_len, hidden_dim=128, dropout=0.0,
                 use_graph_logging=True, graph_hidden=32):
        super().__init__(num_nodes, in_dim, horizon, use_graph_logging, graph_hidden)
        self.temporal = TemporalMLP(window_len=window_len, in_dim=in_dim, out_dim=horizon, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        B, L, N, Fdim = x_seq.shape
        if N != self.num_nodes or Fdim != self.in_dim:
            raise ValueError(f"Input shape mismatch, got {tuple(x_seq.shape)}, expect N={self.num_nodes}, F={self.in_dim}")

        A_learn = self._log_graph(x_seq)

        y_seq = self.temporal(x_seq)      # (B,N,H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y_seq, A_learn, gamma


class Transformer_Baseline(_BasePanelModel):
    """
    Transformer encoder per node (shared weights):
      - flatten (B,N,L,F) -> (B*N,L,F)
      - TransformerEncoder -> take last token -> (B,N,H)
    """
    def __init__(self, num_nodes, in_dim, horizon, d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 use_graph_logging=True, graph_hidden=32):
        super().__init__(num_nodes, in_dim, horizon, use_graph_logging, graph_hidden)
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(4 * d_model, 128),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        B, L, N, Fdim = x_seq.shape
        if N != self.num_nodes or Fdim != self.in_dim:
            raise ValueError(f"Input shape mismatch, got {tuple(x_seq.shape)}, expect N={self.num_nodes}, F={self.in_dim}")

        A_learn = self._log_graph(x_seq)

        x = x_seq.permute(0, 2, 1, 3).contiguous().view(B * N, L, Fdim)  # (B*N,L,F)
        h = self.proj(x)
        h = self.encoder(h)
        h_last = h[:, -1, :]                       # (B*N,d)
        y = self.head(h_last).view(B, N, self.horizon)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y, A_learn, gamma


class TCN_Baseline(_BasePanelModel):
    """
    Simple TCN per node (shared weights):
      x: (B*N, F, L) -> TCN -> last -> (B,N,H)
    """
    def __init__(self, num_nodes, in_dim, horizon, channel=64, levels=3, k=3, dropout=0.0,
                 use_graph_logging=True, graph_hidden=32):
        super().__init__(num_nodes, in_dim, horizon, use_graph_logging, graph_hidden)

        layers = []
        C_in = in_dim
        for i in range(levels):
            dilation = 2 ** i
            pad = (k - 1) * dilation
            conv = nn.Conv1d(C_in, channel, kernel_size=k, padding=pad, dilation=dilation)
            bn = nn.BatchNorm1d(channel)
            layers += [conv, bn, nn.ReLU(), nn.Dropout(dropout)]
            C_in = channel
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channel, horizon)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        B, L, N, Fdim = x_seq.shape
        if N != self.num_nodes or Fdim != self.in_dim:
            raise ValueError(f"Input shape mismatch, got {tuple(x_seq.shape)}, expect N={self.num_nodes}, F={self.in_dim}")

        A_learn = self._log_graph(x_seq)

        x = x_seq.permute(0, 2, 3, 1).contiguous().view(B * N, Fdim, L)  # (B*N,F,L)
        h = self.tcn(x)                                                 # (B*N,C,L')
        h_last = h[:, :, -1]                                            # (B*N,C)
        y = self.head(h_last).view(B, N, self.horizon)                  # (B,N,H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y, A_learn, gamma

