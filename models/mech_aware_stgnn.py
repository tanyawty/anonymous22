# models/mech_aware_stgnn.py
# Extracted from gp_mech_multitask_stgnn.py and made import-friendly.
# This file defines ONLY model components (no data loading, no training loop).

from __future__ import annotations

import math
import torch
import torch.nn as nn


class LearnedGraphAttn(nn.Module):
    """
    Data-driven learned adjacency via scaled dot-product attention.

    Input:
        H_node: (B, N, F_in)
    Output:
        A_learn: (B, N, N) row-stochastic (softmax over last dim)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 32, symmetric: bool = True):
        super().__init__()
        self.symmetric = symmetric
        self.proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, H_node: torch.Tensor) -> torch.Tensor:
        Z = self.proj(H_node)  # (B,N,D)
        scores = torch.matmul(Z, Z.transpose(1, 2)) / self.scale  # (B,N,N)
        if self.symmetric:
            scores = 0.5 * (scores + scores.transpose(1, 2))
        A = torch.softmax(scores, dim=-1)
        return A


class GraphConv(nn.Module):
    """
    Simple graph convolution: ReLU(A @ (XW)).
    X: (B,N,F_in), A: (B,N,N)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        Xw = self.linear(X)
        AX = torch.matmul(A, Xw)
        return torch.relu(AX)


class MechAware_GP_STGNN_MultiTask(nn.Module):
    """
    Mechanism-aware multi-task STGNN (sequence prediction version).

    mode:
        - "learn":          use A_learn only
        - "mech":           use A_mech only
        - "prior_residual": A_dyn = gamma*A_mech + (1-gamma)*A_learn, gamma in (0,1)

    Input:
        x_seq:  (B, L, N, F_in)
        A_mech: (N, N)  (assumed normalized if you want)
    Output:
        y_seq:  (B, N, H) future return sequence
        A_learn:(B, N, N)
        gamma:  scalar tensor
    """
    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        mode: str = "prior_residual",
        gcn_hidden: int = 32,
        gru_hidden: int = 64,
        graph_hidden: int = 32,
        horizon: int = 5,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.mode = str(mode)
        self.horizon = int(horizon)

        self.graph_learner = LearnedGraphAttn(in_dim=self.in_dim, hidden_dim=graph_hidden, symmetric=True)
        self.gcn = GraphConv(self.in_dim, gcn_hidden)
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)
        self.fc_seq = nn.Linear(gru_hidden, self.horizon)

        # gamma_raw -> sigmoid -> (0,1)
        self.gamma_raw = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_seq: torch.Tensor, A_mech: torch.Tensor):
        B, L, N, _ = x_seq.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")

        # Learn graph from window summary (mean over time)
        H_node = x_seq.mean(dim=1)           # (B,N,F_in)
        A_learn = self.graph_learner(H_node) # (B,N,N)

        if A_mech is None:
            # allow None for learn-only usage
            A_mech_batch = None
        else:
            A_mech_batch = A_mech.unsqueeze(0).expand(B, -1, -1)

        if self.mode == "learn" or A_mech_batch is None:
            gamma = torch.tensor(0.0, device=x_seq.device)
            A_dyn = A_learn
        elif self.mode == "mech":
            gamma = torch.tensor(1.0, device=x_seq.device)
            A_dyn = A_mech_batch
        else:  # "prior_residual"
            gamma = torch.sigmoid(self.gamma_raw)
            A_dyn = gamma * A_mech_batch + (1.0 - gamma) * A_learn

        # GCN over each time step
        gcn_outputs = []
        for t in range(L):
            Xt = x_seq[:, t, :, :]      # (B,N,F_in)
            Ht = self.gcn(Xt, A_dyn)    # (B,N,gcn_hidden)
            gcn_outputs.append(Ht)
        H = torch.stack(gcn_outputs, dim=1)  # (B,L,N,Fg)

        # GRU per node: reshape to (B*N, L, Fg)
        H_reshape = H.permute(0, 2, 1, 3).contiguous().view(B * N, L, H.shape[-1])
        _, h_last = self.gru(H_reshape)        # h_last: (1, B*N, gru_hidden)
        h_last = h_last.squeeze(0)             # (B*N, gru_hidden)

        y_seq = self.fc_seq(h_last)            # (B*N, H)
        y_seq = y_seq.view(B, N, self.horizon) # (B, N, H)

        return y_seq, A_learn, gamma


def seq_to_pf_ma_gap(y_seq: torch.Tensor):
    """
    Convert a future return sequence into PF/MA/GAP tasks.

    y_seq: (B, N, H)
    PF  = sum over horizon
    MA  = mean over horizon
    GAP = max - min over horizon
    """
    pf = y_seq.sum(dim=-1)
    ma = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap
