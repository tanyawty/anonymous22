"""Graph structure learner layers."""

import math
import torch
import torch.nn as nn


class LearnedGraphAttn(nn.Module):
    """
    纯数据驱动 learned graph:
      H_node: (B,N,F_in) -> A_learn: (B,N,N)
    """
    def __init__(self, in_dim, hidden_dim=32, symmetric=True):
        super().__init__()
        self.symmetric = symmetric
        self.proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, H_node):
        B, N, F = H_node.shape
        Z = self.proj(H_node)        # (B,N,D)
        Q = Z
        K = Z
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # (B,N,N)
        if self.symmetric:
            scores = 0.5 * (scores + scores.transpose(1, 2))
        A = torch.softmax(scores, dim=-1)
        return A
