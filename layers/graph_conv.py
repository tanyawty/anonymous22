"""Graph convolution / message passing layers."""

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        # X: (B,N,F_in), A: (B,N,N)
        Xw = self.linear(X)
        AX = torch.matmul(A, Xw)
        return torch.relu(AX)
