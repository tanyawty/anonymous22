# models/fouriergnn.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph import LearnedGraphAttn


# ===== FourierGNN building blocks (kept in this model file) =====

class ComplexLinear(nn.Module):
    """Complex linear layer with real parameters."""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.Wr = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.Wi = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        if bias:
            self.br = nn.Parameter(torch.zeros(out_dim))
            self.bi = nn.Parameter(torch.zeros(out_dim))
        else:
            self.br = None
            self.bi = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.real, x.imag
        yr = xr @ self.Wr - xi @ self.Wi
        yi = xr @ self.Wi + xi @ self.Wr
        if self.br is not None:
            yr = yr + self.br
            yi = yi + self.bi
        return torch.complex(yr, yi)


class FourierGraphOperator(nn.Module):
    """n-invariant FGO parameterized by a complex dxd map."""
    def __init__(self, d_model: int):
        super().__init__()
        self.S = ComplexLinear(d_model, d_model, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(d_model))
        self.bias_i = nn.Parameter(torch.zeros(d_model))

    def forward(self, X_hat: torch.Tensor) -> torch.Tensor:
        return self.S(X_hat) + torch.complex(self.bias_r, self.bias_i)


class FourierGNNEncoder(nn.Module):
    """
    Flatten window into hypervariate graph:
      n = L * N nodes, each node is (time, variate)
    Do FFT along node dimension and stack FGOs.
    """
    def __init__(self, in_dim: int, d_model: int = 128, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(in_dim, d_model)
        self.fgos = nn.ModuleList([FourierGraphOperator(d_model) for _ in range(n_layers)])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B,L,N,F)
        B, L, N, _ = x_seq.shape
        x = self.in_proj(x_seq)                       # (B,L,N,d)
        x = x.contiguous().view(B, L * N, self.d_model)  # (B, n, d)

        X_hat = torch.fft.fft(x.to(torch.float32), dim=1)  # complex along n

        out_time = 0.0
        Y_hat = None
        for k, fgo in enumerate(self.fgos):
            Y_hat = fgo(X_hat) if k == 0 else fgo(Y_hat)
            y_time = torch.fft.ifft(Y_hat, dim=1).real
            y_time = self.act(y_time)
            y_time = self.drop(y_time)
            out_time = out_time + y_time

        out_time = out_time + self.res_scale * x  # residual
        return out_time.view(B, L, N, self.d_model)  # (B,L,N,d)


# ===== Learn-only FourierGNN multi-task model (returns y_seq) =====

class FourierGNN_LearnOnly(nn.Module):
    """
    Learn-only baseline:
      - A_learn is produced by LearnedGraphAttn for logging/consistency only
      - Encoder is FourierGNNEncoder
      - Predict y_seq (B,N,H)

    forward returns: (y_seq, A_learn, gamma=0.0)
    """
    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        horizon: int = 5,
        d_model: int = 128,
        n_layers: int = 3,
        graph_hidden: int = 32,
        dropout: float = 0.1,
        graph_topk: int | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon

        self.graph_learner = LearnedGraphAttn(
            in_dim=in_dim,
            hidden_dim=graph_hidden,
            symmetric=True,
            topk=graph_topk,
        )
        self.encoder = FourierGNNEncoder(in_dim=in_dim, d_model=d_model, n_layers=n_layers, dropout=dropout)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        # x_seq: (B,L,N,F)
        B, L, N, _ = x_seq.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")

        # graph for logging
        H_node = x_seq.mean(dim=1)              # (B,N,F)
        A_learn = self.graph_learner(H_node)    # (B,N,N)

        # encoder
        H = self.encoder(x_seq)                 # (B,L,N,d)
        H_last = H[:, -1, :, :]                 # (B,N,d)
        y_seq = self.head(H_last)               # (B,N,H)

        gamma = torch.tensor(0.0, device=x_seq.device)
        return y_seq, A_learn, gamma

