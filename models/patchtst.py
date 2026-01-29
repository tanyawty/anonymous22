# models/patchtst.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction


class PatchTST_Baseline(nn.Module):
    """
    Wrapper around HF PatchTSTForPrediction to match your unified interface.

    Expected input:
      x: (B, L, N + Fm)   where first N channels are asset returns,
                          last Fm channels are macro features.

    Output:
      y_seq: (B, N, H)    predicted future returns sequence
      A_learn: None
      gamma: tensor(0.0)
    """

    def __init__(
        self,
        num_assets: int,
        num_macros: int,
        seq_len: int,
        horizon: int,
        patch_len: int = 8,
        patch_stride: int = 4,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.N = int(num_assets)
        self.Fm = int(num_macros)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        cfg = PatchTSTConfig(
            num_input_channels=self.N + self.Fm,
            context_length=self.seq_len,
            prediction_length=self.horizon,
            patch_length=patch_len,
            patch_stride=patch_stride,
            d_model=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.model = PatchTSTForPrediction(cfg)

    def forward(self, x: torch.Tensor, A_mech=None):
        """
        x: (B, L, N+Fm)
        returns: (y_seq, A_learn=None, gamma=0)
        """
        if x.dim() != 3:
            raise ValueError(f"PatchTST expects x=(B,L,C), got {tuple(x.shape)}")

        B, L, C = x.shape
        if L != self.seq_len:
            raise ValueError(f"seq_len mismatch: got L={L}, expected {self.seq_len}")
        if C != self.N + self.Fm:
            raise ValueError(f"channel mismatch: got C={C}, expected N+Fm={self.N + self.Fm}")

        out = self.model(past_values=x)  # prediction_outputs: (B, H, C)
        yhat = out.prediction_outputs[:, :, : self.N]   # (B, H, N)

        # unify to (B, N, H)
        y_seq = yhat.permute(0, 2, 1).contiguous()

        gamma = torch.tensor(0.0, device=x.device)
        return y_seq, None, gamma

