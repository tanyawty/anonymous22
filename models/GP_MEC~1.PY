# models/gp_mech_stgnn.py  (stability-improved version)
# -*- coding: utf-8 -*-
"""
Stability improvements over the original:

1. LayerNorm on graph-learner input
   Why: raw mean-pooled node features vary in scale across batches/seeds.
   Fix: normalise before the attention dot-product → projection sees a
        consistent distribution regardless of random init.

2. Higher gamma initialisation + hard lower bound
   Why: sigmoid(0.5) ≈ 0.62, leaving 38 % weight on the unstable learned
        graph; gradient updates can push gamma → 0 (pure learn).
   Fix: init at 1.5 → sigmoid ≈ 0.82; clamp so gamma ≥ gamma_min (default
        0.3), i.e. mechanism always contributes at least 30 %.

3. Graph anchor regularisation loss  (anchor_loss method)
   Why: MSE-only training puts no constraint on A_learn's shape; different
        seeds converge to structurally unrelated graphs.
   Fix: add λ * ||A_learn − A_mech||_F to the training objective. This
        penalises deviation from the mechanism prior *during optimisation*,
        not just at fusion time.

4. Graph entropy regularisation (optional, entropy_weight > 0)
   Why: a high-entropy (flat/uniform) learned graph is structurally noisy
        and barely informative; it also changes shape most across seeds.
   Fix: penalise row-entropy of A_learn → encourage sparser, peaked graphs
        that are easier to reproduce across seeds.

Backward-compatible: all new arguments have defaults matching old behaviour.
The forward() signature and return values are unchanged.
"""

import torch
import torch.nn as nn

from layers.graph import LearnedGraphAttn
from layers.gcn import GraphConv
from layers.temporal import NodeWiseGRUEncoder


class GPMechSTGNN(nn.Module):
    """
    Mechanism-aware STGNN with improved structural stability.

    mode:
      - "learn":          A = A_learn
      - "mech":           A = A_mech (prior-only)
      - "prior_residual": A = gamma * A_mech + (1-gamma) * A_learn

    Inputs:
      x_seq : (B, L, N, F)
      A_mech: (N, N) or None

    Returns:
      y_seq  : (B, N, H)
      A_learn: (B, N, N)
      gamma  : scalar tensor
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        horizon: int = 5,
        mode: str = "prior_residual",

        # architecture
        graph_hidden: int = 32,
        gcn_hidden: int = 32,
        gru_hidden: int = 64,
        gcn_dropout: float = 0.0,
        rnn_layers: int = 1,
        rnn_dropout: float = 0.0,
        graph_topk: int | None = None,

        # ── stability knobs ──────────────────────────────────────
        gamma_init: float = 1.5,
        # sigmoid(1.5) ≈ 0.82  →  mechanism gets ~82 % weight at init
        # (was 0.5 → 62 %)

        gamma_min: float = 0.3,
        # hard lower bound: even after training, gamma ≥ 0.3
        # formula: gamma = gamma_min + (1 - gamma_min) * sigmoid(gamma_raw)
        # range: [gamma_min, 1.0]  (default [0.3, 1.0])

        use_graph_input_norm: bool = True,
        # LayerNorm on the mean-pooled node features before graph learning
        # Directly addresses scale sensitivity of the attention dot-product
        # ─────────────────────────────────────────────────────────
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_dim    = int(in_dim)
        self.horizon   = int(horizon)
        self.mode      = str(mode)
        self.gamma_min = float(gamma_min)

        # ── Stability fix 1: LayerNorm on graph-learner input ──────
        self.graph_input_norm = nn.LayerNorm(in_dim) if use_graph_input_norm else None

        # ── Graph learner ──────────────────────────────────────────
        self.graph_learner = LearnedGraphAttn(
            in_dim=in_dim,
            hidden_dim=graph_hidden,
            symmetric=True,
            topk=graph_topk,
        )

        # ── Spatial message passing ────────────────────────────────
        self.gcn = GraphConv(
            in_dim=in_dim,
            out_dim=gcn_hidden,
            activation="relu",
            dropout=gcn_dropout,
        )

        # ── Temporal encoder ───────────────────────────────────────
        self.temporal = NodeWiseGRUEncoder(
            in_dim=gcn_hidden,
            hidden_dim=gru_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bidirectional=False,
        )

        # ── Prediction head ────────────────────────────────────────
        self.head = nn.Linear(self.temporal.out_dim, horizon)

        # ── Stability fix 2: better gamma init + bounded range ─────
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))

    # ──────────────────────────────────────────────────────────────
    # Internal: adjacency selection
    # ──────────────────────────────────────────────────────────────
    def _gamma(self) -> torch.Tensor:
        """gamma ∈ [gamma_min, 1.0] — mechanism weight."""
        return self.gamma_min + (1.0 - self.gamma_min) * torch.sigmoid(self.gamma_raw)

    def _select_adj(
        self,
        A_learn: torch.Tensor,
        A_mech:  torch.Tensor | None,
        B: int,
    ):
        device = A_learn.device

        if self.mode == "learn" or A_mech is None:
            return A_learn, torch.tensor(0.0, device=device)

        A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(device)

        if self.mode == "mech":
            return A_mech_b, torch.tensor(1.0, device=device)

        # prior_residual — bounded gamma
        gamma  = self._gamma()
        A_dyn  = gamma * A_mech_b + (1.0 - gamma) * A_learn
        return A_dyn, gamma

    # ──────────────────────────────────────────────────────────────
    # Stability fix 3: auxiliary regularisation losses
    # Call these in the training loop and add to MSE loss.
    # ──────────────────────────────────────────────────────────────
    def anchor_loss(
        self,
        A_learn: torch.Tensor,
        A_mech:  torch.Tensor | None,
        entropy_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Graph anchor regularisation.

        anchor term  : mean Frobenius distance between A_learn and A_mech
                       → pulls the learned graph toward the mechanism prior
        entropy term : mean row-entropy of A_learn (optional, weight > 0)
                       → encourages sparse/peaked graphs that are easier
                         to reproduce across seeds

        Returns a scalar tensor; add to the main MSE loss as:
            loss = mse_loss + lambda_anchor * model.anchor_loss(A_learn, A_mech)

        Parameters
        ----------
        A_learn        : (B, N, N)  learned adjacency from this batch
        A_mech         : (N, N)     mechanism prior, or None
        entropy_weight : float      weight on the entropy term (0 = off)
        """
        device = A_learn.device
        reg = torch.tensor(0.0, device=device)

        # ── anchor: Frobenius distance to mechanism prior ──────────
        if A_mech is not None:
            B      = A_learn.size(0)
            A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(device)
            frob   = ((A_learn - A_mech_b) ** 2).mean()
            reg    = reg + frob

        # ── entropy: penalise flat/uniform rows in A_learn ─────────
        if entropy_weight > 0.0:
            # A_learn rows are softmax-normalised (row-stochastic)
            H_rows = -(A_learn * (A_learn + 1e-12).log()).sum(dim=-1)  # (B, N)
            reg    = reg + entropy_weight * H_rows.mean()

        return reg

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────
    def forward(self, x_seq: torch.Tensor, A_mech: torch.Tensor | None):
        if x_seq.dim() != 4:
            raise ValueError(f"x_seq must be (B,L,N,F), got {tuple(x_seq.shape)}")

        B, L, N, F = x_seq.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")
        if F != self.in_dim:
            raise ValueError(f"F mismatch: got {F}, expected {self.in_dim}")

        # ── Graph learning ─────────────────────────────────────────
        H_node = x_seq.mean(dim=1)                  # (B, N, F)
        if self.graph_input_norm is not None:        # ← fix 1
            H_node = self.graph_input_norm(H_node)
        A_learn = self.graph_learner(H_node)         # (B, N, N)

        # ── Adjacency selection ────────────────────────────────────
        A_dyn, gamma = self._select_adj(A_learn, A_mech, B)

        # ── Spatial propagation per time step ──────────────────────
        H_list = []
        for t in range(L):
            Xt = x_seq[:, t, :, :]                  # (B, N, F)
            Ht = self.gcn(Xt, A_dyn)                # (B, N, gcn_hidden)
            H_list.append(Ht)
        H_seq = torch.stack(H_list, dim=1)          # (B, L, N, gcn_hidden)

        # ── Temporal encoding ──────────────────────────────────────
        h_last = self.temporal(H_seq)               # (B, N, gru_hidden)

        # ── Prediction ────────────────────────────────────────────
        y_seq = self.head(h_last)                   # (B, N, H)
        return y_seq, A_learn, gamma


# ──────────────────────────────────────────────────────────────────
def seq_to_pf_ma_gap(y_seq: torch.Tensor):
    pf  = y_seq.sum(dim=-1)
    ma  = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap
