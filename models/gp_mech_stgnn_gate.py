# models/gp_mech_stgnn.py  (gate-ablation version)
# Adds --gate_type global | node | edge to support Table gate-granularity ablation.
# Everything else is identical to the stability-improved version.

import torch
import torch.nn as nn

from layers.graph import LearnedGraphAttn
from layers.gcn import GraphConv
from layers.temporal import NodeWiseGRUEncoder


class GPMechSTGNN(nn.Module):
    """
    Mechanism-aware STGNN with gate-granularity ablation support.

    gate_type controls the shape of gamma_raw:
      "global"  : scalar       — one gate for the whole graph  (default / paper)
      "node"    : (N,)         — per-node gate
      "edge"    : (N, N)       — per-edge gate

    All other arguments are identical to the stability-improved version.
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

        # stability knobs
        gamma_init: float = 1.5,
        gamma_min: float = 0.3,
        use_graph_input_norm: bool = True,

        # gate ablation
        gate_type: str = "global",   # "global" | "node" | "edge"
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_dim    = int(in_dim)
        self.horizon   = int(horizon)
        self.mode      = str(mode)
        self.gamma_min = float(gamma_min)
        self.gate_type = str(gate_type)

        assert gate_type in ("global", "node", "edge"), \
            f"gate_type must be 'global', 'node', or 'edge', got '{gate_type}'"

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

        # ── Gate parameter — shape depends on gate_type ────────────
        N = self.num_nodes
        if gate_type == "global":
            init_tensor = torch.tensor(float(gamma_init))          # scalar
        elif gate_type == "node":
            init_tensor = torch.full((N,), float(gamma_init))      # (N,)
        else:  # "edge"
            init_tensor = torch.full((N, N), float(gamma_init))    # (N, N)

        self.gamma_raw = nn.Parameter(init_tensor)

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────
    def _gamma(self) -> torch.Tensor:
        """
        Returns gamma in [gamma_min, 1.0].
        Shape: scalar / (N,) / (N,N) depending on gate_type.
        """
        return self.gamma_min + (1.0 - self.gamma_min) * torch.sigmoid(self.gamma_raw)

    def gamma_stats(self) -> dict:
        """
        Diagnostic: call after training to check if node/edge gates collapsed.
        Returns mean, std, min, max of the gamma values.
        """
        g = self._gamma().detach()
        return {
            "shape": tuple(g.shape),
            "mean":  g.mean().item(),
            "std":   g.std().item() if g.numel() > 1 else 0.0,
            "min":   g.min().item(),
            "max":   g.max().item(),
        }

    def _select_adj(
        self,
        A_learn: torch.Tensor,   # (B, N, N)
        A_mech:  torch.Tensor | None,
        B: int,
    ):
        device = A_learn.device

        if self.mode == "learn" or A_mech is None:
            return A_learn, torch.tensor(0.0, device=device)

        A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(device)  # (B, N, N)

        if self.mode == "mech":
            return A_mech_b, torch.tensor(1.0, device=device)

        # prior_residual — broadcast gamma to (B, N, N)
        gamma = self._gamma()   # scalar / (N,) / (N,N)

        if self.gate_type == "global":
            # scalar → broadcasts automatically
            g = gamma
        elif self.gate_type == "node":
            # (N,) → (1, N, 1) so each row (source node) has its own weight
            g = gamma.view(1, self.num_nodes, 1)
        else:  # "edge"
            # (N, N) → (1, N, N)
            g = gamma.unsqueeze(0)

        A_dyn = g * A_mech_b + (1.0 - g) * A_learn
        return A_dyn, gamma

    # ──────────────────────────────────────────────────────────────
    # Regularisation losses (unchanged)
    # ──────────────────────────────────────────────────────────────
    def anchor_loss(
        self,
        A_learn: torch.Tensor,
        A_mech:  torch.Tensor | None,
        entropy_weight: float = 0.0,
    ) -> torch.Tensor:
        device = A_learn.device
        reg = torch.tensor(0.0, device=device)

        if A_mech is not None:
            B        = A_learn.size(0)
            A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(device)
            reg      = reg + ((A_learn - A_mech_b) ** 2).mean()

        if entropy_weight > 0.0:
            H_rows = -(A_learn * (A_learn + 1e-12).log()).sum(dim=-1)
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
        H_node = x_seq.mean(dim=1)
        if self.graph_input_norm is not None:
            H_node = self.graph_input_norm(H_node)
        A_learn = self.graph_learner(H_node)                        # (B, N, N)

        # ── Adjacency selection ────────────────────────────────────
        A_dyn, gamma = self._select_adj(A_learn, A_mech, B)

        # ── Spatial propagation per time step ──────────────────────
        H_list = []
        for t in range(L):
            Xt = x_seq[:, t, :, :]
            Ht = self.gcn(Xt, A_dyn)
            H_list.append(Ht)
        H_seq = torch.stack(H_list, dim=1)                          # (B, L, N, gcn_h)

        # ── Temporal encoding ──────────────────────────────────────
        h_last = self.temporal(H_seq)                               # (B, N, gru_h)

        # ── Prediction ────────────────────────────────────────────
        y_seq = self.head(h_last)                                   # (B, N, H)
        return y_seq, A_learn, gamma


# ──────────────────────────────────────────────────────────────────
def seq_to_pf_ma_gap(y_seq: torch.Tensor):
    pf  = y_seq.sum(dim=-1)
    ma  = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap
