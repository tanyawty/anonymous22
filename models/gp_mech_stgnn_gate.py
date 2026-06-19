# models/gp_mech_stgnn_gate.py  (dual-path gate-granularity version)
#
# Architecture: dual-path GCN, feature-space mixing.
#   H_learn = GCN_learn(X, A_learn)          data-driven branch
#   H_mech  = GCN_mech (X, A_mech)           mechanism-prior branch
#   H       = γ · H_mech + (1-γ) · H_learn  feature-space fusion
#
# gate_type controls the shape of γ:
#   "global"  : scalar        — one gate for the whole model      (paper default)
#   "node"    : (N,)          — per-node  mixing weight
#   "feat"    : (gcn_hidden,) — per-feature-dim mixing weight

import torch
import torch.nn as nn

from layers.graph import LearnedGraphAttn
from layers.gcn import GraphConv
from layers.temporal import NodeWiseGRUEncoder


class GPMechSTGNN(nn.Module):
    def __init__(
        self,
        num_nodes, in_dim, horizon=5, mode="prior_residual",
        graph_hidden=32, gcn_hidden=32, gru_hidden=64,
        gcn_dropout=0.0, rnn_layers=1, rnn_dropout=0.0,
        graph_topk=None,
        gamma_init=0.0, gamma_min=0.0,
        use_graph_input_norm=True,
        gate_type="global",
        fixed_gamma=None,
    ):
        super().__init__()
        self.num_nodes   = int(num_nodes)
        self.in_dim      = int(in_dim)
        self.horizon     = int(horizon)
        self.mode        = str(mode)
        self.gamma_min   = float(gamma_min)
        self.gate_type   = str(gate_type)
        self.fixed_gamma = fixed_gamma
        self.gcn_hidden  = int(gcn_hidden)

        assert gate_type in ("global", "node", "feat"), \
            f"gate_type must be 'global', 'node', or 'feat', got '{gate_type}'"

        self.graph_input_norm = nn.LayerNorm(in_dim) if use_graph_input_norm else None
        self.graph_learner = LearnedGraphAttn(
            in_dim=in_dim, hidden_dim=graph_hidden, symmetric=True, topk=graph_topk)

        _gcn_kwargs = dict(in_dim=in_dim, out_dim=gcn_hidden, activation="relu", dropout=gcn_dropout)
        self.gcn_learn = GraphConv(**_gcn_kwargs)
        self.gcn_mech  = GraphConv(**_gcn_kwargs)

        self.temporal = NodeWiseGRUEncoder(
            in_dim=gcn_hidden, hidden_dim=gru_hidden,
            num_layers=rnn_layers, dropout=rnn_dropout, bidirectional=False)
        self.head = nn.Linear(self.temporal.out_dim, horizon)

        if fixed_gamma is None:
            N, d = self.num_nodes, self.gcn_hidden
            if gate_type == "global":
                init_tensor = torch.tensor(float(gamma_init))
            elif gate_type == "node":
                init_tensor = torch.full((N,), float(gamma_init))
            else:
                init_tensor = torch.full((d,), float(gamma_init))
            self.gamma_raw = nn.Parameter(init_tensor)
        else:
            self.gamma_raw = None

    def _gamma(self):
        if self.fixed_gamma is not None:
            dev = next(self.parameters()).device if len(list(self.parameters())) else torch.device("cpu")
            return torch.tensor(float(self.fixed_gamma), device=dev)
        return self.gamma_min + (1.0 - self.gamma_min) * torch.sigmoid(self.gamma_raw)

    def gamma_stats(self):
        g = self._gamma().detach()
        return {
            "gate_type": self.gate_type,
            "shape": tuple(g.shape),
            "mean":  g.mean().item(),
            "std":   g.std().item() if g.numel() > 1 else 0.0,
            "min":   g.min().item(),
            "max":   g.max().item(),
        }

    def _mix_feat(self, H_learn, H_mech):
        gamma = self._gamma()
        if self.gate_type == "global":
            g = gamma
        elif self.gate_type == "node":
            g = gamma.view(1, self.num_nodes, 1)
        else:
            g = gamma.view(1, 1, self.gcn_hidden)
        return g * H_mech + (1.0 - g) * H_learn

    def anchor_loss(self, A_learn, A_mech, entropy_weight=0.0):
        device = A_learn.device
        reg = torch.tensor(0.0, device=device)
        if A_mech is not None:
            B = A_learn.size(0)
            A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(device)
            reg = reg + ((A_learn - A_mech_b) ** 2).mean()
        if entropy_weight > 0.0:
            H_rows = -(A_learn * (A_learn + 1e-12).log()).sum(dim=-1)
            reg = reg + entropy_weight * H_rows.mean()
        return reg

    def forward(self, x_seq, A_mech):
        if x_seq.dim() != 4:
            raise ValueError(f"x_seq must be (B,L,N,F), got {tuple(x_seq.shape)}")
        B, L, N, F = x_seq.shape

        H_node = x_seq.mean(dim=1)
        if self.graph_input_norm is not None:
            H_node = self.graph_input_norm(H_node)
        A_learn = self.graph_learner(H_node)

        use_mech  = (self.mode in ("mech", "prior_residual")) and (A_mech is not None)
        use_learn = (self.mode in ("learn", "prior_residual"))

        H_learn_list, H_mech_list = [], []
        if use_learn:
            for t in range(L):
                H_learn_list.append(self.gcn_learn(x_seq[:, t], A_learn))
        if use_mech:
            A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(x_seq.device)
            for t in range(L):
                H_mech_list.append(self.gcn_mech(x_seq[:, t], A_mech_b))

        if use_mech and use_learn:
            H_seq = torch.stack(
                [self._mix_feat(H_learn_list[t], H_mech_list[t]) for t in range(L)], dim=1)
            gamma_out = self._gamma()
        elif use_mech:
            H_seq     = torch.stack(H_mech_list, dim=1)
            gamma_out = torch.tensor(1.0, device=x_seq.device)
        else:
            H_seq     = torch.stack(H_learn_list, dim=1)
            gamma_out = torch.tensor(0.0, device=x_seq.device)

        h_last = self.temporal(H_seq)
        y_seq  = self.head(h_last)
        return y_seq, A_learn, gamma_out


def seq_to_pf_ma_gap(y_seq):
    pf  = y_seq.sum(dim=-1)
    ma  = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap
