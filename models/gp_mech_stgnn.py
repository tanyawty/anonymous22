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
        graph_topk=None, gamma_init=1.5, gamma_min=0.3,
        use_graph_input_norm=True, gate_type="global",
        fixed_gamma=-1.0,
    ):
        super().__init__()
        self.num_nodes   = int(num_nodes)
        self.in_dim      = int(in_dim)
        self.horizon     = int(horizon)
        self.mode        = str(mode)
        self.gamma_min   = float(gamma_min)
        self.gate_type   = str(gate_type)
        self.fixed_gamma = float(fixed_gamma)

        self.graph_input_norm = nn.LayerNorm(in_dim) if use_graph_input_norm else None
        self.graph_learner = LearnedGraphAttn(
            in_dim=in_dim, hidden_dim=graph_hidden,
            symmetric=True, topk=graph_topk)   # ← topk 传进来
        self.gcn = GraphConv(
            in_dim=in_dim, out_dim=gcn_hidden,
            activation="relu", dropout=gcn_dropout)
        self.temporal = NodeWiseGRUEncoder(
            in_dim=gcn_hidden, hidden_dim=gru_hidden,
            num_layers=rnn_layers, dropout=rnn_dropout, bidirectional=False)
        self.head = nn.Linear(self.temporal.out_dim, horizon)
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))

    def _gamma(self):
        if self.fixed_gamma >= 0:
            return torch.tensor(self.fixed_gamma, device=self.gamma_raw.device)
        return self.gamma_min + (1.0 - self.gamma_min) * torch.sigmoid(self.gamma_raw)

    def gamma_stats(self):
        return {"value": self._gamma().detach().item()}

    @staticmethod
    def _row_norm(A):
        return A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _select_adj(self, A_learn, A_mech, B):
        device = A_learn.device
        if self.mode == "learn" or A_mech is None:
            return A_learn, torch.tensor(0.0, device=device)

        A_mech_b = A_mech.unsqueeze(0).expand(B, -1, -1).to(device)

        if self.mode == "mech":
            return A_mech_b, torch.tensor(1.0, device=device)

        # ── Fix: row-normalize A_mech 和 A_learn 同量纲 ──────────────
        A_mech_b = self._row_norm(A_mech_b)

        gamma = self._gamma()
        if self.gate_type == "node":
            g = gamma.view(1, self.num_nodes, 1)
        elif self.gate_type == "edge":
            g = gamma.unsqueeze(0)
        else:
            g = gamma

        return g * A_mech_b + (1.0 - g) * A_learn, gamma

    def forward(self, x_seq, A_mech):
        B, L, N, F = x_seq.shape
        H_node = x_seq.mean(dim=1)
        if self.graph_input_norm is not None:
            H_node = self.graph_input_norm(H_node)
        A_learn = self.graph_learner(H_node)
        A_dyn, gamma = self._select_adj(A_learn, A_mech, B)
        H_list = [self.gcn(x_seq[:, t], A_dyn) for t in range(L)]
        h_last = self.temporal(torch.stack(H_list, dim=1))
        return self.head(h_last), A_learn, gamma


def seq_to_pf_ma_gap(y_seq):
    return (y_seq.sum(-1), y_seq.mean(-1),
            y_seq.max(-1).values - y_seq.min(-1).values)
