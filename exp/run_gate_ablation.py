#!/usr/bin/env python3
# exp/run_gate_ablation.py
#
# Gate-granularity ablation: global vs node vs feat gate.
# Uses DUAL-PATH GCN (gp_mech_stgnn_gate.py) — feature-space mixing.
# gamma is fixed at 0.5 (optimum from gamma sensitivity analysis) so
# the only variable between runs is the gate granularity.
#
# Usage:
#   python exp/run_gate_ablation.py \
#       --price_path data/prices.csv \
#       --macro_path data/macro.csv  \
#       --edges_path data/edges.csv  \
#       --gate_types global node feat \
#       --fixed_gamma 0.5 \
#       --seeds 0 1 2 3 4
#
# Output: summary table printed to stdout + gate_ablation_results.csv

import os
import sys
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_panel_from_two_files(price_path, macro_path):
    df_p = pd.read_csv(price_path)
    date_col_p = df_p.columns[0]
    df_p[date_col_p] = pd.to_datetime(df_p[date_col_p], dayfirst=True, errors="coerce")
    df_p = df_p.set_index(date_col_p).sort_index()
    price_cols = [c for c in df_p.columns if np.issubdtype(df_p[c].dtype, np.number)]
    price_df = df_p[price_cols].astype("float32").replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    df_m = pd.read_csv(macro_path)
    date_col_m = df_m.columns[0]
    df_m[date_col_m] = pd.to_datetime(df_m[date_col_m], dayfirst=True, errors="coerce")
    df_m = df_m.set_index(date_col_m).sort_index()
    macro_cols = [c for c in df_m.columns if np.issubdtype(df_m[c].dtype, np.number)]
    macro_df = df_m[macro_cols].astype("float32").replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    common_idx = price_df.index.intersection(macro_df.index)
    price_df   = price_df.loc[common_idx].sort_index()
    macro_df   = macro_df.loc[common_idx].sort_index()

    rets = np.log(price_df).diff().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(axis=0, how="all").fillna(0.0).astype("float32")
    price_df = price_df.loc[rets.index]
    macro_df = macro_df.loc[rets.index]
    return price_df, rets, macro_df, price_cols, macro_cols


def build_adjacency_from_edges(
    edges_path, node_list, weight_col="w", default_weight=1.0,
    self_loop=True, symmetrize=True,
):
    edges = pd.read_csv(edges_path)
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)
    A = np.zeros((N, N), dtype=np.float32)
    for _, r in edges.iterrows():
        s, t = r.get("source"), r.get("target")
        if s in idx and t in idx:
            w = r.get(weight_col, np.nan)
            if w is None or (isinstance(w, float) and np.isnan(w)):
                w = default_weight
            A[idx[s], idx[t]] += float(w)
    if self_loop:
        np.fill_diagonal(A, A.diagonal() + 1.0)
    if symmetrize:
        A = 0.5 * (A + A.T)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A, dtype=torch.float32)


def seq_to_pf_ma_gap(y_seq):
    pf  = y_seq.sum(dim=-1)
    ma  = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap


def mae_rmse_np(y_true, y_pred):
    diff = (y_pred - y_true).astype(np.float64)
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff * diff)))


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
class PanelGraphDataset(Dataset):
    def __init__(self, returns_df, macro_df, window_size=20, horizon=5):
        self.window_size = int(window_size)
        self.horizon     = int(horizon)
        self.dates       = returns_df.index
        self.T, self.N   = returns_df.shape
        self.ret         = returns_df.values.astype(np.float32)

        self.macro = None
        self.Fm    = 0
        if macro_df is not None and macro_df.shape[1] > 0:
            mac = macro_df.loc[self.dates].values.astype(np.float32)
            mac = pd.DataFrame(mac, index=self.dates).ffill().fillna(0.0).values.astype(np.float32)
            self.macro = mac
            self.Fm    = mac.shape[1]

        self.valid_t = list(range(self.window_size - 1, self.T - self.horizon))

    def __len__(self):
        return len(self.valid_t)

    def _rolling_mean(self, x, w):
        if w <= 1: return x.copy()
        L, N   = x.shape
        out    = np.empty_like(x, dtype=np.float32)
        csum   = np.cumsum(x, axis=0, dtype=np.float64)
        for i in range(L):
            s     = max(0, i - w + 1)
            denom = i - s + 1
            out[i] = ((csum[i] / denom) if s == 0 else
                      ((csum[i] - csum[s-1]) / denom)).astype(np.float32)
        return out

    def _rolling_std(self, x, w):
        if w <= 1: return np.zeros_like(x, dtype=np.float32)
        L, N = x.shape
        out  = np.empty_like(x, dtype=np.float32)
        for i in range(L):
            s     = max(0, i - w + 1)
            out[i] = x[s:i+1].std(axis=0).astype(np.float32)
        return out

    def __getitem__(self, idx):
        t = self.valid_t[idx]
        s, e = t - (self.window_size - 1), t + 1
        ret_win = self.ret[s:e, :]

        feat1 = ret_win
        feat3 = self._rolling_mean(ret_win, 3)
        feat5 = self._rolling_mean(ret_win, 5)
        vol5  = self._rolling_std(ret_win, 5)

        node_feat = np.stack([feat1, feat3, feat5, vol5], axis=-1).astype(np.float32)
        node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=0.0, neginf=0.0)

        L_, N_, F_ = node_feat.shape
        flat = node_feat.reshape(L_ * N_, F_)
        mu, sd = flat.mean(0, keepdims=True), flat.std(0, keepdims=True) + 1e-8
        node_feat = ((flat - mu) / sd).reshape(L_, N_, F_).astype(np.float32)

        if self.macro is not None:
            mac   = self.macro[s:e, :][:, None, :]
            mac   = np.repeat(mac, self.N, axis=1)
            x_seq = np.concatenate([node_feat, mac], axis=-1).astype(np.float32)
        else:
            x_seq = node_feat

        y_seq = self.ret[t+1:t+1+self.horizon, :].T.astype(np.float32)
        return torch.from_numpy(x_seq), torch.from_numpy(y_seq), t


def make_time_split(valid_t, T, train_ratio=0.7, val_ratio=0.15):
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))
    tr, va, te = [], [], []
    for i, t in enumerate(valid_t):
        if   t < train_end: tr.append(i)
        elif t < val_end:   va.append(i)
        else:               te.append(i)
    return tr, va, te


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model_fn, loader, device):
    pf_t, pf_p, ma_t, ma_p, gap_t, gap_p = [], [], [], [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        yp, _, _ = model_fn(x, None)
        for lst, v in [(pf_t,  seq_to_pf_ma_gap(y)[0]),  (pf_p,  seq_to_pf_ma_gap(yp)[0]),
                       (ma_t,  seq_to_pf_ma_gap(y)[1]),  (ma_p,  seq_to_pf_ma_gap(yp)[1]),
                       (gap_t, seq_to_pf_ma_gap(y)[2]),  (gap_p, seq_to_pf_ma_gap(yp)[2])]:
            lst.append(v.cpu().numpy().reshape(-1))
    return {
        "PF":  mae_rmse_np(np.concatenate(pf_t),  np.concatenate(pf_p)),
        "MA":  mae_rmse_np(np.concatenate(ma_t),  np.concatenate(ma_p)),
        "GAP": mae_rmse_np(np.concatenate(gap_t), np.concatenate(gap_p)),
    }


# ─────────────────────────────────────────────────────────────────
# Train one (gate_type, seed) combination
# ─────────────────────────────────────────────────────────────────
def train_one_seed(args, seed: int, gate_type: str):
    set_seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    price_df, returns_df, macro_df, price_cols, _ = load_panel_from_two_files(
        args.price_path, args.macro_path)

    ds = PanelGraphDataset(returns_df, macro_df,
                           window_size=args.window, horizon=args.horizon)
    tr_ids, va_ids, te_ids = make_time_split(ds.valid_t, T=len(returns_df),
                                             train_ratio=args.train_ratio,
                                             val_ratio=args.val_ratio)
    tr_loader = DataLoader(Subset(ds, tr_ids), args.batch, shuffle=True,  drop_last=False)
    va_loader = DataLoader(Subset(ds, va_ids), args.batch, shuffle=False, drop_last=False)
    te_loader = DataLoader(Subset(ds, te_ids), args.batch, shuffle=False, drop_last=False)

    x0, _, _ = ds[0]
    _, N, F_total = x0.shape

    # ── A_mech: structure × training-period correlation weights ───
    A_mech = build_adjacency_from_edges(
        args.edges_path, list(price_cols),
        weight_col=args.weight_col, default_weight=args.default_weight,
    ).to(device)

    train_end  = int(len(returns_df) * args.train_ratio)
    train_rets = returns_df.iloc[:train_end].values.astype(np.float32)
    corr = np.corrcoef(train_rets.T)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    corr_t = torch.tensor(corr, device=device)
    A_mech = A_mech * corr_t
    row_sum = A_mech.sum(dim=1, keepdim=True).clamp(min=1e-8)
    A_mech  = A_mech / row_sum

    # ── Model: dual-path with fixed gamma ─────────────────────────
    from models.gp_mech_stgnn_gate import GPMechSTGNN
    fixed_gamma = args.fixed_gamma if args.fixed_gamma >= 0.0 else None
    model = GPMechSTGNN(
        num_nodes=N, in_dim=F_total, horizon=args.horizon,
        mode="prior_residual",
        graph_hidden=args.graph_hidden,
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        gcn_dropout=args.gcn_dropout,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
        graph_topk=args.graph_topk,
        gamma_min=0.0,
        use_graph_input_norm=True,
        gate_type=gate_type,
        fixed_gamma=fixed_gamma,
    ).to(device)

    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def _loop(loader, train: bool):
        model.train() if train else model.eval()
        total_loss, n = 0.0, 0
        for x_seq, y_seq, _ in loader:
            x_seq, y_seq = x_seq.to(device), y_seq.to(device)
            if train:
                opt.zero_grad()
                y_pred, A_learn, _ = model(x_seq, A_mech)
                loss = mse(y_pred, y_seq)
                if args.lambda_anchor > 0.0:
                    loss = loss + args.lambda_anchor * model.anchor_loss(
                        A_learn, A_mech, entropy_weight=args.entropy_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            else:
                with torch.no_grad():
                    y_pred, _, _ = model(x_seq, A_mech)
                    loss = mse(y_pred, y_seq)
            total_loss += float(loss.detach()) * x_seq.size(0)
            n += x_seq.size(0)
        return total_loss / max(1, n)

    best_val, best_state, bad = 1e18, None, 0
    for ep in range(args.epochs):
        tr = _loop(tr_loader, True)
        va = _loop(va_loader, False)
        if (ep + 1) % 20 == 0:
            gs = model.gamma_stats()
            print(f"  [{gate_type}] seed={seed} ep={ep+1:3d} "
                  f"train={tr:.6f} val={va:.6f} "
                  f"γ_mean={gs['mean']:.4f} γ_std={gs['std']:.4f}")
        if va < best_val:
            best_val  = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Collapse check ─────────────────────────────────────────────
    gs = model.gamma_stats()
    if gate_type != "global" and gs["std"] < 1e-3:
        print(f"  WARNING [{gate_type}] seed={seed}: "
              f"gamma std={gs['std']:.6f} — gate may have collapsed")

    model_fn = lambda x, _: model(x, A_mech)
    test_metrics = evaluate(model_fn, te_loader, device)
    print(f"  [{gate_type}] seed={seed} DONE  "
          f"PF={test_metrics['PF'][0]*100:.4f}  "
          f"GAP={test_metrics['GAP'][0]*100:.4f}  "
          f"γ mean={gs['mean']:.4f} std={gs['std']:.4f}")
    return test_metrics, gs


# ─────────────────────────────────────────────────────────────────
# Main: loop over gate_types × seeds, print summary
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Gate granularity ablation (dual-path)")
    p.add_argument("--price_path", required=True)
    p.add_argument("--macro_path", required=True)
    p.add_argument("--edges_path", required=True)
    p.add_argument("--weight_col",     default="w")
    p.add_argument("--default_weight", type=float, default=1.0)

    p.add_argument("--gate_types",  nargs="+", default=["global", "node", "feat"],
                   choices=["global", "node", "feat"],
                   help="gate granularities to ablate")
    p.add_argument("--fixed_gamma", type=float, default=0.5,
                   help="fix gamma at this value; pass -1 to let model learn it")
    p.add_argument("--seeds",       nargs="+", type=int, default=[0, 1, 2, 3, 4])

    # model
    p.add_argument("--horizon",      type=int,   default=5)
    p.add_argument("--window",       type=int,   default=20)
    p.add_argument("--graph_hidden", type=int,   default=32)
    p.add_argument("--gcn_hidden",   type=int,   default=32)
    p.add_argument("--gru_hidden",   type=int,   default=64)
    p.add_argument("--gcn_dropout",  type=float, default=0.0)
    p.add_argument("--rnn_layers",   type=int,   default=1)
    p.add_argument("--rnn_dropout",  type=float, default=0.0)
    p.add_argument("--graph_topk",   type=int,   default=5)

    # training
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--patience",       type=int,   default=15)
    p.add_argument("--batch",          type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--wd",             type=float, default=1e-4)
    p.add_argument("--lambda_anchor",  type=float, default=0.01)
    p.add_argument("--entropy_weight", type=float, default=0.0)
    p.add_argument("--train_ratio",    type=float, default=0.70)
    p.add_argument("--val_ratio",      type=float, default=0.15)
    p.add_argument("--cpu",            action="store_true")
    p.add_argument("--out_csv",        default="gate_ablation_results.csv")

    args = p.parse_args()

    all_rows = []
    for gt in args.gate_types:
        pf_vals, gap_vals = [], []
        print(f"\n{'='*60}")
        print(f"  gate_type={gt}  fixed_gamma={args.fixed_gamma}")
        print(f"{'='*60}")
        for seed in args.seeds:
            metrics, gs = train_one_seed(args, seed, gt)
            pf_vals.append(metrics["PF"][0] * 100)
            gap_vals.append(metrics["GAP"][0] * 100)

        pf_arr  = np.array(pf_vals)
        gap_arr = np.array(gap_vals)
        row = {
            "gate_type":   gt,
            "fixed_gamma": args.fixed_gamma,
            "PF_mean":     pf_arr.mean(),
            "PF_std":      pf_arr.std(ddof=1),
            "PF_CV%":      pf_arr.std(ddof=1) / pf_arr.mean() * 100,
            "GAP_mean":    gap_arr.mean(),
            "GAP_std":     gap_arr.std(ddof=1),
            "GAP_CV%":     gap_arr.std(ddof=1) / gap_arr.mean() * 100,
        }
        all_rows.append(row)
        print(f"\n  >> {gt}: PF={row['PF_mean']:.4f}±{row['PF_std']:.4f} "
              f"(CV={row['PF_CV%']:.2f}%)  "
              f"GAP={row['GAP_mean']:.4f}±{row['GAP_std']:.4f} "
              f"(CV={row['GAP_CV%']:.2f}%)")

    df = pd.DataFrame(all_rows)
    print("\n\n" + "="*60)
    print("GATE GRANULARITY ABLATION SUMMARY")
    print("="*60)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved → {args.out_csv}")


if __name__ == "__main__":
    main()
