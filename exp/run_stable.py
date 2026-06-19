# exp/run_stable.py
import os, sys, random, argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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
    price_df = price_df.loc[common_idx].sort_index()
    macro_df = macro_df.loc[common_idx].sort_index()

    rets = np.log(price_df).diff().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(axis=0, how="all").fillna(0.0).astype("float32")
    price_df = price_df.loc[rets.index]
    macro_df = macro_df.loc[rets.index]
    return price_df, rets, macro_df, price_cols, macro_cols


def build_adjacency_from_edges(edges_path, node_list, weight_col="w", default_weight=1.0,
                                self_loop=True, symmetrize=True):
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
    deg = A.sum(axis=1); deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A, dtype=torch.float32)


def seq_to_pf_ma_gap(y_seq):
    return y_seq.sum(-1), y_seq.mean(-1), y_seq.max(-1).values - y_seq.min(-1).values


def mae_rmse_np(y_true, y_pred):
    diff = (y_pred - y_true).astype(np.float64)
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff * diff)))


class PanelGraphDataset(Dataset):
    def __init__(self, returns_df, macro_df, window_size=20, horizon=5):
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.dates = returns_df.index
        self.T, self.N = returns_df.shape
        self.ret = returns_df.values.astype(np.float32)
        self.macro = None; self.Fm = 0
        if macro_df is not None and macro_df.shape[1] > 0:
            mac = macro_df.loc[self.dates].values.astype(np.float32)
            mac = pd.DataFrame(mac, index=self.dates).ffill().fillna(0.0).values.astype(np.float32)
            self.macro = mac; self.Fm = mac.shape[1]
        self.valid_t = list(range(self.window_size - 1, self.T - self.horizon))

    def __len__(self): return len(self.valid_t)

    def _rolling_mean(self, x, w):
        if w <= 1: return x.copy()
        L, N = x.shape; out = np.empty_like(x, dtype=np.float32)
        csum = np.cumsum(x, axis=0, dtype=np.float64)
        for i in range(L):
            s = max(0, i - w + 1); denom = i - s + 1
            out[i] = ((csum[i] / denom) if s == 0 else ((csum[i] - csum[s-1]) / denom)).astype(np.float32)
        return out

    def _rolling_std(self, x, w):
        if w <= 1: return np.zeros_like(x, dtype=np.float32)
        L, N = x.shape; out = np.empty_like(x, dtype=np.float32)
        for i in range(L):
            s = max(0, i - w + 1); out[i] = x[s:i+1].std(axis=0).astype(np.float32)
        return out

    def __getitem__(self, idx):
        t = self.valid_t[idx]; s, e = t - (self.window_size - 1), t + 1
        ret_win = self.ret[s:e, :]
        node_feat = np.stack([ret_win, self._rolling_mean(ret_win, 3),
                              self._rolling_mean(ret_win, 5), self._rolling_std(ret_win, 5)], axis=-1).astype(np.float32)
        node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=0.0, neginf=0.0)
        L_, N_, F_ = node_feat.shape; flat = node_feat.reshape(L_ * N_, F_)
        mu, sd = flat.mean(0, keepdims=True), flat.std(0, keepdims=True) + 1e-8
        node_feat = ((flat - mu) / sd).reshape(L_, N_, F_).astype(np.float32)
        if self.macro is not None:
            mac = self.macro[s:e, :][:, None, :]; mac = np.repeat(mac, self.N, axis=1)
            x_seq = np.concatenate([node_feat, mac], axis=-1).astype(np.float32)
        else:
            x_seq = node_feat
        y_seq = self.ret[t+1:t+1+self.horizon, :].T.astype(np.float32)
        return torch.from_numpy(x_seq), torch.from_numpy(y_seq), t


def make_time_split(valid_t, T, train_ratio=0.7, val_ratio=0.15):
    train_end = int(T * train_ratio); val_end = int(T * (train_ratio + val_ratio))
    tr, va, te = [], [], []
    for i, t in enumerate(valid_t):
        if   t < train_end: tr.append(i)
        elif t < val_end:   va.append(i)
        else:               te.append(i)
    return tr, va, te


@torch.no_grad()
def evaluate(model_fn, loader, device):
    pf_t, pf_p, ma_t, ma_p, gap_t, gap_p = [], [], [], [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        yp, _, _ = model_fn(x, None)
        for lst, v in [(pf_t, seq_to_pf_ma_gap(y)[0]),  (pf_p, seq_to_pf_ma_gap(yp)[0]),
                       (ma_t, seq_to_pf_ma_gap(y)[1]),  (ma_p, seq_to_pf_ma_gap(yp)[1]),
                       (gap_t,seq_to_pf_ma_gap(y)[2]), (gap_p, seq_to_pf_ma_gap(yp)[2])]:
            lst.append(v.cpu().numpy().reshape(-1))
    return {"PF":  mae_rmse_np(np.concatenate(pf_t), np.concatenate(pf_p)),
            "MA":  mae_rmse_np(np.concatenate(ma_t), np.concatenate(ma_p)),
            "GAP": mae_rmse_np(np.concatenate(gap_t),np.concatenate(gap_p))}


def train_one_seed(args, seed):
    set_seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    price_df, returns_df, macro_df, price_cols, _ = load_panel_from_two_files(args.price_path, args.macro_path)
    ds = PanelGraphDataset(returns_df, macro_df, window_size=args.window, horizon=args.horizon)
    tr_ids, va_ids, te_ids = make_time_split(ds.valid_t, T=len(returns_df),
                                             train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    tr_loader = DataLoader(Subset(ds, tr_ids), args.batch, shuffle=True,  drop_last=False)
    va_loader = DataLoader(Subset(ds, va_ids), args.batch, shuffle=False, drop_last=False)
    te_loader = DataLoader(Subset(ds, te_ids), args.batch, shuffle=False, drop_last=False)

    x0, y0, _ = ds[0]; L, N, F_total = x0.shape

    A_mech = None
    if args.mode in ("mech", "prior_residual"):
        A_mech = build_adjacency_from_edges(
            args.edges_path, list(price_cols),
            weight_col=args.weight_col, default_weight=args.default_weight,
        ).to(device)

    from models.gp_mech_stgnn import GPMechSTGNN
    model = GPMechSTGNN(
        num_nodes=N, in_dim=F_total, horizon=args.horizon,
        mode=args.mode,
        graph_hidden=args.graph_hidden, gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden, gcn_dropout=args.gcn_dropout,
        rnn_layers=args.rnn_layers, rnn_dropout=args.rnn_dropout,
        graph_topk=args.graph_topk,
        gamma_init=args.gamma_init, gamma_min=args.gamma_min,
        use_graph_input_norm=not args.no_graph_norm,
        fixed_gamma=args.fixed_gamma,   # ← 新增
    ).to(device)

    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def _loop(loader, train):
        model.train() if train else model.eval()
        total, n = 0.0, 0
        for x_seq, y_seq, _ in loader:
            x_seq, y_seq = x_seq.to(device), y_seq.to(device)
            if train:
                opt.zero_grad()
                y_pred, A_learn, gamma = model(x_seq, A_mech)
                loss = mse(y_pred, y_seq)
                if args.lambda_anchor > 0.0 and A_mech is not None and hasattr(model, 'anchor_loss'):
                    loss = loss + args.lambda_anchor * model.anchor_loss(A_learn, A_mech,
                                                                          entropy_weight=args.entropy_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            else:
                with torch.no_grad():
                    y_pred, A_learn, gamma = model(x_seq, A_mech)
                    loss = mse(y_pred, y_seq)
            total += float(loss.detach()) * x_seq.size(0); n += x_seq.size(0)
        return total / max(1, n)

    best_val, best_state, bad = 1e18, None, 0
    for ep in range(args.epochs):
        tr = _loop(tr_loader, True); va = _loop(va_loader, False)
        g_val = model._gamma().item() if args.mode == "prior_residual" else float(args.mode == "mech")
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep={ep+1:3d}  train={tr:.6f}  val={va:.6f}  gamma={g_val:.4f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience: break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_fn = lambda x, _: model(x, A_mech)
    return evaluate(model_fn, va_loader, device), evaluate(model_fn, te_loader, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",            required=True)
    p.add_argument("--price_path",      required=True)
    p.add_argument("--macro_path",      required=True)
    p.add_argument("--edges_path",      default="")
    p.add_argument("--weight_col",      default="w")
    p.add_argument("--default_weight",  type=float, default=1.0)
    p.add_argument("--window",          type=int,   default=20)
    p.add_argument("--horizon",         type=int,   default=5)
    p.add_argument("--train_ratio",     type=float, default=0.7)
    p.add_argument("--val_ratio",       type=float, default=0.15)
    p.add_argument("--batch",           type=int,   default=32)
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--patience",        type=int,   default=10)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--wd",              type=float, default=0.0)
    p.add_argument("--graph_hidden",    type=int,   default=32)
    p.add_argument("--gcn_hidden",      type=int,   default=32)
    p.add_argument("--gru_hidden",      type=int,   default=64)
    p.add_argument("--gcn_dropout",     type=float, default=0.0)
    p.add_argument("--rnn_layers",      type=int,   default=1)
    p.add_argument("--rnn_dropout",     type=float, default=0.0)
    p.add_argument("--graph_topk",      type=int,   default=-1)
    p.add_argument("--gamma_init",      type=float, default=1.5)
    p.add_argument("--gamma_min",       type=float, default=0.3)
    p.add_argument("--fixed_gamma",     type=float, default=-1.0,  # ← 新增
                   help="-1 = learned; >=0 = fixed value")
    p.add_argument("--lambda_anchor",   type=float, default=0.0)
    p.add_argument("--entropy_weight",  type=float, default=0.0)
    p.add_argument("--no_graph_norm",   action="store_true")
    p.add_argument("--seeds",           default="1,2,3,4,5")
    p.add_argument("--cpu",             action="store_true")
    p.add_argument("--out_csv",         default="")
    args = p.parse_args()

    if args.mode in ("mech", "prior_residual") and not args.edges_path:
        raise ValueError("--edges_path required for mech/prior_residual")
    if args.graph_topk < 0:
        args.graph_topk = None

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for sd in seeds:
        print(f"\n{'='*55}\nseed={sd}\n{'='*55}")
        val_m, test_m = train_one_seed(args, sd)
        for split_name, metrics in [("val", val_m), ("test", test_m)]:
            for task in ["PF", "MA", "GAP"]:
                mae, rmse = metrics[task]
                rows.append(dict(model=f"magn_{args.mode}", seed=sd,
                                 split=split_name, task=task, mae=mae, rmse=rmse))
        print(f"[seed={sd}] test PF={test_m['PF']}  MA={test_m['MA']}  GAP={test_m['GAP']}")

    df = pd.DataFrame(rows)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print("Saved:", args.out_csv)
    else:
        print(df.groupby(["model", "split", "task"])[["mae", "rmse"]].mean())


if __name__ == "__main__":
    main()
