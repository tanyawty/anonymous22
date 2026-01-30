# exp/01_forecasting/run_gp_mech.py
# Run main model (GPMechSTGNN) for PF/MA/GAP on val/test across multiple seeds.
#
# Usage (Colab):
#   %cd /content/anonymous22
#   !python exp/01_forecasting/run_gp_mech.py \
#       --mode mech \
#       --price_path dataset/panel_20.csv \
#       --macro_path dataset/panel_macro.csv \
#       --edges_path dataset/derived/edges_candidates_20.csv \
#       --window 20 --horizon 5 --epochs 10 --batch 32 \
#       --seeds 1,2,3,4,5 \
#       --out_csv results/gp_mech_stgnn_mech_panel20_h5.csv

import os
import sys
import math
import random
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# --- make repo root importable (critical) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)


# -----------------------
# Repro
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Load panel + macro
# -----------------------
def load_panel_from_two_files(price_path: str, macro_path: str):
    df_p = pd.read_csv(price_path)
    date_col_p = df_p.columns[0]
    df_p[date_col_p] = pd.to_datetime(df_p[date_col_p], dayfirst=True, errors="coerce")
    df_p = df_p.set_index(date_col_p).sort_index()

    price_cols = [c for c in df_p.columns if np.issubdtype(df_p[c].dtype, np.number)]
    price_df = (
        df_p[price_cols]
        .astype("float32")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    df_m = pd.read_csv(macro_path)
    date_col_m = df_m.columns[0]
    df_m[date_col_m] = pd.to_datetime(df_m[date_col_m], dayfirst=True, errors="coerce")
    df_m = df_m.set_index(date_col_m).sort_index()

    macro_cols = [c for c in df_m.columns if np.issubdtype(df_m[c].dtype, np.number)]
    macro_df = (
        df_m[macro_cols]
        .astype("float32")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    common_idx = price_df.index.intersection(macro_df.index)
    price_df = price_df.loc[common_idx].sort_index()
    macro_df = macro_df.loc[common_idx].sort_index()

    # log returns
    rets = np.log(price_df).diff().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(axis=0, how="all").fillna(0.0).astype("float32")

    price_df = price_df.loc[rets.index]
    macro_df = macro_df.loc[rets.index]

    return price_df, rets, macro_df, price_cols, macro_cols


# -----------------------
# Build adjacency from edges csv
#   Expect columns: source, target, (optional) weight_col
#   Node names must match price_df columns (e.g., px_wti ...)
# -----------------------
def build_adjacency_from_edges(
    edges_path: str,
    node_list: List[str],
    weight_col: str = "w",
    default_weight: float = 1.0,
    self_loop: bool = True,
    symmetrize: bool = True,
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

    # symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A, dtype=torch.float32)  # (N,N)


# -----------------------
# Task transform
# -----------------------
def seq_to_pf_ma_gap(y_seq: torch.Tensor):
    # y_seq: (B,N,H)
    pf = y_seq.sum(dim=-1)
    ma = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap


def mae_rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    diff = (y_pred - y_true).astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse


@torch.no_grad()
def evaluate_seq_model(model, loader, device) -> Dict[str, Tuple[float, float]]:
    model.eval()
    pf_t, pf_p, ma_t, ma_p, gap_t, gap_p = [], [], [], [], [], []

    for x, y, _ in loader:
        x = x.to(device)        # (B,L,N,F)
        y = y.to(device)        # (B,N,H)

        y_pred, _, _ = model(x, None)  # (B,N,H)

        pf_pred, ma_pred, gap_pred = seq_to_pf_ma_gap(y_pred)
        pf_true, ma_true, gap_true = seq_to_pf_ma_gap(y)

        pf_t.append(pf_true.detach().cpu().numpy().reshape(-1))
        pf_p.append(pf_pred.detach().cpu().numpy().reshape(-1))
        ma_t.append(ma_true.detach().cpu().numpy().reshape(-1))
        ma_p.append(ma_pred.detach().cpu().numpy().reshape(-1))
        gap_t.append(gap_true.detach().cpu().numpy().reshape(-1))
        gap_p.append(gap_pred.detach().cpu().numpy().reshape(-1))

    pf_true = np.concatenate(pf_t); pf_pred = np.concatenate(pf_p)
    ma_true = np.concatenate(ma_t); ma_pred = np.concatenate(ma_p)
    gap_true = np.concatenate(gap_t); gap_pred = np.concatenate(gap_p)

    return {
        "PF": mae_rmse_np(pf_true, pf_pred),
        "MA": mae_rmse_np(ma_true, ma_pred),
        "GAP": mae_rmse_np(gap_true, gap_pred),
    }


# -----------------------
# Dataset (same shape as run_baselines panel models)
#   x_seq: (L,N,F_total)
#   y_seq: (N,H)
# -----------------------
class PanelGraphDataset(Dataset):
    def __init__(self, returns_df: pd.DataFrame, macro_df: pd.DataFrame,
                 window_size: int = 20, horizon: int = 5):
        self.window_size = int(window_size)
        self.horizon = int(horizon)

        self.dates = returns_df.index
        self.T, self.N = returns_df.shape

        self.ret = returns_df.values.astype(np.float32)  # (T,N)

        self.macro = None
        self.Fm = 0
        if macro_df is not None and macro_df.shape[1] > 0:
            mac = macro_df.loc[self.dates].values.astype(np.float32)
            mac = pd.DataFrame(mac, index=self.dates).ffill().fillna(0.0).values.astype(np.float32)
            self.macro = mac
            self.Fm = mac.shape[1]

        self.valid_t = list(range(self.window_size - 1, self.T - self.horizon))

    def __len__(self):
        return len(self.valid_t)

    def _rolling_mean(self, x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return x.copy()
        L, N = x.shape
        out = np.empty_like(x, dtype=np.float32)
        csum = np.cumsum(x, axis=0, dtype=np.float64)
        for i in range(L):
            s = max(0, i - w + 1)
            denom = i - s + 1
            if s == 0:
                out[i] = (csum[i] / denom).astype(np.float32)
            else:
                out[i] = ((csum[i] - csum[s - 1]) / denom).astype(np.float32)
        return out

    def _rolling_std(self, x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return np.zeros_like(x, dtype=np.float32)
        L, N = x.shape
        out = np.empty_like(x, dtype=np.float32)
        for i in range(L):
            s = max(0, i - w + 1)
            out[i] = x[s:i+1].std(axis=0).astype(np.float32)
        return out

    def __getitem__(self, idx: int):
        t = self.valid_t[idx]
        L = self.window_size
        H = self.horizon

        s = t - (L - 1)
        e = t + 1
        ret_win = self.ret[s:e, :]  # (L,N)

        feat1 = ret_win
        feat3 = self._rolling_mean(ret_win, 3)
        feat5 = self._rolling_mean(ret_win, 5)
        vol5  = self._rolling_std(ret_win, 5)

        node_feat = np.stack([feat1, feat3, feat5, vol5], axis=-1).astype(np.float32)  # (L,N,4)
        node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # local standardize within window
        L_, N_, F_ = node_feat.shape
        flat = node_feat.reshape(L_ * N_, F_)
        mu = flat.mean(axis=0, keepdims=True)
        sd = flat.std(axis=0, keepdims=True) + 1e-8
        node_feat = ((flat - mu) / sd).reshape(L_, N_, F_).astype(np.float32)

        if self.macro is not None:
            mac = self.macro[s:e, :]             # (L,Fm)
            mac = mac[:, None, :]                # (L,1,Fm)
            mac = np.repeat(mac, self.N, axis=1) # (L,N,Fm)
            x_seq = np.concatenate([node_feat, mac], axis=-1).astype(np.float32)  # (L,N,F_total)
        else:
            x_seq = node_feat

        fut = self.ret[t+1:t+1+H, :]       # (H,N)
        y_seq = fut.T.astype(np.float32)   # (N,H)

        return torch.from_numpy(x_seq), torch.from_numpy(y_seq), t


def make_time_split_from_t(valid_t: List[int], T: int, train_ratio=0.7, val_ratio=0.15):
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    train_ids, val_ids, test_ids = [], [], []
    for idx, t in enumerate(valid_t):
        if t < train_end:
            train_ids.append(idx)
        elif t < val_end:
            val_ids.append(idx)
        else:
            test_ids.append(idx)
    return train_ids, val_ids, test_ids


def train_one_seed(args, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    price_df, returns_df, macro_df, price_cols, _ = load_panel_from_two_files(args.price_path, args.macro_path)

    ds = PanelGraphDataset(returns_df, macro_df, window_size=args.window, horizon=args.horizon)
    train_ids, val_ids, test_ids = make_time_split_from_t(ds.valid_t, T=len(returns_df),
                                                         train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    train_loader = DataLoader(Subset(ds, train_ids), batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader   = DataLoader(Subset(ds, val_ids),   batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(Subset(ds, test_ids),  batch_size=args.batch, shuffle=False, drop_last=False)

    # infer dims
    x0, y0, _ = ds[0]
    L, N, F_total = x0.shape
    assert y0.shape == (N, args.horizon)

    # build A_mech (prior graph) for mech/prior_residual
    A_mech = None
    if args.mode in ("mech", "prior_residual"):
        A_mech = build_adjacency_from_edges(
            edges_path=args.edges_path,
            node_list=list(price_cols),
            weight_col=args.weight_col,
            default_weight=args.default_weight,
            self_loop=True,
            symmetrize=True,
        ).to(device)

    from models.gp_mech_stgnn import GPMechSTGNN
    model = GPMechSTGNN(
        num_nodes=N,
        in_dim=F_total,
        horizon=args.horizon,
        mode=args.mode,
        graph_hidden=args.graph_hidden,
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        gcn_dropout=args.gcn_dropout,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
        graph_topk=args.graph_topk,
    ).to(device)

    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def _loop(loader, train: bool):
        model.train() if train else model.eval()
        total = 0.0
        n = 0
        for x_seq, y_seq, _ in loader:
            x_seq = x_seq.to(device)  # (B,L,N,F)
            y_seq = y_seq.to(device)  # (B,N,H)

            if train:
                opt.zero_grad()
                y_pred, _, _ = model(x_seq, A_mech)
                loss = mse(y_pred, y_seq)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    y_pred, _, _ = model(x_seq, A_mech)
                    loss = mse(y_pred, y_seq)

            total += float(loss.detach()) * x_seq.size(0)
            n += x_seq.size(0)
        return total / max(1, n)

    best_val = 1e18
    best_state = None
    bad = 0
    for ep in range(args.epochs):
        tr = _loop(train_loader, True)
        va = _loop(val_loader, False)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate_seq_model(lambda x, _: model(x, A_mech), val_loader, device)  # wrapper below
    test_metrics = evaluate_seq_model(lambda x, _: model(x, A_mech), test_loader, device)

    return val_metrics, test_metrics


# allow passing a callable into evaluate_seq_model
@torch.no_grad()
def evaluate_seq_model(model_or_callable, loader, device):
    pf_t, pf_p, ma_t, ma_p, gap_t, gap_p = [], [], [], [], [], []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        if callable(model_or_callable):
            y_pred, _, _ = model_or_callable(x, None)
        else:
            y_pred, _, _ = model_or_callable(x, None)

        pf_pred, ma_pred, gap_pred = seq_to_pf_ma_gap(y_pred)
        pf_true, ma_true, gap_true = seq_to_pf_ma_gap(y)

        pf_t.append(pf_true.detach().cpu().numpy().reshape(-1))
        pf_p.append(pf_pred.detach().cpu().numpy().reshape(-1))
        ma_t.append(ma_true.detach().cpu().numpy().reshape(-1))
        ma_p.append(ma_pred.detach().cpu().numpy().reshape(-1))
        gap_t.append(gap_true.detach().cpu().numpy().reshape(-1))
        gap_p.append(gap_pred.detach().cpu().numpy().reshape(-1))

    pf_true = np.concatenate(pf_t); pf_pred = np.concatenate(pf_p)
    ma_true = np.concatenate(ma_t); ma_pred = np.concatenate(ma_p)
    gap_true = np.concatenate(gap_t); gap_pred = np.concatenate(gap_p)

    return {
        "PF": mae_rmse_np(pf_true, pf_pred),
        "MA": mae_rmse_np(ma_true, ma_pred),
        "GAP": mae_rmse_np(gap_true, gap_pred),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, help="mech | learn | prior_residual")

    p.add_argument("--price_path", type=str, required=True)
    p.add_argument("--macro_path", type=str, required=True)
    p.add_argument("--edges_path", type=str, default="", help="required for mech/prior_residual")

    p.add_argument("--weight_col", type=str, default="w")
    p.add_argument("--default_weight", type=float, default=1.0)

    p.add_argument("--window", type=int, default=20)
    p.add_argument("--horizon", type=int, default=5)

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)

    # model params
    p.add_argument("--graph_hidden", type=int, default=32)
    p.add_argument("--gcn_hidden", type=int, default=32)
    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--gcn_dropout", type=float, default=0.0)
    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--rnn_dropout", type=float, default=0.0)
    p.add_argument("--graph_topk", type=int, default=-1)

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out_csv", type=str, default="")

    args = p.parse_args()

    if args.mode in ("mech", "prior_residual") and not args.edges_path:
        raise ValueError("--edges_path is required for mode=mech/prior_residual")

    if args.graph_topk < 0:
        args.graph_topk = None

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for sd in seeds:
        val_metrics, test_metrics = train_one_seed(args, sd)

        for split_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
            for task in ["PF", "MA", "GAP"]:
                mae, rmse = metrics[task]
                rows.append({
                    "model": f"gp_mech_stgnn_{args.mode}",
                    "seed": sd,
                    "split": split_name,
                    "task": task,
                    "mae": mae,
                    "rmse": rmse,
                })

        # print all three tasks for quick check
        print(f"[seed={sd}] "
              f"val PF={val_metrics['PF']} MA={val_metrics['MA']} GAP={val_metrics['GAP']} | "
              f"test PF={test_metrics['PF']} MA={test_metrics['MA']} GAP={test_metrics['GAP']}")

    df = pd.DataFrame(rows)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print("Saved:", args.out_csv)
    else:
        print(df.groupby(["model", "split", "task"])[["mae", "rmse"]].mean())


if __name__ == "__main__":
    main()


