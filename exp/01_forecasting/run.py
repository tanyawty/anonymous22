# run_baselines/run_gp_mech_stgnn.py
# Runner for MAIN model: models/gp_mech_stgnn.py::GPMechSTGNN
# Keep logic close to original gp_mech_multitask_stgnn.py, but use refactored model.

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import sys

# make "models/" importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
# Load panel + macro (same spirit as your original scripts)
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
        .ffill().bfill()
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
        .ffill().bfill()
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
# Build A_mech from edges csv (source,target,w)
# -----------------------
def build_adjacency_from_edges(
    edges_path: str,
    node_list: List[str],
    weight_col: str = "w",
    default_weight: float = 1.0,
    self_loop: bool = True,
    symmetrize: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    edges = pd.read_csv(edges_path)
    nodes_idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)
    A = np.zeros((N, N), dtype=np.float32)

    # robust column names
    src_col = "source" if "source" in edges.columns else ("src" if "src" in edges.columns else None)
    tgt_col = "target" if "target" in edges.columns else ("dst" if "dst" in edges.columns else None)
    if src_col is None or tgt_col is None:
        raise ValueError(f"Edges file must contain source/target (or src/dst). Got cols={edges.columns.tolist()}")

    for _, r in edges.iterrows():
        src, tgt = r[src_col], r[tgt_col]
        if src in nodes_idx and tgt in nodes_idx:
            i, j = nodes_idx[src], nodes_idx[tgt]
            w = r.get(weight_col, np.nan)
            if pd.isna(w):
                w = default_weight
            A[i, j] += float(w)

    if self_loop:
        A += np.eye(N, dtype=np.float32)

    if symmetrize:
        A = 0.5 * (A + A.T)

    if normalize:
        deg = A.sum(axis=1)
        deg[deg == 0] = 1.0
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        A = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.tensor(A, dtype=torch.float32)


# -----------------------
# Task transform (same as your seq_to_pf_ma_gap)
# -----------------------
def seq_to_pf_ma_gap(y_seq: torch.Tensor):
    pf = y_seq.sum(dim=-1)
    ma = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap


def mae_rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    diff = (y_pred - y_true).astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse


# -----------------------
# Dataset (close to your original style)
# x_seq: (L,N,F_total), y_seq: (N,H)
# -----------------------
class GPMultiTaskDataset(Dataset):
    def __init__(self, returns_df: pd.DataFrame, macro_df: pd.DataFrame, window_size: int, horizon: int):
        self.window_size = int(window_size)
        self.horizon = int(horizon)

        self.ret = returns_df.values.astype(np.float32)  # (T,N)
        self.dates = returns_df.index
        self.T, self.N = self.ret.shape

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

        # compact node features: [ret, ma3, ma5, vol5]
        feat1 = ret_win
        feat3 = self._rolling_mean(ret_win, 3)
        feat5 = self._rolling_mean(ret_win, 5)
        vol5  = self._rolling_std(ret_win, 5)

        node_feat = np.stack([feat1, feat3, feat5, vol5], axis=-1).astype(np.float32)  # (L,N,4)
        node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # local standardize within window (match your baseline runner)
        L_, N_, F_ = node_feat.shape
        flat = node_feat.reshape(L_ * N_, F_)
        mu = flat.mean(axis=0, keepdims=True)
        sd = flat.std(axis=0, keepdims=True) + 1e-8
        node_feat = ((flat - mu) / sd).reshape(L_, N_, F_).astype(np.float32)

        if self.macro is not None:
            mac = self.macro[s:e, :]                  # (L,Fm)
            mac = mac[:, None, :]                     # (L,1,Fm)
            mac = np.repeat(mac, self.N, axis=1)      # (L,N,Fm)
            x_seq = np.concatenate([node_feat, mac], axis=-1).astype(np.float32)  # (L,N,F_total)
        else:
            x_seq = node_feat

        # future return sequence: (H,N) -> (N,H)
        fut = self.ret[t+1:t+1+H, :]                  # (H,N)
        y_seq = fut.T.astype(np.float32)              # (N,H)

        return torch.from_numpy(x_seq), torch.from_numpy(y_seq)


# -----------------------
# Time split (no leakage)
# -----------------------
@dataclass
class Split:
    train_ids: List[int]
    val_ids: List[int]
    test_ids: List[int]


def make_time_split(valid_t: List[int], T: int, train_ratio=0.7, val_ratio=0.15) -> Split:
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
    return Split(train_ids, val_ids, test_ids)


# -----------------------
# Eval metrics on PF/MA/GAP
# -----------------------
@torch.no_grad()
def evaluate(model, loader, A_mech, device) -> Dict[str, Tuple[float, float]]:
    model.eval()
    pf_t, pf_p, ma_t, ma_p, gap_t, gap_p = [], [], [], [], [], []

    for x, y in loader:
        x = x.to(device)                        # (B,L,N,F)
        y = y.to(device)                        # (B,N,H)
        y_pred, _, _ = model(x, A_mech)          # (B,N,H)

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


def train_one_seed(args, seed: int) -> Dict[str, Dict[str, Tuple[float, float]]]:
    set_seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    price_df, returns_df, macro_df, price_cols, macro_cols = load_panel_from_two_files(args.panel_prices, args.panel_macro)

    # build A_mech
    A_mech = build_adjacency_from_edges(
        edges_path=args.edges,
        node_list=price_cols,
        weight_col=args.weight_col,
        default_weight=args.default_weight,
        self_loop=True,
        symmetrize=True,
        normalize=True,
    ).to(device)

    ds = GPMultiTaskDataset(returns_df, macro_df, window_size=args.window, horizon=args.horizon)
    split = make_time_split(ds.valid_t, T=len(returns_df), train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    train_ds = Subset(ds, split.train_ids)
    val_ds   = Subset(ds, split.val_ids)
    test_ds  = Subset(ds, split.test_ids)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    # infer dims
    x0, y0 = ds[0]
    L, N, F_total = x0.shape
    assert L == args.window
    assert y0.shape == (N, args.horizon)

    # import MAIN model from your repo
    from models.gp_mech_stgnn import GPMechSTGNN

    model = GPMechSTGNN(
        num_nodes=N,
        in_dim=F_total,
        horizon=args.horizon,
        mode=args.mode,                 # learn | mech | prior_residual
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
        for x, y in loader:
            x = x.to(device)  # (B,L,N,F)
            y = y.to(device)  # (B,N,H)

            if train:
                opt.zero_grad()
                y_pred, _, _ = model(x, A_mech)
                loss = mse(y_pred, y)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    y_pred, _, _ = model(x, A_mech)
                    loss = mse(y_pred, y)

            total += float(loss.detach()) * x.size(0)
            n += x.size(0)
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

    val_metrics = evaluate(model, val_loader, A_mech, device)
    test_metrics = evaluate(model, test_loader, A_mech, device)

    # optional: save ckpt per seed
    if args.save_ckpt:
        os.makedirs(args.save_ckpt, exist_ok=True)
        ckpt_path = os.path.join(args.save_ckpt, f"ckpt_mode={args.mode}_h{args.horizon}_seed{seed}.pt")
        torch.save({"state_dict": model.state_dict(), "seed": seed, "mode": args.mode, "horizon": args.horizon}, ckpt_path)

    return {"val": val_metrics, "test": test_metrics}


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--panel_prices", type=str, required=True)
    p.add_argument("--panel_macro", type=str, required=True)
    p.add_argument("--edges", type=str, required=True)

    p.add_argument("--mode", type=str, default="prior_residual", help="learn | mech | prior_residual")

    p.add_argument("--window", type=int, default=20)
    p.add_argument("--horizon", type=int, default=5)

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)

    # mech graph weights
    p.add_argument("--weight_col", type=str, default="w")
    p.add_argument("--default_weight", type=float, default=1.0)

    # model dims
    p.add_argument("--graph_hidden", type=int, default=32)
    p.add_argument("--gcn_hidden", type=int, default=32)
    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--gcn_dropout", type=float, default=0.0)
    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--rnn_dropout", type=float, default=0.0)
    p.add_argument("--graph_topk", type=int, default=0, help="0 means None")

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--save_ckpt", type=str, default="", help="dir to save ckpt per seed (optional)")

    args = p.parse_args()

    if args.graph_topk == 0:
        args.graph_topk = None

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for sd in seeds:
        res = train_one_seed(args, sd)

        # print all tasks (not only PF)
        print(f"[seed={sd}] "
              f"val PF={res['val']['PF']} MA={res['val']['MA']} GAP={res['val']['GAP']} | "
              f"test PF={res['test']['PF']} MA={res['test']['MA']} GAP={res['test']['GAP']}")

        for split in ["val", "test"]:
            for task in ["PF", "MA", "GAP"]:
                mae, rmse = res[split][task]
                rows.append({
                    "mode": args.mode,
                    "seed": sd,
                    "split": split,
                    "task": task,
                    "mae": mae,
                    "rmse": rmse,
                })

    df = pd.DataFrame(rows)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print("Saved:", args.out_csv)
    else:
        print(df.groupby(["mode", "split", "task"])[["mae", "rmse"]].mean())


if __name__ == "__main__":
    main()

