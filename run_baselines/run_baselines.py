# run_baselines/run_baselines.py
# Unified runner for baselines using refactored models/
#
# Supports:
#   - stgnn (learn-only)
#   - fouriergnn (learn-only)
#   - patchtst (HF PatchTST wrapper)
#   - classical: gru/lstm/tcn/transformer/mlp
#
# Metrics: PF/MA/GAP derived from y_seq (B,N,H), report MAE/RMSE over all samples*assets.

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import sys

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
# Load panel + macro
# (same logic as your gp_mech scripts)
# -----------------------
def load_panel_from_two_files(price_path: str, macro_path: str):
    df_p = pd.read_csv(price_path)
    date_col_p = df_p.columns[0]
    df_p[date_col_p] = pd.to_datetime(df_p[date_col_p], dayfirst=True, errors="coerce")
    df_p = df_p.set_index(date_col_p).sort_index()

    # numeric price cols
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

    # log returns
    rets = np.log(price_df).diff().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(axis=0, how="all").fillna(0.0).astype("float32")

    price_df = price_df.loc[rets.index]
    macro_df = macro_df.loc[rets.index]

    return price_df, rets, macro_df, price_cols, macro_cols


# -----------------------
# Task transform (exactly same as your seq_to_pf_ma_gap)
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


# -----------------------
# Dataset A: Panel-graph style (for STGNN/Fourier/Classical)
#   x_seq: (L,N,F_total)
#   y_seq: (N,H)
# Mirrors your gp_mech_multitask_stgnn logic
# -----------------------
class PanelGraphDataset(Dataset):
    def __init__(self, returns_df: pd.DataFrame, price_df: pd.DataFrame, macro_df: pd.DataFrame,
                 window_size: int = 20, horizon: int = 5):
        self.window_size = int(window_size)
        self.horizon = int(horizon)

        self.dates = returns_df.index
        self.T, self.N = returns_df.shape

        self.ret = returns_df.values.astype(np.float32)                  # (T,N)
        self.price = price_df.loc[self.dates].values.astype(np.float32)  # (T,N)

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

        # A compact feature set (same style as your Fourier runner)
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
            mac = self.macro[s:e, :]                  # (L,Fm)
            mac = mac[:, None, :]                     # (L,1,Fm)
            mac = np.repeat(mac, self.N, axis=1)      # (L,N,Fm)
            x_seq = np.concatenate([node_feat, mac], axis=-1).astype(np.float32)  # (L,N,F_total)
        else:
            x_seq = node_feat

        # label: future return seq (H,N) -> (N,H)
        fut = self.ret[t+1:t+1+H, :]                  # (H,N)
        y_seq = fut.T.astype(np.float32)              # (N,H)

        return torch.from_numpy(x_seq), torch.from_numpy(y_seq), t


# -----------------------
# Dataset B: PatchTST style
#   x: (L, N + Fm)
#   y: (H, N)
# -----------------------
class PatchTSTDataset(Dataset):
    def __init__(self, returns_df: pd.DataFrame, macro_df: pd.DataFrame, seq_len: int, pred_len: int):
        self.returns = returns_df.values.astype(np.float32)  # (T,N)
        self.macros = macro_df.values.astype(np.float32) if macro_df is not None else None
        if self.macros is not None:
            self.macros = pd.DataFrame(self.macros, index=returns_df.index).ffill().fillna(0.0).values.astype(np.float32)

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.T, self.N = self.returns.shape
        self.Fm = 0 if self.macros is None else self.macros.shape[1]

        self.max_i = self.T - (self.seq_len + self.pred_len) + 1
        if self.max_i <= 0:
            raise ValueError("Not enough length for given seq_len/pred_len")

    def __len__(self):
        return self.max_i

    def __getitem__(self, i: int):
        x_ret = self.returns[i:i+self.seq_len]  # (L,N)
        if self.macros is not None:
            x_mac = self.macros[i:i+self.seq_len]  # (L,Fm)
            x = np.concatenate([x_ret, x_mac], axis=1).astype(np.float32)  # (L,N+Fm)
        else:
            x = x_ret.astype(np.float32)

        y = self.returns[i+self.seq_len:i+self.seq_len+self.pred_len].astype(np.float32)  # (H,N)
        return torch.from_numpy(x), torch.from_numpy(y), i


# -----------------------
# Time split helpers (by time index, no leakage)
# -----------------------
@dataclass
class Split:
    train_ids: List[int]
    val_ids: List[int]
    test_ids: List[int]


def make_time_split_from_t(valid_t: List[int], T: int, train_ratio=0.7, val_ratio=0.15) -> Split:
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


def make_time_split_patch(T: int, seq_len: int, pred_len: int, train_ratio=0.7, val_ratio=0.15) -> Split:
    # indices i correspond to window starting at i
    max_i = T - (seq_len + pred_len) + 1
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # map i -> last input time index = i+seq_len-1
    train_ids, val_ids, test_ids = [], [], []
    for i in range(max_i):
        t_last = i + seq_len - 1
        if t_last < train_end:
            train_ids.append(i)
        elif t_last < val_end:
            val_ids.append(i)
        else:
            test_ids.append(i)
    return Split(train_ids, val_ids, test_ids)


# -----------------------
# Train/Eval
# -----------------------
@torch.no_grad()
def evaluate_seq_model(model, loader, device) -> Dict[str, Tuple[float, float]]:
    model.eval()
    pf_t, pf_p, ma_t, ma_p, gap_t, gap_p = [], [], [], [], [], []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)  # (B,N,H) for panel models OR (B,H,N) for patchtst dataset (handled upstream)

        y_pred, _, _ = model(x, None)  # must return (B,N,H)

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

    price_df, returns_df, macro_df, price_cols, macro_cols = load_panel_from_two_files(args.price_path, args.macro_path)

    # ---- build dataset/loader by model type ----
    if args.model == "patchtst":
        ds = PatchTSTDataset(returns_df, macro_df, seq_len=args.window, pred_len=args.horizon)
        split = make_time_split_patch(T=len(returns_df), seq_len=args.window, pred_len=args.horizon,
                                      train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        train_ds = Subset(ds, split.train_ids)
        val_ds   = Subset(ds, split.val_ids)
        test_ds  = Subset(ds, split.test_ids)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

        # infer dims
        N = returns_df.shape[1]
        Fm = macro_df.shape[1] if macro_df is not None else 0

        from models.patchtst import PatchTST_Baseline
        model = PatchTST_Baseline(num_assets=N, num_macros=Fm, seq_len=args.window, horizon=args.horizon,
                                  d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                                  dropout=args.dropout).to(device)

        # PatchTSTDataset yields y: (B,H,N); wrapper outputs y_seq: (B,N,H)
        # So we convert y in the loop.
        mse = nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        def _loop(loader, train: bool):
            if train:
                model.train()
            else:
                model.eval()

            total = 0.0
            n = 0
            for x, y, _ in loader:
                x = x.to(device)                 # (B,L,N+Fm)
                y = y.to(device)                 # (B,H,N)
                y_seq = y.permute(0, 2, 1)        # (B,N,H)

                if train:
                    opt.zero_grad()
                    y_pred, _, _ = model(x, None)   # (B,N,H)
                    loss = mse(y_pred, y_seq)
                    loss.backward()
                    opt.step()
                else:
                    with torch.no_grad():
                        y_pred, _, _ = model(x, None)
                        loss = mse(y_pred, y_seq)

                total += float(loss) * x.size(0)
                n += x.size(0)
            return total / max(1, n)

        best_val = 1e18
        best_state = None
        patience = args.patience
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
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # For evaluation, build a loader that yields y_seq (B,N,H)
        def wrap_loader(loader):
            for x, y, idx in loader:
                yield x, y.permute(0, 2, 1), idx

        val_metrics = evaluate_seq_model(model, wrap_loader(val_loader), device)
        test_metrics = evaluate_seq_model(model, wrap_loader(test_loader), device)

        return {"val": val_metrics, "test": test_metrics}

    else:
        # Panel models (STGNN/Fourier/Classical): x_seq (B,L,N,F) ; y_seq (B,N,H)
        ds = PanelGraphDataset(returns_df, price_df, macro_df, window_size=args.window, horizon=args.horizon)
        split = make_time_split_from_t(ds.valid_t, T=len(returns_df), train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        train_ds = Subset(ds, split.train_ids)
        val_ds   = Subset(ds, split.val_ids)
        test_ds  = Subset(ds, split.test_ids)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

        # infer dims
        x0, y0, _ = ds[0]
        L, N, F_total = x0.shape
        assert L == args.window
        assert y0.shape == (N, args.horizon)

        # model zoo
        if args.model == "stgnn":
            from models.stgnn import STGNN
            model = STGNN(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                          gcn_hidden=args.gcn_hidden, gru_hidden=args.gru_hidden,
                          graph_hidden=args.graph_hidden, dropout=args.dropout).to(device)

        elif args.model == "fouriergnn":
            from models.fouriergnn import FourierGNN_LearnOnly
            model = FourierGNN_LearnOnly(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                         d_model=args.d_model, n_layers=args.n_layers,
                                         graph_hidden=args.graph_hidden, dropout=args.dropout).to(device)

        elif args.model in {"gru", "lstm", "tcn", "transformer", "mlp"}:
            from models.classical_baselines import (
                GRU_Baseline, LSTM_Baseline, TCN_Baseline, Transformer_Baseline, MLP_Baseline
            )
            if args.model == "gru":
                model = GRU_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                     hidden_dim=args.gru_hidden, use_graph_logging=True,
                                     graph_hidden=args.graph_hidden).to(device)
            elif args.model == "lstm":
                model = LSTM_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                      hidden_dim=args.gru_hidden, use_graph_logging=True,
                                      graph_hidden=args.graph_hidden).to(device)
            elif args.model == "tcn":
                model = TCN_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                     channel=args.gcn_hidden, use_graph_logging=True,
                                     graph_hidden=args.graph_hidden).to(device)
            elif args.model == "transformer":
                model = Transformer_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                             d_model=args.d_model, nhead=args.n_heads, num_layers=args.n_layers,
                                             dropout=args.dropout, use_graph_logging=True,
                                             graph_hidden=args.graph_hidden).to(device)
            else:
                model = MLP_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon, window_len=args.window,
                                     hidden_dim=args.d_model, dropout=args.dropout, use_graph_logging=True,
                                     graph_hidden=args.graph_hidden).to(device)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        mse = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        def _loop(loader, train: bool):
            if train:
                model.train()
            else:
                model.eval()

            total = 0.0
            n = 0
            for x_seq, y_seq, _ in loader:
                x_seq = x_seq.to(device)                      # (B,L,N,F)
                y_seq = y_seq.to(device)                      # (B,N,H)

                if train:
                    opt.zero_grad()
                    y_pred, _, _ = model(x_seq, None)
                    loss = mse(y_pred, y_seq)
                    loss.backward()
                    opt.step()
                else:
                    with torch.no_grad():
                        y_pred, _, _ = model(x_seq, None)
                        loss = mse(y_pred, y_seq)

                total += float(loss) * x_seq.size(0)
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

        val_metrics = evaluate_seq_model(model, val_loader, device)
        test_metrics = evaluate_seq_model(model, test_loader, device)

        return {"val": val_metrics, "test": test_metrics}


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True,
                   help="stgnn | fouriergnn | patchtst | gru | lstm | tcn | transformer | mlp")
    p.add_argument("--price_path", type=str, required=True)
    p.add_argument("--macro_path", type=str, required=True)

    p.add_argument("--window", type=int, default=20)
    p.add_argument("--horizon", type=int, default=5)

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)

    # shared dims
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=8)

    # stgnn dims (also reused as classical defaults)
    p.add_argument("--gcn_hidden", type=int, default=32)
    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--graph_hidden", type=int, default=32)

    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out_csv", type=str, default="")

    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for sd in seeds:
        res = train_one_seed(args, sd)
        for split in ["val", "test"]:
            for task in ["PF", "MA", "GAP"]:
                mae, rmse = res[split][task]
                rows.append({
                    "model": args.model,
                    "seed": sd,
                    "split": split,
                    "task": task,
                    "mae": mae,
                    "rmse": rmse,
                })

        print(f"[seed={sd}] val PF(MAE/RMSE)={res['val']['PF']} | test PF(MAE/RMSE)={res['test']['PF']}")

    df = pd.DataFrame(rows)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print("Saved:", args.out_csv)
    else:
        print(df.groupby(["model", "split", "task"])[["mae", "rmse"]].mean())

if __name__ == "__main__":
    main()



