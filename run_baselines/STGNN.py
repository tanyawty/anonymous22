#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STGNN learn-only baseline for the GP multi-task setting.

This baseline corresponds to your original MechAware_GP_STGNN_MultiTask
with mode="learn" (i.e., pure learned adjacency A_learn, no A_mech).

Expected repo layout (from your refactor):
  - data_provider/: load_panel_from_two_files, GPMultiTaskDataset
  - models/: MechAware_GP_STGNN_MultiTask
  - utils/: set_seed, seq_to_pf_ma_gap

Run example:
  python run_baselines/STGNN.py \
    --panel_prices panel_40.csv --panel_macro panel_macro.csv \
    --panel_size 40 --horizon 5 --seed 1

Notes:
  - This script builds a dummy A_mech = I (identity). In learn mode it is not used.
  - Outputs are saved under results/baselines/stgnn_learn/...
"""
import os
import math
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_provider import load_panel_from_two_files, GPMultiTaskDataset
from models import MechAware_GP_STGNN_MultiTask
from utils import set_seed
from utils.metrics import seq_to_pf_ma_gap


def split_by_time(full_ds, T, window_size, horizon, train_ratio, val_ratio):
    """
    Reproduce your original split logic based on valid_idx and time cutoffs.
    """
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    v_idx = np.array(full_ds.valid_idx)
    w = window_size
    h = horizon

    train_mask = (v_idx >= (w - 1)) & (v_idx <= (train_end - h - 1))
    val_mask   = (v_idx >= (train_end - 2)) & (v_idx <= (val_end - h - 1))
    test_mask  = (v_idx >= (val_end - 2)) & (v_idx <= (T - h - 1))

    train_indices = np.nonzero(train_mask)[0].tolist()
    val_indices   = np.nonzero(val_mask)[0].tolist()
    test_indices  = np.nonzero(test_mask)[0].tolist()

    return train_indices, val_indices, test_indices


def mae_rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


@torch.no_grad()
def eval_metrics_seq(model, loader, A_mech, device):
    """
    Flatten across batches and nodes and compute MAE/RMSE for PF/MA/GAP
    derived from the predicted/true future return sequences.
    """
    model.eval()

    pf_true_all, pf_pred_all = [], []
    ma_true_all, ma_pred_all = [], []
    gap_true_all, gap_pred_all = [], []

    for x_seq, y_seq_true in loader:
        x_seq = x_seq.to(device)
        y_seq_true = y_seq_true.to(device)  # (B, N, H)

        y_seq_pred, _, _ = model(x_seq, A_mech)  # (B, N, H)

        pf_p, ma_p, gap_p = seq_to_pf_ma_gap(y_seq_pred)
        pf_t, ma_t, gap_t = seq_to_pf_ma_gap(y_seq_true)

        pf_true_all.append(pf_t.detach().cpu().numpy())
        pf_pred_all.append(pf_p.detach().cpu().numpy())
        ma_true_all.append(ma_t.detach().cpu().numpy())
        ma_pred_all.append(ma_p.detach().cpu().numpy())
        gap_true_all.append(gap_t.detach().cpu().numpy())
        gap_pred_all.append(gap_p.detach().cpu().numpy())

    pf_true = np.concatenate([a.reshape(-1) for a in pf_true_all], axis=0)
    pf_pred = np.concatenate([a.reshape(-1) for a in pf_pred_all], axis=0)
    ma_true = np.concatenate([a.reshape(-1) for a in ma_true_all], axis=0)
    ma_pred = np.concatenate([a.reshape(-1) for a in ma_pred_all], axis=0)
    gap_true = np.concatenate([a.reshape(-1) for a in gap_true_all], axis=0)
    gap_pred = np.concatenate([a.reshape(-1) for a in gap_pred_all], axis=0)

    mae_pf, rmse_pf = mae_rmse_np(pf_true, pf_pred)
    mae_ma, rmse_ma = mae_rmse_np(ma_true, ma_pred)
    mae_gap, rmse_gap = mae_rmse_np(gap_true, gap_pred)

    return {
        "PF":  {"MAE": mae_pf,  "RMSE": rmse_pf},
        "MA":  {"MAE": mae_ma,  "RMSE": rmse_ma},
        "GAP": {"MAE": mae_gap, "RMSE": rmse_gap},
    }


def train_one_epoch(model, loader, A_mech, optimizer, device):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0

    for x_seq, y_seq_true in loader:
        x_seq = x_seq.to(device)
        y_seq_true = y_seq_true.to(device)

        optimizer.zero_grad()
        y_seq_pred, _, _ = model(x_seq, A_mech)
        loss = mse(y_seq_pred, y_seq_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_seq.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, A_mech, device):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    for x_seq, y_seq_true in loader:
        x_seq = x_seq.to(device)
        y_seq_true = y_seq_true.to(device)
        y_seq_pred, _, _ = model(x_seq, A_mech)
        loss = mse(y_seq_pred, y_seq_true)
        total_loss += loss.item() * x_seq.size(0)
    return total_loss / len(loader.dataset)


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- data ----
    price_df, returns_df, macro_df, node_cols, macro_cols = load_panel_from_two_files(
        args.panel_prices, args.panel_macro
    )
    T = len(returns_df)

    full_ds = GPMultiTaskDataset(
        returns_df=returns_df,
        price_df=price_df,
        macro_df=macro_df,
        window_size=args.window_size,
        horizon=args.horizon,
    )

    train_idx, val_idx, test_idx = split_by_time(
        full_ds, T, args.window_size, args.horizon, args.train_ratio, args.val_ratio
    )

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(Subset(full_ds, test_idx),  batch_size=args.batch_size, shuffle=False)

    # ---- dummy A_mech (unused in learn mode) ----
    N = len(node_cols)
    A_mech = torch.eye(N, dtype=torch.float32, device=device)

    # ---- model ----
    sample_x, _ = full_ds[0]
    F_total = sample_x.shape[-1]

    model = MechAware_GP_STGNN_MultiTask(
        num_nodes=N,
        in_dim=F_total,
        mode="learn",                 # <--- baseline: pure learn
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        graph_hidden=args.graph_hidden,
        horizon=args.horizon,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- output dir ----
    out_dir = Path(args.out_dir) / "baselines" / "stgnn_learn" / f"panel{N}_h{args.horizon}" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt_best.pt"

    best_val = math.inf
    best_state = None

    # ---- train ----
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, A_mech, optimizer, device)
        va_loss = eval_loss(model, val_loader, A_mech, device)

        if (epoch == 1) or (epoch % args.log_every == 0):
            print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test metrics ----
    metrics = eval_metrics_seq(model, test_loader, A_mech, device)

    report = {
        "method": "STGNN_learn",
        "panel_size": N,
        "horizon": args.horizon,
        "window_size": args.window_size,
        "seed": args.seed,
        "best_val_loss": best_val,
        "test_metrics": metrics,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[TEST] saved:", str(out_dir / "metrics.json"))
    print("[TEST]", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--panel_prices", type=str, required=True)
    ap.add_argument("--panel_macro",  type=str, required=True)

    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--horizon",     type=int, default=5)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio",   type=float, default=0.15)

    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--epochs",      type=int, default=50)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight_decay",type=float, default=1e-5)

    ap.add_argument("--gcn_hidden",  type=int, default=32)
    ap.add_argument("--gru_hidden",  type=int, default=64)
    ap.add_argument("--graph_hidden",type=int, default=32)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--log_every", type=int, default=10)

    args = ap.parse_args()
    main(args)
