# run_compare_sharpe_gpmech.py
# Compare Sharpe ratio:
#   - Your main model: GPMechSTGNN (mode: mech/learn/prior_residual) from run_gp_mech.py
#   - Baselines: stgnn/fouriergnn/patchtst/gru/lstm/tcn/transformer/mlp from run_baselines.py
#   - Quant baselines: EW / CSM
#
# This script DOES NOT modify your original scripts.

from __future__ import annotations

import argparse
import importlib.util
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# -------------------------
# Dynamic import helper
# -------------------------
def import_from_path(module_name: str, py_path: str):
    py_path = os.path.abspath(py_path)
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_name} from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------------------------
# Backtest helpers
# -------------------------
def sharpe_ratio(returns: np.ndarray, ann: int = 252, eps: float = 1e-12) -> float:
    r = np.asarray(returns).reshape(-1)
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd < eps:
        return 0.0
    return float(mu / sd * np.sqrt(ann))

def _ls_ret(scores: np.ndarray, realized: np.ndarray, top_frac: float) -> float:
    N = scores.shape[0]
    K = max(1, int(round(N * top_frac)))
    idx = np.argsort(scores)  # ascending
    short_idx = idx[:K]
    long_idx = idx[-K:]
    return float(realized[long_idx].mean() - realized[short_idx].mean())

def backtest_long_short(score_mat: np.ndarray, y_true_seq: np.ndarray,
                        top_frac: float, holding: str, step: int, ann: int) -> Dict[str, float]:
    if holding == "1":
        realized = y_true_seq[:, :, 0]
    else:
        realized = y_true_seq.sum(axis=-1)

    rets = []
    for i in range(0, score_mat.shape[0], step):
        rets.append(_ls_ret(score_mat[i], realized[i], top_frac))
    rets = np.asarray(rets, dtype=float)

    return {
        "mean": float(rets.mean()),
        "std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "sharpe": sharpe_ratio(rets, ann=ann),
        "n": int(len(rets)),
    }

# ----- Portfolio / Overlay / Pairs backtests -----
def _long_only_ret(scores: np.ndarray, realized: np.ndarray, top_frac: float) -> float:
    """Long-only top-K, rest in cash."""
    N = scores.shape[0]
    K = max(1, int(round(N * top_frac)))
    idx = np.argsort(scores)
    long_idx = idx[-K:]
    return float(realized[long_idx].mean())

def _long_only_active_ret(scores: np.ndarray, realized: np.ndarray, top_frac: float) -> float:
    """Long-only top-K active return over EW benchmark."""
    N = scores.shape[0]
    K = max(1, int(round(N * top_frac)))
    idx = np.argsort(scores)
    long_idx = idx[-K:]
    long_ret = realized[long_idx].mean()
    ew = realized.mean()
    return float(long_ret - ew)

def backtest_selection(score_mat: np.ndarray, y_true_seq: np.ndarray,
                       top_frac: float, holding: str, step: int, ann: int,
                       portfolio: str) -> Dict[str, float]:
    """Selection-based backtest (longshort / longonly / longonly_active)."""
    if holding == "1":
        realized = y_true_seq[:, :, 0]
    else:
        realized = y_true_seq.sum(axis=-1)

    rets = []
    for i in range(0, score_mat.shape[0], step):
        if portfolio == "longshort":
            r = _ls_ret(score_mat[i], realized[i], top_frac)
        elif portfolio == "longonly":
            r = _long_only_ret(score_mat[i], realized[i], top_frac)
        elif portfolio == "longonly_active":
            r = _long_only_active_ret(score_mat[i], realized[i], top_frac)
        else:
            raise ValueError(f"Unknown portfolio={portfolio}")
        rets.append(r)

    rets = np.asarray(rets, dtype=float)
    return {
        "mean": float(rets.mean()),
        "std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "sharpe": sharpe_ratio(rets, ann=ann),
        "n": int(len(rets)),
    }

def backtest_overlay(
    score_mat: np.ndarray,
    y_true_seq: np.ndarray,
    holding: str,
    step: int,
    ann: int,
    wmax: float = 1.0,
    clip_z: float = 2.0,
    ema_alpha: float = 0.0,
    thr: float = 0.0,
    tcost: float = 0.0,
) -> Dict[str, float]:
    """
    Risk overlay on EW (directional / allocation-style):
      - market signal m_t = mean_i(score_i(t))
      - z-score over time on test window
      - optional EMA smoothing on z (ema_alpha in [0,1], 0 disables)
      - optional thresholding: if |z| < thr => w_t = 0
      - position w_t = clip(z_t, -clip_z, clip_z) / clip_z * wmax
      - portfolio return r_t = w_t * r_EW(t) - tcost * |w_t - w_{t-1}|
    Notes:
      - holding/step control how often we sample returns (same convention as other backtests).
      - tcost is per-unit turnover cost in return units (e.g., 0.0001 = 1bp per full weight change).
    """
    if holding == "1":
        realized = y_true_seq[:, :, 0]  # (T,N)
    else:
        realized = y_true_seq.sum(axis=-1)  # (T,N)

    # EW realized return each step
    r_ew = realized.mean(axis=1)  # (T,)

    # market-level predicted signal (use score spread-robust mean)
    m = score_mat.mean(axis=1)  # (T,)
    m_mu = float(m.mean())
    m_sd = float(m.std(ddof=1)) if len(m) > 1 else 0.0
    z = (m - m_mu) / (m_sd + 1e-12)

    # EMA smoothing on z
    if ema_alpha and ema_alpha > 0.0:
        ema_alpha = float(ema_alpha)
        z_s = np.empty_like(z)
        z_s[0] = z[0]
        for i in range(1, len(z)):
            z_s[i] = ema_alpha * z[i] + (1.0 - ema_alpha) * z_s[i - 1]
        z = z_s

    # thresholding (no-trade zone)
    if thr and thr > 0.0:
        thr = float(thr)
        z = np.where(np.abs(z) < thr, 0.0, z)

    # map to weight
    if clip_z <= 0:
        raise ValueError("clip_z must be > 0")
    z_clip = np.clip(z, -clip_z, clip_z)
    w = (z_clip / clip_z) * float(wmax)  # (T,)

    rets = []
    turnovers = []
    prev_w = 0.0
    for i in range(0, len(w), step):
        wi = float(w[i])
        ri = float(r_ew[i])
        turn = abs(wi - prev_w)
        cost = float(tcost) * turn
        rets.append(wi * ri - cost)
        turnovers.append(turn)
        prev_w = wi

    rets = np.asarray(rets, dtype=float)
    turnovers = np.asarray(turnovers, dtype=float)

    return {
        "mean": float(rets.mean()),
        "std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "sharpe": sharpe_ratio(rets, ann=ann),
        "n": int(len(rets)),
        "turnover": float(turnovers.mean()) if len(turnovers) else 0.0,
    }


def _read_pairs_with_weights(edges_path: str, node_list: List[str], num_pairs: int = 50) -> List[Tuple[int, int, float]]:
    """Read top weighted pairs from edges csv.
    Supports flexible column names. Returns list of (i_idx, j_idx, weight).
    """
    import csv

    node_to_idx = {str(n): k for k, n in enumerate(node_list)}
    rows = []
    with open(edges_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"edges file has no header: {edges_path}")
        fields = [c.strip() for c in reader.fieldnames]

        def pick_col(cands: List[str], default: Optional[str] = None) -> Optional[str]:
            for c in cands:
                if c in fields:
                    return c
            return default

        col_u = pick_col(["u","src","source","from","i","node_i","asset_i"])
        col_v = pick_col(["v","dst","dest","to","j","node_j","asset_j"])
        col_w = pick_col(["w","weight","value","corr","score"], default=None)

        # fallback: first two cols
        if col_u is None or col_v is None:
            if len(fields) < 2:
                raise ValueError(f"edges file must have >=2 columns: {edges_path}")
            col_u, col_v = fields[0], fields[1]
            if col_w is None and len(fields) >= 3:
                col_w = fields[2]

        for r in reader:
            u = r.get(col_u, "")
            v = r.get(col_v, "")
            if u is None or v is None:
                continue
            u = str(u).strip()
            v = str(v).strip()
            if u == "" or v == "":
                continue
            if u == v:
                continue

            # map to indices (allow ints as strings)
            if u in node_to_idx:
                ui = node_to_idx[u]
            else:
                try:
                    ui = int(float(u))
                except Exception:
                    continue

            if v in node_to_idx:
                vi = node_to_idx[v]
            else:
                try:
                    vi = int(float(v))
                except Exception:
                    continue

            w = 1.0
            if col_w is not None:
                try:
                    w = float(r.get(col_w, 1.0))
                except Exception:
                    w = 1.0
            rows.append((ui, vi, w))

    # sort by abs weight desc
    rows.sort(key=lambda x: abs(x[2]), reverse=True)

    # deduplicate undirected pairs (i<j)
    seen = set()
    pairs = []
    for i, j, w in rows:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b, float(w)))
        if len(pairs) >= num_pairs:
            break

    if not pairs:
        raise ValueError(f"No valid pairs found in {edges_path}")
    return pairs


def backtest_pairs(score_mat: np.ndarray, y_true_seq: np.ndarray,
                   edges_path: str, node_list: List[str],
                   holding: str, step: int, ann: int,
                   num_pairs: int = 50,
                   pairs_ema: float = 0.0,
                   pairs_thr: float = 0.0,
                   pairs_pos_clip: float = 1.0,
                   pairs_tcost: float = 0.0,
                   pairs_weighted: bool = False) -> Dict[str, float]:
    """
    Mechanism-pairs relative value backtest (designed for mechanism-anchored models):
      - Pair universe: top-|w| edges from edges_path (mechanism graph).
      - Signal per pair: s_ij(t) = score_i(t) - score_j(t) where score_i is your PF proxy (sum of predicted seq returns).
      - Position per pair (discrete by default):
            raw = s_ij(t); optionally EMA-smoothed.
            if |raw| < pairs_thr -> pos = 0
            else pos = sign(raw) * pairs_pos_clip (clipped to [0, pairs_pos_clip])
      - Return per pair: pos * (r_i(t) - r_j(t)).
      - Portfolio: equal-weight average across pairs, or weighted by |w| if pairs_weighted.
      - Transaction cost: pairs_tcost * turnover, where turnover is average |Δpos| across pairs.
    """
    if holding == "1":
        realized = y_true_seq[:, :, 0]  # (T,N)
    else:
        realized = y_true_seq.sum(axis=-1)

    pairs = _read_pairs_with_weights(edges_path, node_list=node_list, num_pairs=num_pairs)
    P = len(pairs)
    weights = np.ones(P, dtype=float)
    if pairs_weighted:
        weights = np.array([abs(w) for (_i, _j, w) in pairs], dtype=float)
        weights = weights / (weights.sum() + 1e-12)

    # EMA state on raw signals per pair
    ema = float(pairs_ema)
    use_ema = ema > 0.0 and ema < 1.0
    sig_state = np.zeros(P, dtype=float)

    # positions
    pos_prev = np.zeros(P, dtype=float)
    rets = []
    turnovers = []

    for t in range(0, score_mat.shape[0], step):
        s = score_mat[t]  # (N,)
        r = realized[t]   # (N,)

        # compute raw signals
        raw = np.empty(P, dtype=float)
        spread = np.empty(P, dtype=float)
        for k, (i, j, _w) in enumerate(pairs):
            raw[k] = float(s[i] - s[j])
            spread[k] = float(r[i] - r[j])

        if use_ema:
            sig_state = ema * sig_state + (1.0 - ema) * raw
            sig = sig_state
        else:
            sig = raw

        # positions (discrete, clipped)
        pos = np.zeros(P, dtype=float)
        if pairs_thr > 0.0:
            active = np.abs(sig) >= float(pairs_thr)
        else:
            active = np.ones(P, dtype=bool)

        pos_mag = float(max(0.0, pairs_pos_clip))
        # sign position only where active and sig != 0
        sgn = np.sign(sig)
        pos[active] = sgn[active] * pos_mag

        # turnover + cost
        turnover = float(np.mean(np.abs(pos - pos_prev)))
        cost = float(pairs_tcost) * turnover
        turnovers.append(turnover)

        port_ret = float(np.sum(weights * (pos * spread)))
        rets.append(port_ret - cost)

        pos_prev = pos

    rets = np.asarray(rets, dtype=float)
    turnovers = np.asarray(turnovers, dtype=float)
    return {
        "mean": float(rets.mean()),
        "std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "sharpe": sharpe_ratio(rets, ann=ann),
        "n": int(len(rets)),
        "turnover": float(turnovers.mean()) if len(turnovers) else 0.0,
    }

def backtest_dispatch(score_mat: np.ndarray, y_true_seq: np.ndarray, args,
                      node_list: List[str], step: int) -> Dict[str, float]:
    """Route to the chosen backtest."""
    if args.strategy == "selection":
        return backtest_selection(score_mat, y_true_seq,
                                  top_frac=args.top_frac, holding=args.holding,
                                  step=step, ann=args.ann,
                                  portfolio=args.portfolio)
    if args.strategy == "overlay":
        return backtest_overlay(score_mat, y_true_seq,
                                holding=args.holding, step=step, ann=args.ann,
                                wmax=args.overlay_wmax, clip_z=args.overlay_clipz,
                                ema_alpha=args.overlay_ema, thr=args.overlay_thr,
                                tcost=args.overlay_tcost)
    if args.strategy == "pairs":
        if not args.edges_path:
            raise ValueError("--edges_path required for strategy=pairs")
        return backtest_pairs(score_mat, y_true_seq,
                         edges_path=args.edges_path, node_list=node_list,
                         holding=args.holding, step=step, ann=args.ann,
                         num_pairs=args.num_pairs,
                         pairs_ema=args.pairs_ema, pairs_thr=args.pairs_thr,
                         pairs_pos_clip=args.pairs_pos_clip,
                         pairs_tcost=args.pairs_tcost,
                         pairs_weighted=args.pairs_weighted)
    raise ValueError(f"Unknown strategy={args.strategy}")


def backtest_equal_weight(y_true_seq: np.ndarray, holding: str, step: int, ann: int) -> Dict[str, float]:
    if holding == "1":
        realized = y_true_seq[:, :, 0]
    else:
        realized = y_true_seq.sum(axis=-1)

    rets = []
    for i in range(0, realized.shape[0], step):
        rets.append(float(realized[i].mean()))
    rets = np.asarray(rets, dtype=float)

    return {
        "mean": float(rets.mean()),
        "std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "sharpe": sharpe_ratio(rets, ann=ann),
        "n": int(len(rets)),
    }

def backtest_csm(returns_array: np.ndarray, t_idx_test: np.ndarray, y_true_seq: np.ndarray,
                 lookback: int, top_frac: float, holding: str, step: int, ann: int) -> Dict[str, float]:
    if holding == "1":
        realized = y_true_seq[:, :, 0]
    else:
        realized = y_true_seq.sum(axis=-1)

    rets = []
    for j in range(0, y_true_seq.shape[0], step):
        t = int(t_idx_test[j])
        start = max(0, t - lookback + 1)
        sig = returns_array[start:t + 1].sum(axis=0)  # (N,)
        rets.append(_ls_ret(sig, realized[j], top_frac))
    rets = np.asarray(rets, dtype=float)

    return {
        "mean": float(rets.mean()),
        "std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "sharpe": sharpe_ratio(rets, ann=ann),
        "n": int(len(rets)),
    }


# -------------------------
# Collect scores
# -------------------------
@torch.no_grad()
def collect_scores_ytrue_t_panel(model: nn.Module, loader: DataLoader, device: torch.device,
                                 A: Optional[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Panel loader yields: x:(B,L,N,F), y:(B,N,H), t:(B,)
    Model forward: y_pred, _, _ = model(x, A_or_None)
    """
    model.eval()
    scores, ytrues, ts = [], [], []
    for x, y, t in loader:
        x = x.to(device)
        y = y.to(device)
        y_pred, _, _ = model(x, A)
        score = y_pred.sum(dim=-1)  # PF score (B,N)
        scores.append(score.detach().cpu().numpy())
        ytrues.append(y.detach().cpu().numpy())
        ts.append(np.asarray(t))
    return (
        np.concatenate(scores, axis=0),
        np.concatenate(ytrues, axis=0),
        np.concatenate(ts, axis=0).astype(int)
    )

@torch.no_grad()
def collect_scores_ytrue_t_patchtst(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PatchTST loader yields: x:(B,L,N+Fm), y:(B,H,N), t:(B,)
    Convert y -> (B,N,H) for backtest.
    """
    model.eval()
    scores, ytrues, ts = [], [], []
    for x, y, t in loader:
        x = x.to(device)
        y = y.to(device)
        y_seq = y.permute(0, 2, 1)          # (B,N,H)
        y_pred, _, _ = model(x, None)       # (B,N,H)
        score = y_pred.sum(dim=-1)
        scores.append(score.detach().cpu().numpy())
        ytrues.append(y_seq.detach().cpu().numpy())
        ts.append(np.asarray(t))
    return (
        np.concatenate(scores, axis=0),
        np.concatenate(ytrues, axis=0),
        np.concatenate(ts, axis=0).astype(int)
    )


# -------------------------
# Train helper (same early-stop style as your runners)
# -------------------------
def train_earlystop_panel(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                          device: torch.device, A: Optional[torch.Tensor],
                          epochs: int, patience: int, lr: float, wd: float) -> nn.Module:
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = float("inf")
    best_state = None
    bad = 0

    for _ep in range(epochs):
        model.train()
        for x, y, _t in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            y_pred, _, _ = model(x, A)
            loss = mse(y_pred, y)
            loss.backward()
            opt.step()

        # val
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x, y, _t in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred, _, _ = model(x, A)
                loss = mse(y_pred, y)
                total += float(loss) * x.size(0)
                n += x.size(0)
        v = total / max(1, n)

        if v < best_val:
            best_val = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_earlystop_patchtst(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                            device: torch.device, epochs: int, patience: int, lr: float, wd: float) -> nn.Module:
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val = float("inf")
    best_state = None
    bad = 0

    for _ep in range(epochs):
        model.train()
        for x, y, _t in train_loader:
            x = x.to(device)
            y = y.to(device)                # (B,H,N)
            y_seq = y.permute(0, 2, 1)       # (B,N,H)
            opt.zero_grad()
            y_pred, _, _ = model(x, None)
            loss = mse(y_pred, y_seq)
            loss.backward()
            opt.step()

        # val
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x, y, _t in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_seq = y.permute(0, 2, 1)
                y_pred, _, _ = model(x, None)
                loss = mse(y_pred, y_seq)
                total += float(loss) * x.size(0)
                n += x.size(0)
        v = total / max(1, n)

        if v < best_val:
            best_val = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@dataclass
class Row:
    model: str
    seed: int
    sharpe: float
    mean: float
    std: float
    n: int


def main():
    p = argparse.ArgumentParser()

    # your two runner scripts
    p.add_argument("--gp_runner_py", type=str, required=True, help="path to exp/01_forecasting/run_gp_mech.py (your main model runner)")
    p.add_argument("--baseline_runner_py", type=str, required=True, help="path to run_baselines.py")

    # data paths
    p.add_argument("--price_path", type=str, required=True)
    p.add_argument("--macro_path", type=str, required=True)
    p.add_argument("--edges_path", type=str, default="", help="required for mode=mech/prior_residual")

    # split + training
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)

    # model dims (shared)
    p.add_argument("--graph_hidden", type=int, default=32)
    p.add_argument("--gcn_hidden", type=int, default=32)
    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=8)

    # your model extra params
    p.add_argument("--gcn_dropout", type=float, default=0.0)
    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--rnn_dropout", type=float, default=0.0)
    p.add_argument("--graph_topk", type=int, default=-1)

    # backtest protocol
    p.add_argument("--top_frac", type=float, default=0.2)
    p.add_argument("--holding", choices=["1", "H"], default="H")
    p.add_argument("--nonoverlap", action="store_true")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--ann", type=int, default=252)


    # backtest variant
    p.add_argument("--strategy", choices=["selection", "overlay", "pairs"], default="selection",
                   help="selection: (longshort/longonly/active); overlay: EW risk overlay; pairs: mechanism-pairs RV")
    p.add_argument("--portfolio", choices=["longshort", "longonly", "longonly_active"], default="longshort",
                   help="only used when strategy=selection")
    p.add_argument("--overlay_wmax", type=float, default=1.0, help="max position size for overlay strategy")
    p.add_argument("--overlay_clipz", type=float, default=2.0, help="z-score clip for overlay strategy")
p.add_argument("--overlay_ema", type=float, default=0.0,
               help="EMA smoothing alpha on overlay z-signal (0 disables). e.g., 0.2")
p.add_argument("--overlay_thr", type=float, default=0.0,
               help="No-trade threshold on |z| for overlay (0 disables). e.g., 0.5")
p.add_argument("--overlay_tcost", type=float, default=0.0,
               help="Per-unit turnover transaction cost for overlay (0 disables). e.g., 0.0001 = 1bp")

p.add_argument("--pairs_ema", type=float, default=0.0,
               help="EMA smoothing for pair signals (0 disables). Suggested: 0.1~0.3")
p.add_argument("--pairs_thr", type=float, default=0.0,
               help="Deadzone threshold for pair signals (0 disables). If |signal| < thr => position=0")
p.add_argument("--pairs_pos_clip", type=float, default=1.0,
               help="Position magnitude per pair after sign(signal), clipped to [0, pairs_pos_clip]")
p.add_argument("--pairs_tcost", type=float, default=0.0,
               help="Per-unit turnover transaction cost for pairs (0 disables). e.g., 0.0001 = 1bp")
p.add_argument("--pairs_weighted", action="store_true",
               help="Weight pairs by mechanism edge weights instead of equal-weight")
    p.add_argument("--num_pairs", type=int, default=50, help="number of mechanism pairs for strategy=pairs")


    # model lists
    p.add_argument("--your_modes", type=str, default="prior_residual,learn,mech")
    p.add_argument("--baselines", type=str, default="stgnn,fouriergnn,patchtst,gru,lstm,tcn,transformer,mlp")
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")

    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    gm = import_from_path("gp_runner", args.gp_runner_py)
    rb = import_from_path("base_runner", args.baseline_runner_py)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    your_modes = [m.strip() for m in args.your_modes.split(",") if m.strip()]
    baselines = [m.strip() for m in args.baselines.split(",") if m.strip()]

    step = args.horizon if (args.holding == "H" and args.nonoverlap) else 1

    # ---------- shared data ----------
    # Use gp_runner's loader (same logic as your main model runner)
    price_df, returns_df, macro_df, price_cols, _ = gm.load_panel_from_two_files(args.price_path, args.macro_path)

    # Panel dataset used by your GPMechSTGNN runner (and we reuse for panel baselines)
    ds_panel = gm.PanelGraphDataset(returns_df, macro_df, window_size=args.window, horizon=args.horizon)
    train_ids, val_ids, test_ids = gm.make_time_split_from_t(ds_panel.valid_t, T=len(returns_df),
                                                            train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    panel_train_ds = Subset(ds_panel, train_ids)
    panel_val_ds   = Subset(ds_panel, val_ids)
    panel_test_ds  = Subset(ds_panel, test_ids)

    # For CSM signals
    returns_array = returns_df.values.astype(np.float32)

    # Build fixed y_true_seq + t_idx from panel test loader (order fixed)
    panel_test_loader_fixed = DataLoader(panel_test_ds, batch_size=args.batch, shuffle=False, drop_last=False)
    y_true_list, t_list = [], []
    for _x, y, t in panel_test_loader_fixed:
        y_true_list.append(y.numpy())
        t_list.append(np.asarray(t))
    y_true_seq_fixed = np.concatenate(y_true_list, axis=0)  # (T_test,N,H)
    t_idx_fixed = np.concatenate(t_list, axis=0).astype(int)

    # Quant baselines (panel)
    res_EW = backtest_equal_weight(y_true_seq_fixed, holding=args.holding, step=step, ann=args.ann)
    res_CSM = backtest_csm(returns_array, t_idx_fixed, y_true_seq_fixed,
                           lookback=args.lookback, top_frac=args.top_frac,
                           holding=args.holding, step=step, ann=args.ann)

    print("\n========== Backtest Protocol ==========")
    print(f"strategy={args.strategy} | portfolio={args.portfolio} | holding={args.holding} | step={step} | top_frac={args.top_frac} | ann={args.ann} | lookback={args.lookback} | overlay_wmax={args.overlay_wmax} | overlay_clipz={args.overlay_clipz} | overlay_ema={args.overlay_ema} | overlay_thr={args.overlay_thr} | overlay_tcost={args.overlay_tcost} | pairs_ema={args.pairs_ema} | pairs_thr={args.pairs_thr} | pairs_pos_clip={args.pairs_pos_clip} | pairs_tcost={args.pairs_tcost} | pairs_weighted={args.pairs_weighted} | num_pairs={args.num_pairs}")
    print(f"[EW ] Sharpe={res_EW['sharpe']:.4f} mean={res_EW['mean']:.6f} std={res_EW['std']:.6f} n={res_EW['n']}")
    print(f"[CSM] Sharpe={res_CSM['sharpe']:.4f} mean={res_CSM['mean']:.6f} std={res_CSM['std']:.6f} n={res_CSM['n']}")

    rows: List[Row] = []

    # ---------- your model: GPMechSTGNN ----------
    # Build A_mech once (used for mech/prior_residual)
    A_mech = None
    if any(m in ("mech", "prior_residual") for m in your_modes):
        if not args.edges_path:
            raise ValueError("--edges_path required when your_modes includes mech/prior_residual")
        if args.graph_topk < 0:
            graph_topk = None
        else:
            graph_topk = int(args.graph_topk)

        A_mech = gm.build_adjacency_from_edges(
            edges_path=args.edges_path,
            node_list=list(price_cols),
            weight_col="w",
            default_weight=1.0,
            self_loop=True,
            symmetrize=True,
        ).to(device)

    # infer dims
    x0, y0, _t0 = ds_panel[0]
    L, N, F_total = x0.shape

    from models.gp_mech_stgnn import GPMechSTGNN  # uses repo's models/

    for mode in your_modes:
        for sd in seeds:
            gm.set_seed(sd)

            train_loader = DataLoader(panel_train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
            val_loader   = DataLoader(panel_val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
            test_loader  = DataLoader(panel_test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)

            A_for_mode = A_mech if mode in ("mech", "prior_residual") else None

            model = GPMechSTGNN(
                num_nodes=N,
                in_dim=F_total,
                horizon=args.horizon,
                mode=mode,
                graph_hidden=args.graph_hidden,
                gcn_hidden=args.gcn_hidden,
                gru_hidden=args.gru_hidden,
                gcn_dropout=args.gcn_dropout,
                rnn_layers=args.rnn_layers,
                rnn_dropout=args.rnn_dropout,
                graph_topk=(None if args.graph_topk < 0 else int(args.graph_topk)),
            ).to(device)

            model = train_earlystop_panel(
                model, train_loader, val_loader, device, A_for_mode,
                epochs=args.epochs, patience=args.patience, lr=args.lr, wd=args.wd
            )

            score_mat, y_true_seq, _tidx = collect_scores_ytrue_t_panel(model, test_loader, device, A_for_mode)
            res = backtest_dispatch(score_mat, y_true_seq, args, node_list=price_cols, step=step)

            rows.append(Row(model=f"ours_gpmech_{mode}", seed=sd, sharpe=res["sharpe"], mean=res["mean"], std=res["std"], n=res["n"]))
            _turn = f" turnover={res['turnover']:.4f}" if "turnover" in res else ""
            print(f"[DONE] ours_gpmech_{mode:14s} seed={sd} Sharpe={res['sharpe']:.4f} mean={res['mean']:.6f} std={res['std']:.6f} n={res['n']}{_turn}")

    # ---------- baselines ----------
    # Baseline script already defines which model maps to which class; we replicate that mapping here.
    for bname in baselines:
        for sd in seeds:
            rb.set_seed(sd)

            if bname == "patchtst":
                # use baseline's PatchTSTDataset/split
                ds = rb.PatchTSTDataset(returns_df, macro_df, seq_len=args.window, pred_len=args.horizon)
                split = rb.make_time_split_patch(T=len(returns_df), seq_len=args.window, pred_len=args.horizon,
                                                 train_ratio=args.train_ratio, val_ratio=args.val_ratio)
                train_ds = Subset(ds, split.train_ids)
                val_ds   = Subset(ds, split.val_ids)
                test_ds  = Subset(ds, split.test_ids)

                train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
                val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)
                test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

                Fm = macro_df.shape[1] if macro_df is not None else 0
                from models.patchtst import PatchTST_Baseline
                model = PatchTST_Baseline(num_assets=N, num_macros=Fm, seq_len=args.window, horizon=args.horizon,
                                          d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                                          dropout=args.dropout).to(device)

                model = train_earlystop_patchtst(model, train_loader, val_loader, device,
                                                 epochs=args.epochs, patience=args.patience, lr=args.lr, wd=args.wd)

                score_mat, y_true_seq, _tidx = collect_scores_ytrue_t_patchtst(model, test_loader, device)
                res = backtest_dispatch(score_mat, y_true_seq, args, node_list=price_cols, step=step)

                rows.append(Row(model=f"base_{bname}", seed=sd, sharpe=res["sharpe"], mean=res["mean"], std=res["std"], n=res["n"]))
                _turn = f" turnover={res['turnover']:.4f}" if "turnover" in res else ""
                print(f"[DONE] base_{bname:14s} seed={sd} Sharpe={res['sharpe']:.4f} mean={res['mean']:.6f} std={res['std']:.6f} n={res['n']}{_turn}")
                continue

            # panel baselines use the same panel dataset as your model
            train_loader = DataLoader(panel_train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
            val_loader   = DataLoader(panel_val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
            test_loader  = DataLoader(panel_test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)

            if bname == "stgnn":
                from models.stgnn import STGNN_LearnOnly
                model = STGNN_LearnOnly(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                        gcn_hidden=args.gcn_hidden, gru_hidden=args.gru_hidden,
                                        graph_hidden=args.graph_hidden, dropout=args.dropout).to(device)
            elif bname == "fouriergnn":
                from models.fouriergnn import FourierGNN_LearnOnly
                model = FourierGNN_LearnOnly(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                             d_model=args.d_model, n_layers=args.n_layers,
                                             graph_hidden=args.graph_hidden, dropout=args.dropout).to(device)
            elif bname in {"gru", "lstm", "tcn", "transformer", "mlp"}:
                from models.classical_baselines import (
                    GRU_Baseline, LSTM_Baseline, TCN_Baseline, Transformer_Baseline, MLP_Baseline
                )
                if bname == "gru":
                    model = GRU_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                         hidden_dim=args.gru_hidden, use_graph_logging=True,
                                         graph_hidden=args.graph_hidden).to(device)
                elif bname == "lstm":
                    model = LSTM_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                          hidden_dim=args.gru_hidden, use_graph_logging=True,
                                          graph_hidden=args.graph_hidden).to(device)
                elif bname == "tcn":
                    model = TCN_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                         channel=args.gcn_hidden, use_graph_logging=True,
                                         graph_hidden=args.graph_hidden).to(device)
                elif bname == "transformer":
                    model = Transformer_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon,
                                                 d_model=args.d_model, nhead=args.n_heads, num_layers=args.n_layers,
                                                 dropout=args.dropout, use_graph_logging=True,
                                                 graph_hidden=args.graph_hidden).to(device)
                else:
                    model = MLP_Baseline(num_nodes=N, in_dim=F_total, horizon=args.horizon, window_len=args.window,
                                         hidden_dim=args.d_model, dropout=args.dropout, use_graph_logging=True,
                                         graph_hidden=args.graph_hidden).to(device)
            else:
                raise ValueError(f"Unknown baseline: {bname}")

            model = train_earlystop_panel(
                model, train_loader, val_loader, device, A=None,
                epochs=args.epochs, patience=args.patience, lr=args.lr, wd=args.wd
            )

            score_mat, y_true_seq, _tidx = collect_scores_ytrue_t_panel(model, test_loader, device, A=None)
            res = backtest_dispatch(score_mat, y_true_seq, args, node_list=price_cols, step=step)

            rows.append(Row(model=f"base_{bname}", seed=sd, sharpe=res["sharpe"], mean=res["mean"], std=res["std"], n=res["n"]))
            _turn = f" turnover={res['turnover']:.4f}" if "turnover" in res else ""
            print(f"[DONE] base_{bname:14s} seed={sd} Sharpe={res['sharpe']:.4f} mean={res['mean']:.6f} std={res['std']:.6f} n={res['n']}{_turn}")

    # ---------- summary ----------
    print("\n========== Summary (Sharpe mean ± std over seeds) ==========")
    names = sorted(set(r.model for r in rows))
    for name in names:
        vals = np.array([r.sharpe for r in rows if r.model == name], dtype=float)
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        print(f"{name:22s} Sharpe={mu:.4f} ± {sd:.4f}  (n_seeds={len(vals)})")

    print("\n[Quant baselines, no seed]")
    print(f"EW  Sharpe={res_EW['sharpe']:.4f}")
    print(f"CSM Sharpe={res_CSM['sharpe']:.4f}")


if __name__ == "__main__":
    main()


