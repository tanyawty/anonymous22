# run_compare_sharpe.py
# Compare Sharpe ratio for:
#   - Your model: mech / learn / prior_residual (from gp_mech_multitask_stgnn.py)
#   - Baselines: stgnn / fouriergnn / gru / lstm / tcn / transformer / mlp (from run_baselines.py ecosystem)
#   - Quant baselines: EW / CSM
#
# It reuses:
#   - run_baselines.py: load_panel_from_two_files, PanelGraphDataset, make_time_split_from_t
#   - gp_mech_multitask_stgnn.py: MechAware_GP_STGNN_MultiTask (your model)
#
# NOTE: This script uses PanelGraphDataset features for ALL models, including yours.

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# -------------------------
# Dynamic import helpers
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

@torch.no_grad()
def collect_scores_ytrue_t(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_A_mech: bool,
    A_mech: Optional[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    loader yields: (x_seq, y_seq, t)
      x_seq: (B,L,N,F)
      y_seq: (B,N,H)
      t:     (B,)
    model forward expected: y_pred, _, _ = model(x_seq, A_mech_or_None)
      y_pred: (B,N,H)
    returns:
      score_mat: (T_test,N) = sum_h y_pred
      y_true:    (T_test,N,H)
      t_idx:     (T_test,)
    """
    model.eval()
    scores, ytrues, ts = [], [], []
    for x_seq, y_seq, t in loader:
        x_seq = x_seq.to(device)
        y_seq = y_seq.to(device)

        if use_A_mech:
            assert A_mech is not None
            y_pred, _, _ = model(x_seq, A_mech)
        else:
            y_pred, _, _ = model(x_seq, None)

        score = y_pred.sum(dim=-1)  # (B,N)
        scores.append(score.detach().cpu().numpy())
        ytrues.append(y_seq.detach().cpu().numpy())
        ts.append(np.asarray(t))

    score_mat = np.concatenate(scores, axis=0)
    y_true = np.concatenate(ytrues, axis=0)
    t_idx = np.concatenate(ts, axis=0).astype(int)
    return score_mat, y_true, t_idx

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
# Training loops (generic)
# -------------------------

def train_val_earlystop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    wd: float,
    use_A_mech: bool,
    A_mech: Optional[torch.Tensor],
) -> nn.Module:
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val = float("inf")
    best_state = None
    bad = 0

    for _ep in range(epochs):
        model.train()
        for x_seq, y_seq, _t in train_loader:
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)

            opt.zero_grad()
            if use_A_mech:
                y_pred, _, _ = model(x_seq, A_mech)
            else:
                y_pred, _, _ = model(x_seq, None)
            loss = mse(y_pred, y_seq)
            loss.backward()
            opt.step()

        # val
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x_seq, y_seq, _t in val_loader:
                x_seq = x_seq.to(device)
                y_seq = y_seq.to(device)
                if use_A_mech:
                    y_pred, _, _ = model(x_seq, A_mech)
                else:
                    y_pred, _, _ = model(x_seq, None)
                loss = mse(y_pred, y_seq)
                total += float(loss) * x_seq.size(0)
                n += x_seq.size(0)
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


# -------------------------
# Baseline model builders
# -------------------------

def build_baseline_model(rb, model_name: str, N: int, F_total: int, horizon: int, args) -> nn.Module:
    """
    Build baseline models exactly like run_baselines.py does.
    They all follow: y_pred, _, _ = model(x_seq, None)
    """
    if model_name == "stgnn":
        from models.stgnn import STGNN_LearnOnly
        return STGNN_LearnOnly(
            num_nodes=N, in_dim=F_total, horizon=horizon,
            gcn_hidden=args.gcn_hidden, gru_hidden=args.gru_hidden,
            graph_hidden=args.graph_hidden, dropout=args.dropout
        )
    if model_name == "fouriergnn":
        from models.fouriergnn import FourierGNN_LearnOnly
        return FourierGNN_LearnOnly(
            num_nodes=N, in_dim=F_total, horizon=horizon,
            d_model=args.d_model, n_layers=args.n_layers,
            graph_hidden=args.graph_hidden, dropout=args.dropout
        )
    if model_name in {"gru", "lstm", "tcn", "transformer", "mlp"}:
        from models.classical_baselines import (
            GRU_Baseline, LSTM_Baseline, TCN_Baseline, Transformer_Baseline, MLP_Baseline
        )
        if model_name == "gru":
            return GRU_Baseline(num_nodes=N, in_dim=F_total, horizon=horizon,
                                hidden_dim=args.gru_hidden, use_graph_logging=True,
                                graph_hidden=args.graph_hidden)
        if model_name == "lstm":
            return LSTM_Baseline(num_nodes=N, in_dim=F_total, horizon=horizon,
                                 hidden_dim=args.gru_hidden, use_graph_logging=True,
                                 graph_hidden=args.graph_hidden)
        if model_name == "tcn":
            return TCN_Baseline(num_nodes=N, in_dim=F_total, horizon=horizon,
                                channel=args.gcn_hidden, use_graph_logging=True,
                                graph_hidden=args.graph_hidden)
        if model_name == "transformer":
            return Transformer_Baseline(num_nodes=N, in_dim=F_total, horizon=horizon,
                                        d_model=args.d_model, nhead=args.n_heads, num_layers=args.n_layers,
                                        dropout=args.dropout, use_graph_logging=True,
                                        graph_hidden=args.graph_hidden)
        return MLP_Baseline(num_nodes=N, in_dim=F_total, horizon=horizon, window_len=args.window,
                            hidden_dim=args.d_model, dropout=args.dropout, use_graph_logging=True,
                            graph_hidden=args.graph_hidden)

    raise ValueError(f"Unknown baseline model: {model_name}")


# -------------------------
# Result record
# -------------------------

@dataclass
class Row:
    model: str
    seed: int
    sharpe: float
    mean: float
    std: float
    n: int


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--run_baselines_py", type=str, required=True, help="path to your run_baselines.py")
    p.add_argument("--gp_mech_py", type=str, required=True, help="path to your gp_mech_multitask_stgnn.py")

    p.add_argument("--price_path", type=str, required=True)
    p.add_argument("--macro_path", type=str, required=True)
    p.add_argument("--edges_path", type=str, default="", help="(optional) edges csv for A_mech; required for your model modes")

    # data / split
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    # training
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)

    # baseline hyperparams (match run_baselines defaults)
    p.add_argument("--gcn_hidden", type=int, default=32)
    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--graph_hidden", type=int, default=32)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # backtest protocol
    p.add_argument("--top_frac", type=float, default=0.2)
    p.add_argument("--holding", choices=["1", "H"], default="H")
    p.add_argument("--nonoverlap", action="store_true")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--ann", type=int, default=252)

    # which models
    p.add_argument("--your_modes", type=str, default="prior_residual,learn,mech",
                   help="comma-separated: prior_residual,learn,mech")
    p.add_argument("--baselines", type=str, default="stgnn,fouriergnn,gru,lstm,tcn,transformer,mlp",
                   help="comma-separated baselines")
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")

    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    # import two scripts
    rb = import_from_path("rb", args.run_baselines_py)
    gm = import_from_path("gm", args.gp_mech_py)

    # device + seeds
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    your_modes = [m.strip() for m in args.your_modes.split(",") if m.strip()]
    baselines = [m.strip() for m in args.baselines.split(",") if m.strip()]

    # fixed step for backtest
    step = args.horizon if (args.holding == "H" and args.nonoverlap) else 1
    tag_step = "non-overlap" if (args.holding == "H" and args.nonoverlap) else "overlap"

    # Load data ONCE (same for all models)
    price_df, returns_df, macro_df, price_cols, macro_cols = rb.load_panel_from_two_files(args.price_path, args.macro_path)

    # Build dataset/split ONCE (same for all models)
    ds = rb.PanelGraphDataset(returns_df, price_df, macro_df, window_size=args.window, horizon=args.horizon)
    split = rb.make_time_split_from_t(ds.valid_t, T=len(returns_df), train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    train_ds = Subset(ds, split.train_ids)
    val_ds   = Subset(ds, split.val_ids)
    test_ds  = Subset(ds, split.test_ids)

    # NOTE: shuffle True only affects train loader; test loader stays ordered
    def make_loaders(seed: int):
        rb.set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)
        return train_loader, val_loader, test_loader

    # Prepare A_mech for your model modes (needs edges_path)
    A_mech = None
    if your_modes:
        if not args.edges_path:
            raise ValueError("You requested your model modes, but --edges_path is empty. Provide edges csv.")
        # gp_mech_multitask_stgnn.py provides build_adjacency_from_edges
        # signature in your file: build_adjacency_from_edges(edges_path, node_list, ...)
        # node_list should be price_cols (asset names)
        A_mech = gm.build_adjacency_from_edges(
            edges_path=args.edges_path,
            node_list=price_cols,
            weight_col="w",
            default_weight=1.0,
            self_loop=True,
            symmetrize=True,
        ).to(device)

    # For CSM signals we use realized returns array aligned by time index t (same as ds.ret)
    returns_array = returns_df.values.astype(np.float32)

    # Gather y_true_seq + t_idx once (needed for EW/CSM, and reused)
    # We can take it from the first seed's test loader (order fixed).
    _train0, _val0, test0 = make_loaders(seeds[0])
    # Use a dummy model to just fetch y_true_seq & t_idx
    y_true_list, t_list = [], []
    for _x, y, t in test0:
        y_true_list.append(y.numpy())
        t_list.append(np.asarray(t))
    y_true_seq_fixed = np.concatenate(y_true_list, axis=0)  # (T_test,N,H)
    t_idx_fixed = np.concatenate(t_list, axis=0).astype(int)

    # Quant baselines (no training)
    res_EW  = backtest_equal_weight(y_true_seq_fixed, holding=args.holding, step=step, ann=args.ann)
    res_CSM = backtest_csm(returns_array, t_idx_fixed, y_true_seq_fixed,
                           lookback=args.lookback, top_frac=args.top_frac,
                           holding=args.holding, step=step, ann=args.ann)

    print("\n========== Backtest Protocol ==========")
    print(f"holding={args.holding} | rebalance={tag_step} | top_frac={args.top_frac} | ann={args.ann} | lookback={args.lookback}")
    print(f"[EW]  Sharpe={res_EW['sharpe']:.4f} mean={res_EW['mean']:.6f} std={res_EW['std']:.6f} n={res_EW['n']}")
    print(f"[CSM] Sharpe={res_CSM['sharpe']:.4f} mean={res_CSM['mean']:.6f} std={res_CSM['std']:.6f} n={res_CSM['n']}")

    rows: List[Row] = []

    # ---- run your model modes ----
    for mode in your_modes:
        for sd in seeds:
            gm.set_seed(sd)
            train_loader, val_loader, test_loader = make_loaders(sd)

            # infer dims
            x0, y0, _t0 = ds[0]
            L, N, F_total = x0.shape

            model = gm.MechAware_GP_STGNN_MultiTask(
                num_nodes=N, in_dim=F_total, mode=mode,
                gcn_hidden=args.gcn_hidden, gru_hidden=args.gru_hidden,
                graph_hidden=args.graph_hidden, horizon=args.horizon
            ).to(device)

            model = train_val_earlystop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                wd=args.wd,
                use_A_mech=True,
                A_mech=A_mech,
            )

            score_mat, y_true_seq, t_idx = collect_scores_ytrue_t(
                model=model, loader=test_loader, device=device,
                use_A_mech=True, A_mech=A_mech
            )
            res = backtest_long_short(score_mat, y_true_seq, top_frac=args.top_frac,
                                      holding=args.holding, step=step, ann=args.ann)

            rows.append(Row(model=f"ours_{mode}", seed=sd, sharpe=res["sharpe"], mean=res["mean"], std=res["std"], n=res["n"]))
            print(f"[DONE] ours_{mode:14s} seed={sd} Sharpe={res['sharpe']:.4f} mean={res['mean']:.6f} std={res['std']:.6f} n={res['n']}")

    # ---- run baselines ----
    for bname in baselines:
        for sd in seeds:
            rb.set_seed(sd)
            train_loader, val_loader, test_loader = make_loaders(sd)

            # infer dims
            x0, y0, _t0 = ds[0]
            L, N, F_total = x0.shape

            model = build_baseline_model(rb, bname, N=N, F_total=F_total, horizon=args.horizon, args=args).to(device)
            model = train_val_earlystop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                wd=args.wd,
                use_A_mech=False,
                A_mech=None
            )

            score_mat, y_true_seq, t_idx = collect_scores_ytrue_t(
                model=model, loader=test_loader, device=device,
                use_A_mech=False, A_mech=None
            )
            res = backtest_long_short(score_mat, y_true_seq, top_frac=args.top_frac,
                                      holding=args.holding, step=step, ann=args.ann)

            rows.append(Row(model=f"base_{bname}", seed=sd, sharpe=res["sharpe"], mean=res["mean"], std=res["std"], n=res["n"]))
            print(f"[DONE] base_{bname:14s} seed={sd} Sharpe={res['sharpe']:.4f} mean={res['mean']:.6f} std={res['std']:.6f} n={res['n']}")

    # ---- summarize (mean±std over seeds) ----
    print("\n========== Summary (Sharpe mean ± std over seeds) ==========")
    model_names = sorted(set(r.model for r in rows))
    for mn in model_names:
        vals = np.array([r.sharpe for r in rows if r.model == mn], dtype=float)
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        print(f"{mn:18s} Sharpe={mu:.4f} ± {sd:.4f}  (n_seeds={len(vals)})")

    print("\n[NOTE] EW/CSM (no seed):")
    print(f"EW  Sharpe={res_EW['sharpe']:.4f}")
    print(f"CSM Sharpe={res_CSM['sharpe']:.4f}")


if __name__ == "__main__":
    main()
