# patchtst_fixed.py
# Timewise PatchTST baseline: predict RETURNS (N assets) with MACROS (F features) as extra inputs.
# Report derived tasks PF/MA/GAP computed from predicted returns + true starting prices.
#
# Shapes:
#   x:   (B, seq_len, N+F)   = [returns, macros]
#   y:   (B, pred_len, N)    = future returns (log-returns)
#   p0:  (B, N)              = price at t0 (last time step of input window)
#   yhat from HF PatchTST: (B, pred_len, N+F) -> we supervise only first N
#
# Note: returns are computed as log(price).diff() in the loader.

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import PatchTSTConfig, PatchTSTForPrediction


# -----------------------------
# Utilities
# -----------------------------
MACROS_5 = ["dxy", "us10y", "vix", "wti", "gscpi"]  # keep only those present in macro file


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    diff = (y_pred - y_true).astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse


@dataclass
class SplitIndex:
    train_end: int
    val_end: int


def make_time_splits(T: int, train_ratio=0.7, val_ratio=0.15) -> SplitIndex:
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, T - 2))
    val_end = max(train_end + 1, min(val_end, T - 1))
    return SplitIndex(train_end=train_end, val_end=val_end)


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """
    Causal rolling mean along time axis=1, with expanding mean for first w-1 steps.
    x: (S, T, N)
    """
    if w <= 1:
        return x.copy().astype(np.float32)
    S, T, N = x.shape
    out = np.zeros_like(x, dtype=np.float32)
    csum = np.cumsum(x, axis=1, dtype=np.float64)
    for t in range(T):
        l = max(0, t - w + 1)
        denom = (t - l + 1)
        if l == 0:
            out[:, t, :] = (csum[:, t, :] / denom).astype(np.float32)
        else:
            out[:, t, :] = ((csum[:, t, :] - csum[:, l - 1, :]) / denom).astype(np.float32)
    return out


def returns_to_price_path(log_returns: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """
    Convert LOG-returns to price path.
    log_returns: (S, pred_len, N)
    p0:          (S, 1, N)
    """
    logp = np.cumsum(log_returns, axis=1, dtype=np.float64)
    return (p0 * np.exp(logp)).astype(np.float32)


# -----------------------------
# Data loading
# -----------------------------
def load_panel_from_two_files(price_path: str, macro_path: str):
    """
    price CSV: first col is date, rest numeric assets prices
    macro CSV: first col is date, rest numeric macro indicators
    """
    df_p = pd.read_csv(price_path)
    date_col_p = df_p.columns[0]
    df_p[date_col_p] = pd.to_datetime(df_p[date_col_p], dayfirst=True, errors="coerce")
    df_p = df_p.set_index(date_col_p).sort_index()

    price_cols = [c for c in df_p.columns if np.issubdtype(df_p[c].dtype, np.number)]
    price_df = df_p[price_cols].astype("float32")
    price_df = price_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    df_m = pd.read_csv(macro_path)
    date_col_m = df_m.columns[0]
    df_m[date_col_m] = pd.to_datetime(df_m[date_col_m], dayfirst=True, errors="coerce")
    df_m = df_m.set_index(date_col_m).sort_index()

    macro_cols = [c for c in df_m.columns if np.issubdtype(df_m[c].dtype, np.number)]
    macro_df = df_m[macro_cols].astype("float32")
    macro_df = macro_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    common_idx = price_df.index.intersection(macro_df.index)
    price_df = price_df.loc[common_idx].sort_index()
    macro_df = macro_df.loc[common_idx].sort_index()

    # log returns
    rets = np.log(price_df).diff().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(axis=0, how="all").fillna(0.0)
    returns_df = rets.astype("float32")

    # align price/macro to returns index
    price_df = price_df.loc[returns_df.index]
    macro_df = macro_df.loc[returns_df.index]

    return price_df, returns_df, macro_df, price_cols, macro_cols


def load_panel_and_macro(panel_prices: str, panel_macro: str):
    price_df, returns_df, macro_df, price_cols, _ = load_panel_from_two_files(panel_prices, panel_macro)

    prices = price_df[price_cols].to_numpy(dtype=np.float32)     # (T, N)
    returns = returns_df[price_cols].to_numpy(dtype=np.float32)  # (T, N)

    use_macros = [c for c in MACROS_5 if c in macro_df.columns]
    if len(use_macros) == 0:
        use_macros = [c for c in macro_df.columns if np.issubdtype(macro_df[c].dtype, np.number)]
    macros = macro_df[use_macros].to_numpy(dtype=np.float32)     # (T, F)

    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    macros = np.nan_to_num(macros, nan=0.0, posinf=0.0, neginf=0.0)

    return prices, returns, macros


# -----------------------------
# Dataset
# -----------------------------
class PanelMacroReturnDatasetTimewise(Dataset):
    def __init__(self, returns, macros, seq_len, pred_len):
        self.returns = returns.astype(np.float32)  # (T,N)
        self.macros  = macros.astype(np.float32)   # (T,F)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.T, self.N = self.returns.shape
        self.F = self.macros.shape[1]
        self.max_i = self.T - (self.seq_len + self.pred_len) + 1
        if self.max_i <= 0:
            raise ValueError("Not enough length for given seq_len/pred_len")

    def __len__(self):
        return self.max_i

    def __getitem__(self, i):
        x_ret = self.returns[i:i+self.seq_len]          # (L,N)
        x_mac = self.macros[i:i+self.seq_len]           # (L,F)
        x = np.concatenate([x_ret, x_mac], axis=1)      # (L,N+F)

        y = self.returns[i+self.seq_len:i+self.seq_len+self.pred_len]  # (H,N)
        return torch.from_numpy(x), torch.from_numpy(y)




# -----------------------------
# Evaluation: derived tasks from predicted returns
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, N: int):
    model.eval()

    pf_t_all, pf_p_all = [], []
    ma_t_all, ma_p_all = [], []
    gap_t_all, gap_p_all = [], []

    for x, y in loader:
        x = x.to(device)  # (B, L, N+F)
        y = y.to(device)  # (B, H, N)

        out = model(past_values=x)
        yhat = out.prediction_outputs[:, :, :N]  # (B, H, N) predicted returns

        # convert to (B, N, H) to match gp_mech_multitask_stgnn.py style
        y_seq_true = y.permute(0, 2, 1)     # (B,N,H)
        y_seq_pred = yhat.permute(0, 2, 1)  # (B,N,H)

        # EXACT same definitions as seq_to_pf_ma_gap
        pf_true = y_seq_true.sum(dim=-1)  # (B,N)
        ma_true = y_seq_true.mean(dim=-1)
        gap_true = y_seq_true.max(dim=-1).values - y_seq_true.min(dim=-1).values

        pf_pred = y_seq_pred.sum(dim=-1)
        ma_pred = y_seq_pred.mean(dim=-1)
        gap_pred = y_seq_pred.max(dim=-1).values - y_seq_pred.min(dim=-1).values

        pf_t_all.append(pf_true.detach().cpu().numpy())
        pf_p_all.append(pf_pred.detach().cpu().numpy())
        ma_t_all.append(ma_true.detach().cpu().numpy())
        ma_p_all.append(ma_pred.detach().cpu().numpy())
        gap_t_all.append(gap_true.detach().cpu().numpy())
        gap_p_all.append(gap_pred.detach().cpu().numpy())

    # flatten all assets & batches together
    pf_true = np.concatenate([a.reshape(-1) for a in pf_t_all], axis=0)
    pf_pred = np.concatenate([a.reshape(-1) for a in pf_p_all], axis=0)
    ma_true = np.concatenate([a.reshape(-1) for a in ma_t_all], axis=0)
    ma_pred = np.concatenate([a.reshape(-1) for a in ma_p_all], axis=0)
    gap_true = np.concatenate([a.reshape(-1) for a in gap_t_all], axis=0)
    gap_pred = np.concatenate([a.reshape(-1) for a in gap_p_all], axis=0)

    pf_mae, pf_rmse = compute_mae_rmse(pf_true, pf_pred)
    ma_mae, ma_rmse = compute_mae_rmse(ma_true, ma_pred)
    gap_mae, gap_rmse = compute_mae_rmse(gap_true, gap_pred)

    return (pf_mae, pf_rmse), (ma_mae, ma_rmse), (gap_mae, gap_rmse)




# -----------------------------
# Train one seed
# -----------------------------
def train_one_seed(args, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    prices, returns, macros = load_panel_and_macro(args.panel_prices, args.panel_macro)
    print("[INFO] T,N,F =", returns.shape[0], returns.shape[1], macros.shape[1])
    print("[INFO] returns nan/inf:", np.isnan(returns).sum(), np.isinf(returns).sum())
    print("[INFO] macros  nan/inf:", np.isnan(macros).sum(), np.isinf(macros).sum())

    T, N = returns.shape
    F = macros.shape[1]

    split = make_time_splits(T, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_end, val_end = split.train_end, split.val_end

    # normalize macros using train split only
    mu = macros[:train_end].mean(axis=0, keepdims=True)
    sd = macros[:train_end].std(axis=0, keepdims=True) + 1e-8
    macros = (macros - mu) / sd

    # split by time
    prices_train, returns_train, macros_train = prices[:train_end], returns[:train_end], macros[:train_end]
    prices_val,   returns_val,   macros_val   = prices[train_end:val_end], returns[train_end:val_end], macros[train_end:val_end]
    prices_test,  returns_test,  macros_test  = prices[val_end:], returns[val_end:], macros[val_end:]

    train_ds = PanelMacroReturnDatasetTimewise(returns_train, macros_train, args.seq_len, args.pred_len)
    val_ds = PanelMacroReturnDatasetTimewise(returns_val, macros_val, args.seq_len, args.pred_len)
    test_ds = PanelMacroReturnDatasetTimewise(returns_test, macros_test, args.seq_len, args.pred_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    cfg = PatchTSTConfig(
        num_input_channels=N + F,
        context_length=args.seq_len,
        prediction_length=args.pred_len,
        patch_length=args.patch_len,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    model = PatchTSTForPrediction(cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    best_score = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        cnt = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            out = model(past_values=x)
            yhat = out.prediction_outputs[:, :, :N]
            loss = loss_fn(yhat, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += float(loss.item())
            cnt += 1

        train_mse = total / max(cnt, 1)

        (pf_mae, pf_rmse), (ma_mae, ma_rmse), (gap_mae, gap_rmse) = evaluate(model, val_loader, device, N)


        # early-stopping metric
        score_map = {"pf": pf_mae, "ma": ma_mae, "gap": gap_mae}

        score = score_map.get(args.es_metric, pf_mae)


        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # print(
        #     f"[EP {ep:03d}] train_mse={train_mse:.6f} "
        #     f"VAL(ret) MAE={ret_mae:.6f} RMSE={ret_rmse:.6f} | "
        #     f"PF MAE={pf_mae:.6f} RMSE={pf_rmse:.6f} | "
        #     f"MA MAE={ma_mae:.6f} RMSE={ma_rmse:.6f} | "
        #     f"GAP MAE={gap_mae:.6f} RMSE={gap_rmse:.6f} | "
        #     f"best({args.es_metric})={best_score:.6f} bad={bad_epochs}"
        # )

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[EARLY STOP] patience={args.patience} reached at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # final test metrics
    (pf_mae, pf_rmse), (ma_mae, ma_rmse), (gap_mae, gap_rmse) = evaluate(model, test_loader, device, N)

    print(f"========== MODEL=PatchTST SEED={seed} ==========")
    print(f"[TEST] PF  MAE={pf_mae:.6f} RMSE={pf_rmse:.6f}")
    print(f"[TEST] MA  MAE={ma_mae:.6f} RMSE={ma_rmse:.6f}")
    print(f"[TEST] GAP MAE={gap_mae:.6f} RMSE={gap_rmse:.6f}")

    return {
        "seed": seed,
        "test_pf_mae": pf_mae, "test_pf_rmse": pf_rmse,
        "test_ma_mae": ma_mae, "test_ma_rmse": ma_rmse,
        "test_gap_mae": gap_mae, "test_gap_rmse": gap_rmse,
    }


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--panel_prices", type=str, required=True)
    ap.add_argument("--panel_macro", type=str, required=True)

    ap.add_argument("--train_ratio", type=float, default=0.3)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--pred_len", type=int, default=5)

    ap.add_argument("--patch_len", type=int, default=8)
    ap.add_argument("--patch_stride", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--ffn_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--ma_window", type=int, default=5)
    ap.add_argument("--es_metric", type=str, default="ret", choices=["ret", "pf", "ma", "gap"])

    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")
    ap.add_argument("--debug", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    results = []
    for s in seeds:
        results.append(train_one_seed(args, s))

    def agg(key):
        vals = [r[key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    keys = [
        "test_ret_mae", "test_pf_mae", "test_ma_mae", "test_gap_mae",
        "test_ret_rmse", "test_pf_rmse", "test_ma_rmse", "test_gap_rmse",
    ]

    print("\n========== SUMMARY (mean ± std over seeds) ==========")
    for k in keys:
        m, sd = agg(k)
        print(f"{k}: {m:.6f} ± {sd:.6f}")


if __name__ == "__main__":
    main()
