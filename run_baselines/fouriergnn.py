# gp_mech_multitask_fouriergnn.py
# Learn-only baseline:
#   Multi-task (PF / MA / GAP) + Learned graph (for logging) + FourierGNN encoder
#
# Drop-in rewrite of gp_mech_multitask_stgnn.py: replace STGNN (GCN+GRU) with FourierGNN.
# Output remains future return sequence y_seq: (B, N, H). PF/MA/GAP are derived from y_seq.

import numpy as np
import pandas as pd
import math
import random
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

warnings.filterwarnings("ignore")


# =============== 1. Utils ===============

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_panel_from_two_files(price_path, macro_path):
    """
    From panel_prices.csv and panel_macro.csv:
      - panel_prices: all columns are price series
      - panel_macro:  all columns are macro factors
    """
    # prices
    df_p = pd.read_csv(price_path)
    date_col_p = df_p.columns[0]
    df_p[date_col_p] = pd.to_datetime(df_p[date_col_p], dayfirst=True, errors="coerce")
    df_p = df_p.set_index(date_col_p).sort_index()

    price_cols = list(df_p.columns)
    price_df = df_p[price_cols].astype("float32")
    price_df = price_df.replace([np.inf, -np.inf], np.nan)
    price_df = price_df.ffill().bfill().fillna(0.0)

    # macro
    df_m = pd.read_csv(macro_path)
    date_col_m = df_m.columns[0]
    df_m[date_col_m] = pd.to_datetime(df_m[date_col_m], dayfirst=True, errors="coerce")
    df_m = df_m.set_index(date_col_m).sort_index()

    macro_cols = list(df_m.columns)
    macro_df = df_m[macro_cols].astype("float32")

    # align by date intersection
    common_idx = price_df.index.intersection(macro_df.index)
    price_df = price_df.loc[common_idx].sort_index()
    macro_df = macro_df.loc[common_idx].sort_index()

    # log returns
    rets = np.log(price_df).diff().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(axis=0, how="all").fillna(0.0)
    returns_df = rets.astype("float32")

    # align again to returns index
    price_df = price_df.loc[returns_df.index]
    macro_df = macro_df.loc[returns_df.index]

    return price_df, returns_df, macro_df, price_cols, macro_cols


# =============== 2. Dataset (keeps your "y_seq future returns" label) ===============

class GPMultiTaskDataset(Dataset):
    """
    Input:
      x_seq: (L, N, F_total)

    Label:
      y_seq: (N, H)  future returns sequence (H, N) -> transpose
    """

    def __init__(self, returns_df, price_df, macro_df, window_size=20, horizon=5):
        self.window_size = window_size
        self.horizon = horizon

        self.dates = returns_df.index
        self.T, self.N = returns_df.shape

        self.ret = returns_df.values.astype(np.float32)                  # (T,N)
        self.price = price_df.loc[self.dates].values.astype(np.float32)  # (T,N)

        self.macro = None
        self.F_macro = 0
        if macro_df is not None and macro_df.shape[1] > 0:
            macro_arr = macro_df.loc[self.dates].values.astype(np.float32)
            macro_arr = pd.DataFrame(macro_arr, index=self.dates).ffill().fillna(0.0).values
            self.macro = macro_arr.astype(np.float32)  # (T,Fm)
            self.F_macro = self.macro.shape[1]

        self.raw_ret = self.ret
        self.valid_idx = list(range(window_size - 1, self.T - horizon))

    def __len__(self):
        return len(self.valid_idx)

    def _rolling_mean(self, x, win):
        L, N = x.shape
        out = np.empty_like(x, dtype=np.float32)
        csum = np.cumsum(x, axis=0, dtype=np.float64)
        for i in range(L):
            start = max(0, i - win + 1)
            if start == 0:
                s = csum[i]
            else:
                s = csum[i] - csum[start - 1]
            n = i - start + 1
            out[i] = s / n
        return out

    def _rolling_std(self, x, win):
        L, N = x.shape
        out = np.empty_like(x, dtype=np.float32)
        for i in range(L):
            start = max(0, i - win + 1)
            seg = x[start:i+1]
            out[i] = seg.std(axis=0)
        return out

    def __getitem__(self, idx):
        t = self.valid_idx[idx]
        L = self.window_size
        h = self.horizon

        start = t - (L - 1)
        end = t + 1

        ret_win = self.ret[start:end, :]  # (L,N)

        # --- keep a compact, stable feature set (you can expand to match your original list if you want) ---
        feat_ret_1 = ret_win
        feat_ret_3 = self._rolling_mean(ret_win, 3)
        feat_ret_5 = self._rolling_mean(ret_win, 5)
        feat_vol_5 = self._rolling_std(ret_win, 5)

        node_feat = np.stack([feat_ret_1, feat_ret_3, feat_ret_5, feat_vol_5], axis=-1).astype(np.float32)
        node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # local standardization within window
        L_, N_, F_node = node_feat.shape
        flat = node_feat.reshape(L_ * N_, F_node)
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True) + 1e-8
        node_feat = ((flat - mean) / std).reshape(L_, N_, F_node).astype(np.float32)

        # macro broadcast
        if self.macro is not None:
            macro_win = self.macro[start:end, :]      # (L,Fm)
            macro_win = macro_win[:, None, :]         # (L,1,Fm)
            macro_win = np.repeat(macro_win, self.N, axis=1)  # (L,N,Fm)
            x_seq = np.concatenate([node_feat, macro_win], axis=-1)
        else:
            x_seq = node_feat

        x_seq = torch.tensor(x_seq, dtype=torch.float32)  # (L,N,F_total)

        # future return sequence label: (H,N) -> (N,H)
        t_start = t + 1
        t_end = t + 1 + h
        future_ret = self.raw_ret[t_start:t_end, :]  # (H,N)
        y_seq = torch.tensor(future_ret.T, dtype=torch.float32)  # (N,H)

        return x_seq, y_seq


# =============== 3. Learned Graph (kept for logging) + FourierGNN encoder ===============

class LearnedGraphAttn(nn.Module):
    """
    H_node: (B,N,F_in) -> A_learn: (B,N,N)
    (kept for consistency / logging; learn-only mode uses it only as an auxiliary output)
    """
    def __init__(self, in_dim, hidden_dim=32, symmetric=True):
        super().__init__()
        self.symmetric = symmetric
        self.proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, H_node):
        Z = self.proj(H_node)  # (B,N,D)
        scores = torch.matmul(Z, Z.transpose(1, 2)) / self.scale
        if self.symmetric:
            scores = 0.5 * (scores + scores.transpose(1, 2))
        A = torch.softmax(scores, dim=-1)
        return A


class ComplexLinear(nn.Module):
    """Complex linear layer with real parameters."""
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.Wr = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.Wi = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        if bias:
            self.br = nn.Parameter(torch.zeros(out_dim))
            self.bi = nn.Parameter(torch.zeros(out_dim))
        else:
            self.br = None
            self.bi = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.real, x.imag
        yr = xr @ self.Wr - xi @ self.Wi
        yi = xr @ self.Wi + xi @ self.Wr
        if self.br is not None:
            yr = yr + self.br
            yi = yi + self.bi
        return torch.complex(yr, yi)


class FourierGraphOperator(nn.Module):
    """
    n-invariant FGO parameterized by a complex dxd map.
    Input/Output in Fourier domain along node dimension.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.S = ComplexLinear(d_model, d_model, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(d_model))
        self.bias_i = nn.Parameter(torch.zeros(d_model))

    def forward(self, X_hat: torch.Tensor) -> torch.Tensor:
        Y_hat = self.S(X_hat) + torch.complex(self.bias_r, self.bias_i)
        return Y_hat


class FourierGNNEncoder(nn.Module):
    """
    Flatten window into hypervariate graph:
      n = L * N nodes, each node is (time, variate)
    Do FFT along node dimension and stack FGOs.
    """
    def __init__(self, in_dim: int, d_model: int = 128, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(in_dim, d_model)
        self.fgos = nn.ModuleList([FourierGraphOperator(d_model) for _ in range(n_layers)])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B,L,N,F)
        B, L, N, _ = x_seq.shape
        x = self.in_proj(x_seq)  # (B,L,N,d)

        # flatten to (B, n, d), n=L*N (time-major)
        x = x.contiguous().view(B, L * N, self.d_model)

        # FFT along node dimension n
        X_hat = torch.fft.fft(x.to(torch.float32), dim=1)  # complex

        out_time = 0.0
        Y_hat = None
        for k, fgo in enumerate(self.fgos):
            if k == 0:
                Y_hat = fgo(X_hat)
            else:
                Y_hat = fgo(Y_hat)
            y_time = torch.fft.ifft(Y_hat, dim=1).real
            y_time = self.act(y_time)
            y_time = self.drop(y_time)
            out_time = out_time + y_time

        out_time = out_time + self.res_scale * x  # residual
        return out_time.view(B, L, N, self.d_model)


class MechAware_GP_FourierGNN_MultiTask(nn.Module):
    """
    Learn-only model:
      x_seq -> FourierGNN -> take last step -> predict future return sequence y_seq (B,N,H)
    """
    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        horizon: int = 5,
        d_model: int = 128,
        n_layers: int = 3,
        graph_hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon

        self.graph_learner = LearnedGraphAttn(in_dim=in_dim, hidden_dim=graph_hidden, symmetric=True)
        self.encoder = FourierGNNEncoder(in_dim=in_dim, d_model=d_model, n_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(d_model, horizon)

    def forward(self, x_seq: torch.Tensor, A_mech=None):
        B, L, N, _ = x_seq.shape
        assert N == self.num_nodes

        # for logging / consistency
        H_node = x_seq.mean(dim=1)  # (B,N,F)
        A_learn = self.graph_learner(H_node)

        H = self.encoder(x_seq)      # (B,L,N,d)
        H_last = H[:, -1, :, :]      # (B,N,d)
        y_seq = self.fc_out(H_last)  # (B,N,H)

        gamma = torch.tensor(0.0, device=x_seq.device)  # learn-only
        return y_seq, A_learn, gamma


def seq_to_pf_ma_gap(y_seq: torch.Tensor):
    """
    y_seq: (B,N,H)
    PF = sum, MA = mean, GAP = max-min
    """
    pf = y_seq.sum(dim=-1)
    ma = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap


# =============== 4. Train / Eval ===============

def train_one_epoch_seq(model, loader, optimizer, device):
    model.train()
    mse = nn.MSELoss()

    total_loss = 0.0
    total_mae_pf = 0.0
    total_mae_ma = 0.0
    total_mae_gap = 0.0

    for x_seq, y_seq_true in loader:
        x_seq = x_seq.to(device)            # (B,L,N,F)
        y_seq_true = y_seq_true.to(device)  # (B,N,H)

        y_seq_pred, _, _ = model(x_seq, None)
        loss = mse(y_seq_pred, y_seq_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pf_p, ma_p, gap_p = seq_to_pf_ma_gap(y_seq_pred)
            pf_t, ma_t, gap_t = seq_to_pf_ma_gap(y_seq_true)
            total_mae_pf += torch.mean(torch.abs(pf_p - pf_t)).item()
            total_mae_ma += torch.mean(torch.abs(ma_p - ma_t)).item()
            total_mae_gap += torch.mean(torch.abs(gap_p - gap_t)).item()
            total_loss += loss.item()

    n = len(loader)
    return total_loss / n, total_mae_pf / n, total_mae_ma / n, total_mae_gap / n


@torch.no_grad()
def eval_one_epoch_seq(model, loader, device):
    model.eval()

    total_pf_mae = total_pf_mse = 0.0
    total_ma_mae = total_ma_mse = 0.0
    total_gap_mae = total_gap_mse = 0.0

    for x_seq, y_seq_true in loader:
        x_seq = x_seq.to(device)
        y_seq_true = y_seq_true.to(device)

        y_seq_pred, _, _ = model(x_seq, None)

        pf_p, ma_p, gap_p = seq_to_pf_ma_gap(y_seq_pred)
        pf_t, ma_t, gap_t = seq_to_pf_ma_gap(y_seq_true)

        total_pf_mae += torch.mean(torch.abs(pf_p - pf_t)).item()
        total_pf_mse += torch.mean((pf_p - pf_t) ** 2).item()

        total_ma_mae += torch.mean(torch.abs(ma_p - ma_t)).item()
        total_ma_mse += torch.mean((ma_p - ma_t) ** 2).item()

        total_gap_mae += torch.mean(torch.abs(gap_p - gap_t)).item()
        total_gap_mse += torch.mean((gap_p - gap_t) ** 2).item()

    n = len(loader)
    return {
        "PF_MAE": total_pf_mae / n,
        "PF_RMSE": math.sqrt(total_pf_mse / n),
        "MA_MAE": total_ma_mae / n,
        "MA_RMSE": math.sqrt(total_ma_mse / n),
        "GAP_MAE": total_gap_mae / n,
        "GAP_RMSE": math.sqrt(total_gap_mse / n),
    }



def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    price_df, returns_df, macro_df, price_cols, macro_cols = load_panel_from_two_files(
        args.price_path, args.macro_path
    )

    dataset = GPMultiTaskDataset(
        returns_df=returns_df,
        price_df=price_df,
        macro_df=macro_df,
        window_size=args.window,
        horizon=args.horizon,
    )

    n_total = len(dataset)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    idx = np.arange(n_total)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, drop_last=False)

    x0, _ = dataset[0]
    L, N, F_total = x0.shape
    print(f"[INFO] T,N,F = {dataset.T} {N} {F_total}")
    print(f"[INFO] train/val/test = {n_train}/{n_val}/{n_test}")

    model = MechAware_GP_FourierGNN_MultiTask(
        num_nodes=N,
        in_dim=F_total,
        horizon=args.horizon,
        d_model=args.d_model,
        n_layers=args.n_layers,
        graph_hidden=args.graph_hidden,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch_seq(model, train_loader, optimizer, device)
        va = eval_one_epoch_seq(model, val_loader, device)

        # # train (tuple)
        # print(
        #     f"[E{ep:03d}] train loss={tr[0]:.6f} | "
        #     f"PF_MAE={tr[1]:.6f} MA_MAE={tr[2]:.6f} GAP_MAE={tr[3]:.6f}"
        # )
        #
        # # val (dict)
        # print(
        #     f"         val  | "
        #     f"PF {va['PF_MAE']:.6f}/{va['PF_RMSE']:.6f} "
        #     f"MA {va['MA_MAE']:.6f}/{va['MA_RMSE']:.6f} "
        #     f"GAP {va['GAP_MAE']:.6f}/{va['GAP_RMSE']:.6f}"
        # )

        # early stopping criterion
        crit = va["PF_MAE"]

        if crit < best_val:
            best_val = crit
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


    # ===== after training: load best checkpoint =====
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== test =====
    te = eval_one_epoch_seq(model, test_loader, device)

    print(
        f"[TEST] "
        f"PF MAE={te['PF_MAE']:.6f} RMSE={te['PF_RMSE']:.6f} | "
        f"MA MAE={te['MA_MAE']:.6f} RMSE={te['MA_RMSE']:.6f} | "
        f"GAP MAE={te['GAP_MAE']:.6f} RMSE={te['GAP_RMSE']:.6f}"
    )



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--price_path", type=str, required=True)
    p.add_argument("--macro_path", type=str, required=True)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true")

    # FourierGNN params
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # learned-graph head (logging)
    p.add_argument("--graph_hidden", type=int, default=32)

    args = p.parse_args()
    main(args)
