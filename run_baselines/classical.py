import numpy as np
import pandas as pd
import math
import random
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings(
    "ignore",
    category=ValueWarning,
    message="A date index has been provided, but it has no associated frequency information"
)


# =============== 1. 工具函数 ===============

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_panel_from_two_files(price_path, macro_path):
    df_p = pd.read_csv(price_path)
    date_col_p = df_p.columns[0]
    df_p[date_col_p] = pd.to_datetime(df_p[date_col_p], dayfirst=True, errors="coerce")
    df_p = df_p.set_index(date_col_p).sort_index()

    # 价格列（资产）
    price_cols = [c for c in df_p.columns if np.issubdtype(df_p[c].dtype, np.number)]
    price_df = df_p[price_cols].astype("float32")
    price_df = price_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    # 宏观
    df_m = pd.read_csv(macro_path)
    date_col_m = df_m.columns[0]
    df_m[date_col_m] = pd.to_datetime(df_m[date_col_m], dayfirst=True, errors="coerce")
    df_m = df_m.set_index(date_col_m).sort_index()
    macro_cols = [c for c in df_m.columns if np.issubdtype(df_m[c].dtype, np.number)]
    macro_df = df_m[macro_cols].astype("float32")

    # 对齐时间
    common_idx = price_df.index.intersection(macro_df.index)
    price_df = price_df.loc[common_idx].sort_index()
    macro_df = macro_df.loc[common_idx].sort_index()

    # log return
    rets = np.log(price_df).diff().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(axis=0, how="all").fillna(0.0)
    returns_df = rets.astype("float32")

    price_df = price_df.loc[returns_df.index]
    macro_df = macro_df.loc[returns_df.index]

    return price_df, returns_df, macro_df, price_cols, macro_cols


# =============== 2. Dataset ===============

class TSMultiTaskDataset(Dataset):
    """
    每个样本 = 单一资产的一个时间窗口:
      x_seq: (L, D)  D = 1 (该资产的 return) + F_macro
      y_pf, y_ma, y_gap: 标量
    """
    def __init__(self, returns_df, price_df, macro_df, window_size=20, horizon=5):
        self.window_size = window_size
        self.horizon = horizon

        self.dates = returns_df.index
        self.T, self.N = returns_df.shape

        self.ret = returns_df.values.astype(np.float32)
        self.price = price_df.loc[self.dates].values.astype(np.float32)

        self.macro = None
        self.F_macro = 0
        if macro_df is not None and macro_df.shape[1] > 0:
            macro_arr = macro_df.loc[self.dates].values.astype(np.float32)
            # 简单前向填充一下
            macro_arr = (
                pd.DataFrame(macro_arr, index=self.dates)
                .ffill()
                .fillna(0.0)
                .values
            )
            self.macro = macro_arr.astype(np.float32)
            self.F_macro = self.macro.shape[1]

        # 所有 (asset_idx, t) 组合
        self.indices = []
        for i in range(self.N):
            for t in range(window_size - 1, self.T - horizon):
                self.indices.append((i, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        asset_idx, t = self.indices[idx]
        L = self.window_size
        h = self.horizon

        start = t - (L - 1)
        end = t + 1

        # 输入：该资产收益 + 宏观
        ret_win = self.ret[start:end, asset_idx]  # (L,)
        x_list = [ret_win[:, None]]

        if self.macro is not None:
            macro_win = self.macro[start:end, :]  # (L, F_macro)
            x_list.append(macro_win)

        x_seq = np.concatenate(x_list, axis=1).astype(np.float32)  # (L, D)

        # 未来 horizon 天收益
        t_start = t + 1
        t_end = t + 1 + h
        future_ret = self.ret[t_start:t_end, asset_idx]  # (h,)

        # PF: log(P_{t+h}) - log(P_t)
        p_t = self.price[t, asset_idx]
        p_th = self.price[t_end - 1, asset_idx]
        y_pf = math.log(max(p_th, 1e-8)) - math.log(max(p_t, 1e-8))

        # MA / GAP（按你的 PF/MA/GAP 定义）
        y_ma = float(future_ret.mean())
        y_gap = float(future_ret.max() - future_ret.min())

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_pf, dtype=torch.float32),
            torch.tensor(y_ma, dtype=torch.float32),
            torch.tensor(y_gap, dtype=torch.float32),
        )


def split_ts_dataset(returns_df, price_df, macro_df, window_size, horizon,
                     train_ratio=0.7, val_ratio=0.15):
    T = len(returns_df)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_ret = returns_df.iloc[:train_end]
    val_ret   = returns_df.iloc[train_end - window_size - 1:val_end]
    test_ret  = returns_df.iloc[val_end - window_size - 1:]

    train_price = price_df.loc[train_ret.index]
    val_price   = price_df.loc[val_ret.index]
    test_price  = price_df.loc[test_ret.index]

    train_macro = macro_df.loc[train_ret.index]
    val_macro   = macro_df.loc[val_ret.index]
    test_macro  = macro_df.loc[test_ret.index]

    train_ds = TSMultiTaskDataset(train_ret, train_price, train_macro, window_size, horizon)
    val_ds   = TSMultiTaskDataset(val_ret,   val_price,   val_macro,   window_size, horizon)
    test_ds  = TSMultiTaskDataset(test_ret,  test_price,  test_macro,  window_size, horizon)
    return train_ds, val_ds, test_ds


# =============== 3. 深度模型 baseline ===============

class LSTMHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        h_last = h[-1]
        y = self.fc(h_last)
        y_pf = y[:, 0]
        y_ma = y[:, 1]
        y_gap = y[:, 2]
        return y_pf, y_ma, y_gap


class GRUHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        out, h = self.gru(x)
        h_last = h[-1]
        y = self.fc(h_last)
        return y[:, 0], y[:, 1], y[:, 2]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k,
                              padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        # x: (B, C, L)
        out = self.conv(x)
        if self.down is not None:
            res = self.down(x)
        else:
            res = x
        out = self.act(self.bn(out))
        # 对齐长度
        if out.shape[-1] != res.shape[-1]:
            minL = min(out.shape[-1], res.shape[-1])
            out = out[..., -minL:]
            res = res[..., -minL:]
        return out + res


class TCNHead(nn.Module):
    def __init__(self, in_dim, channel=64, levels=3):
        super().__init__()
        layers = []
        C_in = in_dim
        for i in range(levels):
            dilation = 2 ** i
            layers.append(TCNBlock(C_in, channel, k=3, dilation=dilation))
            C_in = channel
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channel, 3)

    def forward(self, x):
        # x: (B,L,D) -> (B,D,L)
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h_last = h[..., -1]  # (B, C)
        y = self.fc(h_last)
        return y[:, 0], y[:, 1], y[:, 2]


class TransformerHead(nn.Module):
    def __init__(self, in_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x):
        # x: (B,L,D)
        h = self.input_proj(x)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        y = self.fc(h_last)
        return y[:, 0], y[:, 1], y[:, 2]


# =============== 4. 经典 AR / VAR 基线（statsmodels） ===============

def ar_var_baseline(returns_df, price_df, macro_df, window_size, horizon,
                    model_type="ar"):
    """
    使用训练集拟合 AR(p) 或 VAR(p)，在测试集上计算 PF/MA/GAP 的 MAE/RMSE。
    简化版：
      - AR: 每个资产单独拟合 AR(1) 在收益上，递推 horizon 步
      - VAR: 所有资产拟合 VAR(1)，递推 horizon 步
    """
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.api import VAR

    T = len(returns_df)
    train_end = int(T * 0.7)
    val_end = int(T * 0.85)

    test_ret = returns_df.iloc[val_end:]
    test_price = price_df.loc[test_ret.index]

    asset_cols = list(returns_df.columns)
    N = len(asset_cols)

    # 拟合
    if model_type == "ar":
        models = {}
        train_ret = returns_df.iloc[:train_end]
        for col in asset_cols:
            y = train_ret[col].dropna()
            if len(y) < 50:
                continue
            try:
                models[col] = AutoReg(y, lags=1, old_names=False).fit()
            except Exception:
                continue
    else:  # VAR
        train_ret = returns_df.iloc[:train_end]
        model_var = VAR(train_ret.dropna()).fit(maxlags=1)

    pf_true_all, pf_pred_all = [], []
    ma_true_all, ma_pred_all = [], []
    gap_true_all, gap_pred_all = [], []

    test_vals = test_ret.values
    price_vals = test_price.values
    idx_offset = returns_df.index.get_loc(test_ret.index[0])

    for t_local in range(window_size - 1, len(test_ret) - horizon):
        t_global = idx_offset + t_local

        fut_slice = slice(t_local + 1, t_local + 1 + horizon)
        future_ret = test_vals[fut_slice, :]  # (h,N)

        # True PF/MA/GAP
        p_t = price_vals[t_local, :]
        p_th = price_vals[t_local + horizon, :]
        pf_true = np.log(np.maximum(p_th, 1e-8)) - np.log(np.maximum(p_t, 1e-8))
        ma_true = future_ret.mean(axis=0)
        gap_true = future_ret.max(axis=0) - future_ret.min(axis=0)

        # 预测 horizon 步收益
        if model_type == "ar":
            r_hat_seq = np.zeros((horizon, N), dtype=float)
            for j, col in enumerate(asset_cols):
                model_j = models.get(col, None)
                if model_j is None:
                    r_hat_seq[:, j] = 0.0
                    continue
                full_series = returns_df[col].values[:t_global + 1]
                last_val = full_series[-1]
                # AR(1): r_{t+1} = c + phi * r_t
                params = model_j.params
                if "const" in params.index:
                    c = params["const"]
                    phi = params[1] if len(params) > 1 else 0.0
                else:
                    c = 0.0
                    phi = params[0]
                x = last_val
                for k in range(horizon):
                    x = c + phi * x
                    r_hat_seq[k, j] = x
        else:
            # VAR(1) 递推
            cur_hist = returns_df.iloc[:t_global + 1]
            last = cur_hist.values[-1, :]
            A = model_var.coefs[0]
            # 部分 statsmodels 版本 params 是 MultiIndex，这里简单兜底
            try:
                c = model_var.params.iloc[0].values
            except Exception:
                c = np.zeros(N)
            r_hat_seq = np.zeros((horizon, N), dtype=float)
            x = last
            for k in range(horizon):
                x = c + A @ x
                r_hat_seq[k, :] = x

        pf_pred = r_hat_seq.sum(axis=0)
        ma_pred = r_hat_seq.mean(axis=0)
        gap_pred = r_hat_seq.max(axis=0) - r_hat_seq.min(axis=0)

        pf_true_all.append(pf_true)
        pf_pred_all.append(pf_pred)
        ma_true_all.append(ma_true)
        ma_pred_all.append(ma_pred)
        gap_true_all.append(gap_true)
        gap_pred_all.append(gap_pred)

    def metrics(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        diff = y_true - y_pred
        mae = np.mean(np.abs(diff))
        rmse = math.sqrt(np.mean(diff ** 2))
        return mae, rmse

    pf_mae, pf_rmse = metrics(np.vstack(pf_true_all), np.vstack(pf_pred_all))
    ma_mae, ma_rmse = metrics(np.vstack(ma_true_all), np.vstack(ma_pred_all))
    gap_mae, gap_rmse = metrics(np.vstack(gap_true_all), np.vstack(gap_pred_all))

    print(f"[AR/VAR-{model_type}] PF  MAE={pf_mae:.6f} RMSE={pf_rmse:.6f}")
    print(f"[AR/VAR-{model_type}] MA  MAE={ma_mae:.6f} RMSE={ma_rmse:.6f}")
    print(f"[AR/VAR-{model_type}] GAP MAE={gap_mae:.6f} RMSE={gap_rmse:.6f}")


# =============== 5. 训练 / 评估循环（深度模型） ===============

def train_one_epoch(model, loader, device, optimizer):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_mae_pf = 0.0
    total_mae_ma = 0.0
    total_mae_gap = 0.0

    for x_seq, y_pf, y_ma, y_gap in loader:
        x_seq = x_seq.to(device)
        y_pf = y_pf.to(device)
        y_ma = y_ma.to(device)
        y_gap = y_gap.to(device)

        optimizer.zero_grad()
        y_pf_pred, y_ma_pred, y_gap_pred = model(x_seq)

        loss_pf = mse(y_pf_pred, y_pf)
        loss_ma = mse(y_ma_pred, y_ma)
        loss_gap = mse(y_gap_pred, y_gap)
        loss = loss_pf + loss_ma + loss_gap
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        B = x_seq.size(0)
        total_loss += loss.item() * B
        total_mae_pf += (y_pf_pred - y_pf).abs().mean().item() * B
        total_mae_ma += (y_ma_pred - y_ma).abs().mean().item() * B
        total_mae_gap += (y_gap_pred - y_gap).abs().mean().item() * B

    n = len(loader.dataset)
    return (
        total_loss / n,
        total_mae_pf / n,
        total_mae_ma / n,
        total_mae_gap / n,
    )


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_mae_pf = 0.0
    total_mae_ma = 0.0
    total_mae_gap = 0.0

    for x_seq, y_pf, y_ma, y_gap in loader:
        x_seq = x_seq.to(device)
        y_pf = y_pf.to(device)
        y_ma = y_ma.to(device)
        y_gap = y_gap.to(device)

        y_pf_pred, y_ma_pred, y_gap_pred = model(x_seq)

        loss_pf = mse(y_pf_pred, y_pf)
        loss_ma = mse(y_ma_pred, y_ma)
        loss_gap = mse(y_gap_pred, y_gap)
        loss = loss_pf + loss_ma + loss_gap

        B = x_seq.size(0)
        total_loss += loss.item() * B
        total_mae_pf += (y_pf_pred - y_pf).abs().mean().item() * B
        total_mae_ma += (y_ma_pred - y_ma).abs().mean().item() * B
        total_mae_gap += (y_gap_pred - y_gap).abs().mean().item() * B

    n = len(loader.dataset)
    return (
        total_loss / n,
        total_mae_pf / n,
        total_mae_ma / n,
        total_mae_gap / n,
    )

def compute_mae_rmse(y_true_list, y_pred_list):
    y_true = np.concatenate([np.asarray(a).reshape(-1) for a in y_true_list], axis=0)
    y_pred = np.concatenate([np.asarray(a).reshape(-1) for a in y_pred_list], axis=0)
    diff = y_true - y_pred
    mae = np.mean(np.abs(diff))
    rmse = math.sqrt(np.mean(diff ** 2))
    return mae, rmse


# =============== 6. 主函数 ===============

def main(args):
    set_seed(args.seed)

    price_df, returns_df, macro_df, node_cols, macro_cols = load_panel_from_two_files(
        args.panel_prices, args.panel_macro
    )
    print(f"[INFO] Assets used: {node_cols}")
    print(f"[INFO] Macro features: {macro_cols}")

    # 经典 AR / VAR
    if args.model_type in ["ar", "var"]:
        ar_var_baseline(
            returns_df, price_df, macro_df,
            window_size=args.window_size,
            horizon=args.horizon,
            model_type=args.model_type,
        )
        return

    # 深度模型
    train_ds, val_ds, test_ds = split_ts_dataset(
        returns_df, price_df, macro_df,
        window_size=args.window_size,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sample_x, _, _, _ = train_ds[0]
    in_dim = sample_x.shape[-1]
    print(f"[INFO] Input dim = {in_dim}")

    if args.model_type == "lstm":
        model = LSTMHead(in_dim, hidden_dim=args.hidden_dim)
    elif args.model_type == "gru":
        model = GRUHead(in_dim, hidden_dim=args.hidden_dim)
    elif args.model_type == "tcn":
        model = TCNHead(in_dim, channel=args.hidden_dim, levels=3)
    elif args.model_type == "transformer":
        model = TransformerHead(in_dim, d_model=args.hidden_dim, nhead=4, num_layers=2)
    else:
        raise ValueError(f"Unknown model_type {args.model_type}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae_pf, train_mae_ma, train_mae_gap = train_one_epoch(
            model, train_loader, device, optimizer
        )
        val_loss, val_mae_pf, val_mae_ma, val_mae_gap = eval_one_epoch(model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:03d}] "
                f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
                f"ValMAE(pf/ma/gap)={val_mae_pf:.4f}/{val_mae_ma:.4f}/{val_mae_gap:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ====== TEST（带 MAE + RMSE） ======
    test_loss, test_mae_pf, test_mae_ma, test_mae_gap = eval_one_epoch(model, test_loader, device)
    print(f"[TEST] TotalLoss={test_loss:.6f} | "
          f"MAE(pf/ma/gap)={test_mae_pf:.6f}/{test_mae_ma:.6f}/{test_mae_gap:.6f}")

    # 为 RMSE 做累积
    pf_true_all, pf_pred_all = [], []
    ma_true_all, ma_pred_all = [], []
    gap_true_all, gap_pred_all = [], []

    for x_seq, y_pf, y_ma, y_gap in test_loader:
        x_seq = x_seq.to(device)
        y_pf_pred, y_ma_pred, y_gap_pred = model(x_seq)

        pf_true_all.append(y_pf.numpy())
        pf_pred_all.append(y_pf_pred.detach().cpu().numpy())


        ma_true_all.append(y_ma.numpy())
        ma_pred_all.append(y_ma_pred.detach().cpu().numpy())

        gap_true_all.append(y_gap.numpy())
        gap_pred_all.append(y_gap_pred.detach().cpu().numpy())

    # 计算 RMSE
    pf_mae, pf_rmse = compute_mae_rmse(pf_true_all, pf_pred_all)
    ma_mae, ma_rmse = compute_mae_rmse(ma_true_all, ma_pred_all)
    gap_mae, gap_rmse = compute_mae_rmse(gap_true_all, gap_pred_all)

    print(f"[METRIC] PF:  MAE={pf_mae:.6f} RMSE={pf_rmse:.6f}")
    print(f"[METRIC] MA:  MAE={ma_mae:.6f} RMSE={ma_rmse:.6f}")
    print(f"[METRIC] GAP: MAE={gap_mae:.6f} RMSE={gap_rmse:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_prices", type=str, default="panel_40.csv")
    ap.add_argument("--panel_macro", type=str, default="panel_macro.csv")
    ap.add_argument("--model_type", type=str,
                    choices=["ar", "var", "lstm", "gru", "tcn", "transformer"],
                    default="lstm")

    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden_dim", type=int, default=64)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")

    args, unknown = ap.parse_known_args()
    main(args)
