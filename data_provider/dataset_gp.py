"""Datasets for multi-task commodity forecasting (PF/MA/GAP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class GPFeatureConfig:
    """Feature switches / hyperparameters.

    This keeps the dataset logic stable while making ablations easy.
    """

    # rolling windows
    ret_windows: tuple[int, ...] = (1, 5, 20)
    vol_windows: tuple[int, ...] = (5, 10, 20)
    price_z_window: int = 20
    mom_lag: int = 5


class GPMultiTaskDataset(Dataset):
    """Multi-task dataset.

    Input
    -----
    x_seq: (L, N, F_total)

    Label
    -----
    y_seq: (N, H) future return sequence.

    Notes
    -----
    Your downstream `seq_to_pf_ma_gap` can map y_seq into:
      - PF  : sum over horizon
      - MA  : mean over horizon
      - GAP : max-min over horizon
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        price_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame],
        window_size: int = 20,
        horizon: int = 5,
        feature_cfg: Optional[GPFeatureConfig] = None,
        normalize_within_window: bool = True,
    ):
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.feature_cfg = feature_cfg or GPFeatureConfig()
        self.normalize_within_window = bool(normalize_within_window)

        # ---- raw aligned arrays ----
        self.dates = returns_df.index
        self.T, self.N = returns_df.shape

        self.ret = returns_df.values.astype(np.float32)  # (T,N)
        self.price = price_df.loc[self.dates].values.astype(np.float32)  # (T,N)

        # macro factors
        self.macro = None
        self.F_macro = 0
        if macro_df is not None and macro_df.shape[1] > 0:
            macro_arr = macro_df.loc[self.dates].values.astype(np.float32)
            macro_arr = pd.DataFrame(macro_arr, index=self.dates).ffill().fillna(0.0).values
            self.macro = macro_arr.astype(np.float32)  # (T,F_macro)
            self.F_macro = self.macro.shape[1]

        # labels use raw returns
        self.raw_ret = self.ret

        # valid indices: [window_size-1, T-horizon-1]
        self.valid_idx = list(range(self.window_size - 1, self.T - self.horizon))

    def __len__(self) -> int:
        return len(self.valid_idx)

    @staticmethod
    def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
        """Rolling mean on axis=0 for a window array shaped (L, N)."""
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

    @staticmethod
    def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
        """Rolling std on axis=0 for a window array shaped (L, N)."""
        L, N = x.shape
        out = np.empty_like(x, dtype=np.float32)
        for i in range(L):
            start = max(0, i - win + 1)
            seg = x[start : i + 1]
            out[i] = seg.std(axis=0)
        return out

    def __getitem__(self, idx: int):
        t = self.valid_idx[idx]
        L = self.window_size
        h = self.horizon

        start = t - (L - 1)
        end = t + 1

        # window returns / prices
        ret_win = self.ret[start:end, :]      # (L,N)
        price_win = self.price[start:end, :]  # (L,N)

        cfg = self.feature_cfg

        # --- node features ---
        node_feat_list = []

        # returns (mean-returns with different windows)
        # keep ret_1 as raw
        for w in cfg.ret_windows:
            if w == 1:
                node_feat_list.append(ret_win)
            else:
                node_feat_list.append(self._rolling_mean(ret_win, w))

        # vol
        for w in cfg.vol_windows:
            node_feat_list.append(self._rolling_std(ret_win, w))

        # momentum (price difference)
        mom_lag = int(cfg.mom_lag)
        feat_mom = price_win - np.roll(price_win, mom_lag, axis=0)
        for i in range(min(mom_lag, L)):
            feat_mom[i] = price_win[i] - price_win[0]
        node_feat_list.append(feat_mom)

        # z-score of price vs rolling mean/std
        z_win = int(cfg.price_z_window)
        ma_price = self._rolling_mean(price_win, z_win)
        std_price = self._rolling_std(price_win, z_win)
        feat_z = (price_win - ma_price) / (std_price + 1e-8)
        node_feat_list.append(feat_z)

        node_feat = np.stack(node_feat_list, axis=-1).astype(np.float32)  # (L,N,F_node)
        node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # local normalization (per-window, across all nodes)
        if self.normalize_within_window:
            L_, N_, F_node = node_feat.shape
            flat = node_feat.reshape(L_ * N_, F_node)
            mean = flat.mean(axis=0, keepdims=True)
            std = flat.std(axis=0, keepdims=True) + 1e-8
            flat = (flat - mean) / std
            node_feat = flat.reshape(L_, N_, F_node).astype(np.float32)

        # macro broadcast (L,N,F_macro)
        if self.macro is not None:
            macro_win = self.macro[start:end, :]          # (L,F_macro)
            macro_win = macro_win[:, None, :]             # (L,1,F_macro)
            macro_win = np.repeat(macro_win, self.N, axis=1)
            x_seq = np.concatenate([node_feat, macro_win], axis=-1)
        else:
            x_seq = node_feat

        x_seq = torch.tensor(x_seq, dtype=torch.float32)  # (L,N,F_total)

        # --- labels: future returns sequence ---
        t_start = t + 1
        t_end = t + 1 + h
        future_ret = self.raw_ret[t_start:t_end, :]  # (H,N)
        y_seq = torch.tensor(future_ret.T, dtype=torch.float32)  # (N,H)

        return x_seq, y_seq
