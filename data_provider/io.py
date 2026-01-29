"""I/O helpers for panel-style commodity data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_panel_from_two_files(price_path: str, macro_path: str):
    """Load and align price panel + macro panel from two CSV files.

    Expected CSV format:
      - First column is a date column.
      - Remaining columns are features.

    Returns
    -------
    price_df : pd.DataFrame
        (T, N) aligned price panel.
    returns_df : pd.DataFrame
        (T, N) aligned log-returns panel.
    macro_df : pd.DataFrame
        (T, F_macro) aligned macro factors.
    price_cols : list[str]
    macro_cols : list[str]
    """
    # 1) prices
    df_p = pd.read_csv(price_path)
    date_col_p = df_p.columns[0]
    df_p[date_col_p] = pd.to_datetime(df_p[date_col_p], dayfirst=True, errors="coerce")
    df_p = df_p.set_index(date_col_p).sort_index()

    price_cols = list(df_p.columns)
    price_df = df_p[price_cols].astype("float32")
    price_df = price_df.replace([np.inf, -np.inf], np.nan)
    price_df = price_df.ffill().bfill().fillna(0.0)

    # 2) macro
    df_m = pd.read_csv(macro_path)
    date_col_m = df_m.columns[0]
    df_m[date_col_m] = pd.to_datetime(df_m[date_col_m], dayfirst=True, errors="coerce")
    df_m = df_m.set_index(date_col_m).sort_index()

    macro_cols = list(df_m.columns)
    macro_df = df_m[macro_cols].astype("float32")

    # 3) align on common dates
    common_idx = price_df.index.intersection(macro_df.index)
    price_df = price_df.loc[common_idx].sort_index()
    macro_df = macro_df.loc[common_idx].sort_index()

    # 4) log returns
    rets = np.log(price_df).diff().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(axis=0, how="all").fillna(0.0)
    returns_df = rets.astype("float32")

    # re-align price and macro to returns index
    price_df = price_df.loc[returns_df.index]
    macro_df = macro_df.loc[returns_df.index]

    return price_df, returns_df, macro_df, price_cols, macro_cols
