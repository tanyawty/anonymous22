# import argparse
# from pathlib import Path
# import itertools
# import math
# import warnings
# import numpy as np
# import pandas as pd
#
# warnings.filterwarnings("ignore")
#
# # ---------- 你这份 panel_prices <-> metadata.symbol 的固定映射 ----------
# # metadata.symbol -> panel_prices 列名
# SYMBOL_TO_PRICECOL = {
#     # Energy
#     "CL":"wti",
#     "BZ":"brent",
#     "NG":"natgas",
#     "RB":"rbo_gas",
#     "HO":"heating_oil",
#     # Precious / Metals
#     "GC":"gold",
#     "SI":"silver",
#     "PL":"platinum",
#     "PA":"palladium",
#     "HG":"copper",
#     # Agriculture
#     "ZC":"corn",
#     "ZS":"soy",
#     "ZW":"wheat",     # 你的价格表只有 wheat（SRW/HRW 未区分）
#     "KE":"wheat",
#     "ZL":"soy_oil",
#     "ZM":"soy_meal",
#     "SB":"sugar",
#     "KC":"coffee",
#     "CC":"cocoa",
#     "CT":"cotton",
#     "OJ":"oj",
#     # Livestock
#     "LE":"live_cattle",
#     "HE":"lean_hogs",
#     # 你面板当前没有：铝(AL/ALI)、木材(LBR)等，如后续加列再扩展映射
# }
#
# # 宏观列名（来自你的 panel_macro）
# DEFAULT_MACRO_COLS = ["usd_index", "cpi", "indpro", "fedfunds", "gs10"]
#
#
# # ---------- 通用小工具 ----------
# def _to_returns(df):
#     df = df.sort_index()
#     return np.log(df).diff()
#
# def _minmax(x):
#     x = np.asarray(x, dtype=float)
#     lo = np.nanmin(x)
#     hi = np.nanmax(x)
#     if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
#         return np.zeros_like(x)
#     return (x - lo) / (hi - lo)
#
# # ---------- 信号分数 ----------
# def signal_crosscorr(x, y, max_lag=10):
#     best_score, best_lag = np.nan, None
#     for lag in range(1, max_lag + 1):
#         c = x.corr(y.shift(-lag))
#         if c is not None and not np.isnan(c):
#             s = abs(c)
#             if (isinstance(best_score, float) and np.isnan(best_score)) or (s > best_score):
#                 best_score, best_lag = s, lag
#     return (best_score, best_lag)
#
# def signal_LP(x, y, horizons=(1,5,10), controls=None):
#     import statsmodels.api as sm
#     df = pd.concat({"X": x, "Y": y}, axis=1)
#     if controls is not None and len(controls.columns) > 0:
#         for c in controls.columns:
#             df[c] = controls[c]
#     df = df.dropna()
#     tvals = []
#     for h in horizons:
#         yy = df["Y"].shift(-h) - df["Y"]
#         X = df[["X"]]
#         if controls is not None and len(controls.columns) > 0:
#             X = pd.concat([X, df[controls.columns]], axis=1)
#         X = sm.add_constant(X, has_constant="add")
#         tmp = pd.concat([yy, X], axis=1).dropna()
#         if len(tmp) < 50:
#             continue
#         model = sm.OLS(tmp.iloc[:,0], tmp.iloc[:,1:]).fit()
#         if "X" in model.params.index:
#             tvals.append(abs(model.tvalues["X"]))
#     return np.nanmax(tvals) if len(tvals) else np.nan
#
# def signal_cointegration_ecm(x_level, y_level):
#     from statsmodels.tsa.stattools import coint
#     xy = pd.concat([x_level, y_level], axis=1).dropna()
#     if len(xy) < 80:
#         return np.nan
#     try:
#         stat, pval, _ = coint(xy.iloc[:,0], xy.iloc[:,1])
#         if np.isnan(pval): return np.nan
#         return float(np.clip(1.0 / max(pval, 1e-6), 0, 20))
#     except Exception:
#         return np.nan
#
# def signal_granger(x, y, maxlag=5):
#     from statsmodels.tsa.stattools import grangercausalitytests
#     xy = pd.concat([y, x], axis=1)
#     xy.columns = ["y","x"]
#     xy = xy.dropna()
#     if len(xy) < 100:
#         return np.nan
#     try:
#         res = grangercausalitytests(xy, maxlag=maxlag, verbose=False)
#         best_p = 1.0
#         for _, out in res.items():
#             p = out[0]["ssr_ftest"][1]
#             best_p = min(best_p, p)
#         return -math.log(max(best_p, 1e-8))
#     except Exception:
#         return np.nan
#
#
# # ---------- 模板生成（按你面板列名） ----------
# def gen_T2_chain(meta_mapped):
#     edges = []
#     for _, r in meta_mapped.iterrows():
#         up_sym = r.get("main_output_of", "")
#         if isinstance(up_sym, str) and up_sym.strip() != "":
#             src = r.get("price_col_upstream", "")
#             dst = r.get("price_col", "")
#             if src and dst:
#                 edges.append(dict(source=src, target=dst, template="T2_chain",
#                                   rule=f"{up_sym}->{r['symbol']}", lag_hint=5,
#                                   notes=f"upstream_to_downstream ({up_sym}->{r['symbol']})"))
#     return edges
#
# def gen_T3_substitution(meta_mapped):
#     edges = []
#     groups = meta_mapped["substitution_group"].dropna().unique().tolist()
#     for g in groups:
#         members = meta_mapped.loc[meta_mapped["substitution_group"]==g, "price_col"].dropna().tolist()
#         for a,b in itertools.permutations(members, 2):
#             if a==b: continue
#             edges.append(dict(source=a, target=b, template="T3_substitution",
#                               rule=f"group:{g}", lag_hint=3,
#                               notes=f"substitution/compl. group={g}"))
#     return edges
#
# def gen_T4_macro(meta_mapped, macro_cols):
#     edges = []
#     for _, r in meta_mapped.iterrows():
#         if str(r.get("macro_sensitive","False")).lower()!="true":
#             continue
#         dst = r.get("price_col","")
#         if not dst:
#             continue
#         for m in macro_cols:
#             edges.append(dict(source=m, target=dst, template="T4_macro",
#                               rule="macro_to_commodity", lag_hint=5,
#                               notes=f"{m}->{dst}"))
#     return edges
#
# def gen_T5_T7_parity(meta_mapped):
#     edges = []
#     if "parity_group" not in meta_mapped.columns:
#         return edges
#     for g in meta_mapped["parity_group"].dropna().unique().tolist():
#         members = meta_mapped.loc[meta_mapped["parity_group"]==g, "price_col"].dropna().tolist()
#         if len(members)<2:
#             continue
#         for a,b in itertools.permutations(members,2):
#             if a==b: continue
#             edges.append(dict(source=a, target=b, template="T5_T7_parity",
#                               rule=f"parity:{g}", lag_hint=2,
#                               notes=f"price parity/arbitrage"))
#     return edges
#
# def gen_T1_termstructure(meta_mapped, price_cols):
#     edges = []
#     for _, r in meta_mapped.iterrows():
#         if str(r.get("storability","No")).lower()!="yes":
#             continue
#         symcol = r.get("price_col","")
#         if symcol in price_cols:
#             edges.append(dict(source=symcol, target=symcol, template="T1_term",
#                               rule="intra_maturity_placeholder", lag_hint=1,
#                               notes="needs multi-maturity series"))
#     return edges
#
#
# # ---------- 权重融合 ----------
# def compute_edge_scores(edges_df, prices_level, prices_rets, macro_rets=None,
#                         alpha=(0.4,0.2,0.2,0.2),
#                         max_lag_corr=10, horizons=(1,5,10), granger_lag=5):
#     s_corr_list, s_LP_list, s_co_list, s_gr_list, lag_best_list = [], [], [], [], []
#
#     price_cols = set(prices_rets.columns)
#     macro_cols  = set(macro_rets.columns) if macro_rets is not None else set()
#
#     for _, e in edges_df.iterrows():
#         src, dst = e["source"], e["target"]
#
#         # X: 可能是宏观或价格收益
#         if src in price_cols:
#             x = prices_rets[src]
#         elif src in macro_cols:
#             x = macro_rets[src]
#         else:
#             s_corr_list += [np.nan]; s_LP_list += [np.nan]
#             s_co_list   += [np.nan]; s_gr_list += [np.nan]
#             lag_best_list += [np.nan]
#             continue
#
#         # Y: 目标必须是价格收益
#         if dst not in price_cols:
#             s_corr_list += [np.nan]; s_LP_list += [np.nan]
#             s_co_list   += [np.nan]; s_gr_list += [np.nan]
#             lag_best_list += [np.nan]
#             continue
#         y = prices_rets[dst]
#
#         sc, lagb = signal_crosscorr(x, y, max_lag=max_lag_corr)
#         slp = signal_LP(x, y, horizons=horizons, controls=macro_rets)
#
#         # 协整只在价格-价格对上才计算，用水平价
#         if (src in price_cols) and (dst in price_cols):
#             sco = signal_cointegration_ecm(prices_level[src], prices_level[dst])
#         else:
#             sco = np.nan
#
#         sgr = signal_granger(x, y, maxlag=granger_lag)
#
#         s_corr_list.append(sc); s_LP_list.append(slp)
#         s_co_list.append(sco);  s_gr_list.append(sgr)
#         lag_best_list.append(lagb)
#
#     edges_df["s_corr"] = s_corr_list
#     edges_df["s_LP"]   = s_LP_list
#     edges_df["s_co"]   = s_co_list
#     edges_df["s_gr"]   = s_gr_list
#     edges_df["lag_hint_real"] = lag_best_list
#
#     # 按模板 0-1 归一化后线性融合
#     allw = []
#     for tmpl, sub in edges_df.groupby("template"):
#         idx = sub.index
#         s1 = _minmax(edges_df.loc[idx,"s_LP"].values)
#         s2 = _minmax(edges_df.loc[idx,"s_co"].values)
#         s3 = _minmax(edges_df.loc[idx,"s_gr"].values)
#         s4 = _minmax(edges_df.loc[idx,"s_corr"].values)
#         w = alpha[0]*s1 + alpha[1]*s2 + alpha[2]*s3 + alpha[3]*s4
#         allw.append(pd.Series(w, index=idx))
#     edges_df["w"] = pd.concat(allw).sort_index()
#     return edges_df
#
#
# # ---------- 主流程 ----------
# def main(args):
#     # 1) 读数据
#     meta = pd.read_csv(args.metadata)
#     prices = pd.read_csv(args.prices)
#     macro  = pd.read_csv(args.macro)
#
#     # 统一索引为日期
#     prices["date"] = pd.to_datetime(prices["date"])
#     macro["date"]  = pd.to_datetime(macro["date"])
#     prices = prices.set_index("date").sort_index()
#     macro  = macro.set_index("date").sort_index()
#
#     # 2) 把 metadata.symbol 映射到价格列（新增 price_col / price_col_upstream）
#     meta["price_col"] = meta["symbol"].map(SYMBOL_TO_PRICECOL)
#     meta["price_col_upstream"] = meta["main_output_of"].map(SYMBOL_TO_PRICECOL) if "main_output_of" in meta.columns else None
#
#     # 把没有映射成功或价格表不存在的品种过滤掉（发出提示）
#     available_cols = set(prices.columns)
#     ok = meta["price_col"].isin(available_cols)
#     if (~ok).any():
#         miss = meta.loc[~ok, ["symbol","price_col"]]
#         print("[WARN] 以下 symbol 在 panel_prices 中没有匹配列，将跳过：")
#         print(miss.to_string(index=False))
#     meta_mapped = meta.loc[ok].copy()
#
#     # 3) 候选边（T1/T2/T3/T4/T5/T7）
#     edges = []
#     edges += gen_T2_chain(meta_mapped)
#     edges += gen_T3_substitution(meta_mapped)
#     edges += gen_T4_macro(meta_mapped, [c for c in DEFAULT_MACRO_COLS if c in macro.columns])
#     edges += gen_T5_T7_parity(meta_mapped)
#     edges += gen_T1_termstructure(meta_mapped, prices.columns)
#
#     edges_df = pd.DataFrame(edges)
#     if edges_df.empty:
#         print("[WARN] 未生成候选边，请检查 metadata 字段与映射。")
#         return
#
#     # 4) 计算权重（可选）
#     if args.compute_weights:
#         prices_rets = _to_returns(prices)
#         macro_rets  = _to_returns(macro[[c for c in DEFAULT_MACRO_COLS if c in macro.columns]])
#         edges_df = compute_edge_scores(
#             edges_df,
#             prices_level=prices,
#             prices_rets=prices_rets,
#             macro_rets=macro_rets,
#             alpha=(args.alpha1, args.alpha2, args.alpha3, args.alpha4),
#             max_lag_corr=args.max_lag_corr,
#             horizons=tuple(args.horizons),
#             granger_lag=args.granger_lag
#         )
#
#     # 5) 输出
#     out_cols = ["source","target","template","rule","lag_hint","lag_hint_real","notes"]
#     if args.compute_weights:
#         out_cols += ["w","s_LP","s_co","s_gr","s_corr"]
#     for c in out_cols:
#         if c not in edges_df.columns:
#             edges_df[c] = np.nan
#     edges_df[out_cols].to_csv(args.out_edges, index=False)
#     print(f"[OK] candidates -> {args.out_edges} (rows={len(edges_df)})")
#
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--metadata", default="metadata.csv")
#     ap.add_argument("--prices",   default="panel_prices.csv")
#     ap.add_argument("--macro",    default="panel_macro.csv")
#     ap.add_argument("--out_edges", default="edges_candidates.csv")
#     ap.add_argument("--compute_weights", action="store_true")
#
#     # 权重超参
#     ap.add_argument("--alpha1", type=float, default=0.4)  # s_LP
#     ap.add_argument("--alpha2", type=float, default=0.2)  # s_co
#     ap.add_argument("--alpha3", type=float, default=0.2)  # s_gr
#     ap.add_argument("--alpha4", type=float, default=0.2)  # s_corr
#     ap.add_argument("--max_lag_corr", type=int, default=10)
#     ap.add_argument("--horizons", nargs="+", type=int, default=[1,5,10])
#     ap.add_argument("--granger_lag", type=int, default=5)
#
#     # 关键一行：parse_known_args，而不是 parse_args
#     args, unknown = ap.parse_known_args()
#     # 你也可以 print 一下看看是啥：
#     # print("Unknown args:", unknown)
#
#     main(args)

import argparse
from pathlib import Path
import itertools
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================
# 1. symbol -> panel_prices 列名 的映射（按你的面板来）
# =====================================================

SYMBOL_TO_PRICECOL = {
    # Energy
    "CL": "wti",
    "BZ": "brent",
    "NG": "natgas",
    "RB": "rbo_gas",
    "HO": "heating_oil",

    # Precious / Metals
    "GC": "gold",
    "SI": "silver",
    "PL": "platinum",
    "PA": "palladium",
    "HG": "copper",

    # Agriculture
    "ZC": "corn",
    "ZS": "soy",
    "ZW": "wheat",
    "KE": "wheat",
    "ZL": "soy_oil",
    "ZM": "soy_meal",
    "SB": "sugar",
    "KC": "coffee",
    "CC": "cocoa",
    "CT": "cotton",
    "OJ": "oj",

    # Livestock
    "LE": "live_cattle",
    "HE": "lean_hogs",
}

DEFAULT_MACRO_COLS = ["usd_index", "cpi", "indpro", "fedfunds", "gs10"]


# =====================================================
# 2. 工具函数
# =====================================================

def _to_returns(df):
    df = df.sort_index()
    return np.log(df).diff()


def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# =====================================================
# 3. 信号分数计算
# =====================================================

def signal_crosscorr(x, y, max_lag=10):
    best_score, best_lag = np.nan, None
    for lag in range(1, max_lag + 1):
        c = x.corr(y.shift(-lag))
        if c is not None and not np.isnan(c):
            s = abs(c)
            if np.isnan(best_score) or s > best_score:
                best_score, best_lag = s, lag
    return best_score, best_lag


def signal_LP(x, y, horizons=(1, 5, 10), controls=None):
    import statsmodels.api as sm
    df = pd.concat({"X": x, "Y": y}, axis=1)
    if controls is not None:
        for c in controls.columns:
            df[c] = controls[c]
    df = df.dropna()

    tvals = []
    for h in horizons:
        yy = df["Y"].shift(-h) - df["Y"]
        X = df[["X"]]
        if controls is not None:
            X = pd.concat([X, df[controls.columns]], axis=1)
        X = sm.add_constant(X, has_constant="add")
        tmp = pd.concat([yy, X], axis=1).dropna()
        if len(tmp) < 50:
            continue
        model = sm.OLS(tmp.iloc[:, 0], tmp.iloc[:, 1:]).fit()
        if "X" in model.params.index:
            tvals.append(abs(model.tvalues["X"]))

    return np.nanmax(tvals) if len(tvals) else np.nan


def signal_cointegration_ecm(x_level, y_level):
    from statsmodels.tsa.stattools import coint
    xy = pd.concat([x_level, y_level], axis=1).dropna()
    if len(xy) < 80:
        return np.nan
    try:
        stat, pval, _ = coint(xy.iloc[:, 0], xy.iloc[:, 1])
        if np.isnan(pval):
            return np.nan
        return float(np.clip(1.0 / max(pval, 1e-6), 0, 20))
    except Exception:
        return np.nan


def signal_granger(x, y, maxlag=5):
    from statsmodels.tsa.stattools import grangercausalitytests
    xy = pd.concat([y, x], axis=1)
    xy.columns = ["y", "x"]
    xy = xy.dropna()
    if len(xy) < 100:
        return np.nan
    try:
        res = grangercausalitytests(xy, maxlag=maxlag, verbose=False)
        best_p = min(out[0]["ssr_ftest"][1] for _, out in res.items())
        return -math.log(max(best_p, 1e-8))
    except Exception:
        return np.nan


# =====================================================
# 4. 模板生成候选边 (T1/T2/T3/T4/T5/T7)
# =====================================================

def gen_T2_chain(meta):
    edges = []
    for _, r in meta.iterrows():
        up = r.get("main_output_of", "")
        if isinstance(up, str) and up.strip() != "":
            src = r.get("price_col_upstream", "")
            dst = r.get("price_col", "")
            if src and dst:
                edges.append(dict(
                    source=src, target=dst,
                    template="T2_chain",
                    rule=f"{up}->{r['symbol']}",
                    lag_hint=5,
                    notes=f"upstream_to_downstream ({up}->{r['symbol']})"
                ))
    return edges


def gen_T3_substitution(meta):
    edges = []
    groups = meta["substitution_group"].dropna().unique().tolist()
    for g in groups:
        members = meta.loc[meta["substitution_group"] == g, "price_col"].dropna().tolist()
        for a, b in itertools.permutations(members, 2):
            if a != b:
                edges.append(dict(
                    source=a, target=b,
                    template="T3_substitution",
                    rule=f"group:{g}",
                    lag_hint=3,
                    notes=f"substitution/compl. {g}"
                ))
    return edges


def gen_T4_macro(meta, macro_cols):
    edges = []
    for _, r in meta.iterrows():
        if str(r.get("macro_sensitive", "False")).lower() == "true":
            dst = r.get("price_col", "")
            for m in macro_cols:
                edges.append(dict(
                    source=m, target=dst,
                    template="T4_macro",
                    rule="macro_to_commodity",
                    lag_hint=5,
                    notes=f"{m}->{dst}"
                ))
    return edges


def gen_T5_T7_parity(meta):
    edges = []
    if "parity_group" not in meta.columns:
        return edges
    for g in meta["parity_group"].dropna().unique():
        members = meta.loc[meta["parity_group"] == g, "price_col"].dropna().tolist()
        for a, b in itertools.permutations(members, 2):
            if a != b:
                edges.append(dict(
                    source=a, target=b,
                    template="T5_T7_parity",
                    rule=f"parity:{g}",
                    lag_hint=2,
                    notes="price parity/arbitrage"
                ))
    return edges


def gen_T1_termstructure(meta, price_cols):
    edges = []
    for _, r in meta.iterrows():
        if str(r.get("storability", "No")).lower() == "yes":
            col = r.get("price_col", "")
            if col in price_cols:
                edges.append(dict(
                    source=col, target=col,
                    template="T1_term",
                    rule="intra_maturity_placeholder",
                    lag_hint=1,
                    notes="needs multi-maturity series"
                ))
    return edges


# =====================================================
# 5. 权重融合
# =====================================================

def compute_edge_scores(edges_df, prices_level, prices_rets, macro_rets=None,
                        alpha=(0.4, 0.2, 0.2, 0.2),
                        max_lag_corr=10, horizons=(1, 5, 10), granger_lag=5):

    s_corr_list, s_LP_list, s_co_list, s_gr_list, lag_list = [], [], [], [], []

    price_cols = set(prices_rets.columns)
    macro_cols = set(macro_rets.columns) if macro_rets is not None else set()

    for _, e in edges_df.iterrows():
        src, dst = e["source"], e["target"]

        # X
        if src in price_cols:
            x = prices_rets[src]
        elif src in macro_cols:
            x = macro_rets[src]
        else:
            s_corr_list.append(np.nan)
            s_LP_list.append(np.nan)
            s_co_list.append(np.nan)
            s_gr_list.append(np.nan)
            lag_list.append(np.nan)
            continue

        # Y
        if dst not in price_cols:
            s_corr_list.append(np.nan)
            s_LP_list.append(np.nan)
            s_co_list.append(np.nan)
            s_gr_list.append(np.nan)
            lag_list.append(np.nan)
            continue

        y = prices_rets[dst]

        sc, lag = signal_crosscorr(x, y, max_lag=max_lag_corr)
        slp = signal_LP(x, y, horizons=horizons, controls=macro_rets)

        if src in price_cols and dst in price_cols:
            sco = signal_cointegration_ecm(prices_level[src], prices_level[dst])
        else:
            sco = np.nan

        sgr = signal_granger(x, y, maxlag=granger_lag)

        s_corr_list.append(sc)
        s_LP_list.append(slp)
        s_co_list.append(sco)
        s_gr_list.append(sgr)
        lag_list.append(lag)

    edges_df["s_corr"] = s_corr_list
    edges_df["s_LP"] = s_LP_list
    edges_df["s_co"] = s_co_list
    edges_df["s_gr"] = s_gr_list
    edges_df["lag_hint_real"] = lag_list

    # 每个 template 内做 0-1 normalizing，然后加权
    ws = []
    for tmpl, g in edges_df.groupby("template"):
        idx = g.index
        s1 = _minmax(edges_df.loc[idx, "s_LP"].values)
        s2 = _minmax(edges_df.loc[idx, "s_co"].values)
        s3 = _minmax(edges_df.loc[idx, "s_gr"].values)
        s4 = _minmax(edges_df.loc[idx, "s_corr"].values)
        w = alpha[0] * s1 + alpha[1] * s2 + alpha[2] * s3 + alpha[3] * s4
        ws.append(pd.Series(w, index=idx))

    edges_df["w"] = pd.concat(ws).sort_index()

    return edges_df


# =====================================================
# 6. 主流程
# =====================================================

def main(args):
    meta = pd.read_csv(args.metadata)
    prices = pd.read_csv(args.prices)
    macro = pd.read_csv(args.macro)

    prices["date"] = pd.to_datetime(prices["date"], format="mixed", dayfirst=True, errors="raise")
    macro["date"] = pd.to_datetime(macro["date"], format="mixed", dayfirst=True, errors="raise")

    prices = prices.set_index("date").sort_index()
    macro = macro.set_index("date").sort_index()

    # symbol → price_col
    # 如果 symbol 已经是 px_* 这种“价格列名”，直接用它
    if meta["symbol"].iloc[0].startswith("px_"):
        meta["price_col"] = meta["symbol"]
        # 下游的 main_output_of 也直接写成上游的 px_* 名字
        if "main_output_of" in meta.columns:
            meta["price_col_upstream"] = meta["main_output_of"]
        else:
            meta["price_col_upstream"] = None
    else:
        # 否则使用期货代码 -> 价格列名的映射（CL->wti 这种）
        meta["price_col"] = meta["symbol"].map(SYMBOL_TO_PRICECOL)
        if "main_output_of" in meta.columns:
            meta["price_col_upstream"] = meta["main_output_of"].map(SYMBOL_TO_PRICECOL)
        else:
            meta["price_col_upstream"] = None

    available_cols = set(prices.columns)
    ok = meta["price_col"].isin(available_cols)
    if (~ok).any():
        print("[WARN] 以下 symbol 未映射到价格列，将会跳过：")
        print(meta.loc[~ok, ["symbol", "price_col"]].to_string(index=False))

    meta_mapped = meta.loc[ok].copy()

    # 生成 T1–T7 边
    edges = []
    edges += gen_T2_chain(meta_mapped)
    edges += gen_T3_substitution(meta_mapped)
    edges += gen_T4_macro(meta_mapped, [c for c in DEFAULT_MACRO_COLS if c in macro.columns])
    edges += gen_T5_T7_parity(meta_mapped)
    edges += gen_T1_termstructure(meta_mapped, prices.columns)

    edges_df = pd.DataFrame(edges)
    if edges_df.empty:
        print("[WARN] 无候选边，请检查 metadata")
        return

    # 加权（可选）
    if args.compute_weights:
        prices_rets = _to_returns(prices)
        macro_rets = _to_returns(macro[[c for c in DEFAULT_MACRO_COLS if c in macro.columns]])

        edges_df = compute_edge_scores(
            edges_df,
            prices_level=prices,
            prices_rets=prices_rets,
            macro_rets=macro_rets,
            alpha=(args.alpha1, args.alpha2, args.alpha3, args.alpha4),
            max_lag_corr=args.max_lag_corr,
            horizons=tuple(args.horizons),
            granger_lag=args.granger_lag
        )

    # 输出
    out_cols = ["source", "target", "template", "rule", "lag_hint", "lag_hint_real", "notes"]
    if args.compute_weights:
        out_cols += ["w", "s_LP", "s_co", "s_gr", "s_corr"]

    for c in out_cols:
        if c not in edges_df.columns:
            edges_df[c] = np.nan

    edges_df[out_cols].to_csv(args.out_edges, index=False)
    print(f"[OK] candidates -> {args.out_edges} (rows={len(edges_df)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--prices", default="panel_prices.csv")
    ap.add_argument("--macro", default="panel_macro.csv")
    ap.add_argument("--out_edges", default="edges_candidates.csv")
    ap.add_argument("--compute_weights", action="store_true")

    ap.add_argument("--alpha1", type=float, default=0.4)
    ap.add_argument("--alpha2", type=float, default=0.2)
    ap.add_argument("--alpha3", type=float, default=0.2)
    ap.add_argument("--alpha4", type=float, default=0.2)
    ap.add_argument("--max_lag_corr", type=int, default=10)
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--granger_lag", type=int, default=5)

    args, unknown = ap.parse_known_args()
    main(args)