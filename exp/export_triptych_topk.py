#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone triptych exporter + plotter (no dependency on case_oil_study.py)

Inputs:
- panel_prices CSV (must have 'date' column + asset columns)
- panel_macro  CSV (must have 'date' column + macro columns)
- edges_candidates CSV (src/dst/w or u/v/w)
- ckpt: state_dict of MechAware_GP_STGNN_MultiTask

Outputs:
- topK_triptych_<anchor>_<date>.csv  (src,dst,w,tag,gamma,date_end + optional mechanism info)
- triptych_<anchor>_<date>.png / .pdf (top-conf style, fixed layout)
"""

import os
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# IMPORTANT: model/data definitions come from this module (NOT case_oil_study)
import gp_mech_multitask_stgnn as gp


# -------------------------
# Utilities: build A_mech
# -------------------------
def build_A_mech_from_edges(edges_csv: str, node_cols: list[str], weight_col: str = "w",
                            allow_self: bool = False, symmetrize: bool = True,
                            default_weight: float = 0.0) -> np.ndarray:
    """
    Build A_mech (N,N) from edges_candidates.csv without relying on case_oil code.
    Accepts src/dst or u/v columns + weight_col.
    """
    df = pd.read_csv(edges_csv)

    # infer columns
    # infer src/dst columns (robust)
    cands = [
        ("src", "dst"),
        ("u", "v"),
        ("source", "target"),
        ("from", "to"),
        ("from_node", "to_node"),
        ("node_i", "node_j"),
    ]

    src_col = dst_col = None
    for a, b in cands:
        if a in df.columns and b in df.columns:
            src_col, dst_col = a, b
            break

    if src_col is None:
        raise ValueError(
            f"edges csv must have one of these column pairs: {cands}. "
            f"Got columns={list(df.columns)}"
        )

    if weight_col not in df.columns:
        # fallback guesses
        for cand in ["w", "weight", "value", "score"]:
            if cand in df.columns:
                weight_col = cand
                break
        if weight_col not in df.columns:
            raise ValueError(f"Cannot find weight column. Provided weight_col not in file, and no fallback found.")

    idx = {n: i for i, n in enumerate(node_cols)}
    N = len(node_cols)
    A = np.full((N, N), float(default_weight), dtype=float)

    # default_weight usually 0; if not 0, you may want to only write provided edges
    if default_weight != 0.0:
        A[:] = float(default_weight)

    # start from zeros if default is 0
    if default_weight == 0.0:
        A[:] = 0.0

    for _, r in df.iterrows():
        s = str(r[src_col])
        d = str(r[dst_col])
        if (s not in idx) or (d not in idx):
            continue
        if (not allow_self) and (s == d):
            continue
        w = float(r[weight_col])
        A[idx[s], idx[d]] = w
        if symmetrize:
            A[idx[d], idx[s]] = w

    return A


# -------------------------
# Utilities: pick snapshot
# -------------------------
@torch.no_grad()
def pick_one_snapshot(model, loader, A_mech_t, device, pick="last", pick_date=None):
    """
    Return a single snapshot from test loader:
      date_end (YYYY-MM-DD), A_learn (N,N) numpy, gamma float
    """
    model.eval()
    chosen = None

    for x_seq, y_seq_true, date_end in loader:
        x_seq = x_seq.to(device)

        # IMPORTANT: model forward signature must be (x_seq, A_mech)
        y_pred, A_learn, gamma = model(x_seq, A_mech_t)

        g = float(gamma.detach().item()) if torch.is_tensor(gamma) else float(gamma)

        # normalize date_end
        if isinstance(date_end, (list, tuple, np.ndarray, pd.Series)):
            dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in date_end]
        else:
            try:
                dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in list(date_end)]
            except Exception:
                dates = [pd.to_datetime(date_end).strftime("%Y-%m-%d")] * x_seq.size(0)

        if pick == "first":
            return dates[0], A_learn[0].detach().cpu().numpy(), g

        if pick == "date":
            if pick_date is None:
                raise ValueError("pick_date must be provided when pick='date'")
            for b in range(x_seq.size(0)):
                if dates[b] == pick_date:
                    return dates[b], A_learn[b].detach().cpu().numpy(), g

        chosen = (dates[-1], A_learn[-1].detach().cpu().numpy(), g)

    if chosen is None:
        raise RuntimeError("Empty loader; cannot pick snapshot.")
    return chosen


# -------------------------
# Utilities: top-k edges
# -------------------------
def topk_edges_from_adj(A: np.ndarray, node_cols: list[str], anchor: str, k: int = 12, mode: str = "out"):
    """
    mode:
      - out: anchor -> others top-k by |w|
      - in:  others -> anchor top-k by |w|
      - global: top-k over all pairs
    Return list of dicts: src,dst,w
    """
    A = np.asarray(A)
    idx = {n: i for i, n in enumerate(node_cols)}
    if mode != "global" and anchor not in idx:
        raise ValueError(f"anchor '{anchor}' not found in node_cols")

    rows = []
    N = A.shape[0]

    if mode == "global":
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = float(A[i, j])
                if w == 0.0:
                    continue
                rows.append({"src": node_cols[i], "dst": node_cols[j], "w": w})
    elif mode == "out":
        a = idx[anchor]
        for j in range(N):
            if j == a:
                continue
            w = float(A[a, j])
            if w == 0.0:
                continue
            rows.append({"src": anchor, "dst": node_cols[j], "w": w})
    elif mode == "in":
        a = idx[anchor]
        for i in range(N):
            if i == a:
                continue
            w = float(A[i, a])
            if w == 0.0:
                continue
            rows.append({"src": node_cols[i], "dst": anchor, "w": w})
    else:
        raise ValueError("mode must be one of: out, in, global")

    rows = sorted(rows, key=lambda r: abs(r["w"]), reverse=True)[:k]
    return rows


# -------------------------
# Optional: attach mechanism labels
# -------------------------
def build_mech_lookup(edges_csv: str):
    df = pd.read_csv(edges_csv)

    if "src" in df.columns and "dst" in df.columns:
        src_col, dst_col = "src", "dst"
    elif "u" in df.columns and "v" in df.columns:
        src_col, dst_col = "u", "v"
    else:
        return {}

    mech_col = None
    for c in ["mechanism", "type", "channel", "relation", "template", "notes", "rule"]:
        if c in df.columns:
            mech_col = c
            break
    if mech_col is None:
        return {}

    lookup = {}
    for _, r in df.iterrows():
        lookup[(str(r[src_col]), str(r[dst_col]))] = str(r[mech_col])
    return lookup


def attach_mechanism(df_edges: pd.DataFrame, mech_lookup: dict):
    if not mech_lookup:
        return df_edges
    df_edges = df_edges.copy()
    df_edges["mechanism"] = [
        mech_lookup.get((r["src"], r["dst"]), "") for _, r in df_edges.iterrows()
    ]
    return df_edges


# -------------------------
# Plot: top-conf triptych (fixed layout)
# -------------------------
DEFAULT_FUTURE_LABEL = {
    "px_wti": "CL",
    "px_brent": "BZ",
    "px_heating_oil": "HO",
    "px_rbo_gas": "RB",
    "px_copper": "HG",
    "px_silver": "SI",
    "px_platinum": "PL",
    "px_cocoa": "CC",
    "px_coffee": "KC",
    "px_soy_oil": "BO",
}

DEFAULT_ECON_POS = {
    "HO": (0.0, 0.0),
    "CL": (-1.6, 0.6),
    "BZ": (-1.6, -0.6),
    "RB": (1.6, 0.0),
    "HG": (0.0, 1.4),
    "SI": (0.8, 1.1),
    "PL": (1.2, 0.9),
    "KC": (-1.1, 0.2),
    "CC": (-0.9, 1.05),
    "BO": (0.0, -1.4),
}


def draw_triptych_from_df(df: pd.DataFrame, out_png: str, out_pdf: str,
                          future_label: dict, econ_pos: dict, tau: float = 0.05):
    gamma = float(df["gamma"].iloc[0]) if "gamma" in df.columns and len(df) else None

    def draw_panel(ax, edges_df, title, color, linestyle):
        ax.set_title(title, fontsize=12)
        ax.axis("off")

        # nodes
        for code, (x, y) in econ_pos.items():
            ax.add_patch(Circle((x, y), 0.16, facecolor="white", edgecolor="black", lw=1.2, zorder=3))
            ax.text(x, y, code, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)

        kept = edges_df.loc[edges_df["w"].abs() >= tau].copy()
        if kept.empty:
            ax.text(0, -0.2, f"No edges with |w| ≥ {tau:.2f}", ha="center", fontsize=10)
            return

        wmax = kept["w"].abs().max()
        for _, r in kept.iterrows():
            su = future_label.get(r["src"], r["src"])
            sv = future_label.get(r["dst"], r["dst"])
            if su not in econ_pos or sv not in econ_pos:
                # skip nodes not in layout (keeps figure clean)
                continue

            (x1, y1), (x2, y2) = econ_pos[su], econ_pos[sv]
            lw = 1.5 + 4.0 * abs(r["w"]) / (wmax + 1e-6)

            ax.annotate(
                "",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", linewidth=lw, color=color,
                                linestyle=linestyle, shrinkA=12, shrinkB=12),
                zorder=2
            )
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, f"{r['w']:.2f}", fontsize=9, ha="center", va="center")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=220)

    draw_panel(axes[0], df[df["tag"] == "mech"], r"Mechanism prior $A_{\mathrm{mech}}$", "black", "-")
    draw_panel(axes[1], df[df["tag"] == "learn"], r"Learned graph $A_{\mathrm{learn}}$", "gray", "--")

    title3 = r"Hybrid graph $A_{hyb}$"
    if gamma is not None:
        title3 = rf"Hybrid graph $A_{{hyb}}$ ($\gamma={gamma:.2f}$)"
    draw_panel(axes[2], df[df["tag"] == "hyb"], title3, "#1f4fd8", "-")

    for ax in axes:
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-1.8, 1.8)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_prices", required=True, type=str)
    ap.add_argument("--panel_macro", required=True, type=str)
    ap.add_argument("--edges", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)

    ap.add_argument("--out_dir", default="out_triptych", type=str)
    ap.add_argument("--topk", default=12, type=int)
    ap.add_argument("--mode", default="out", choices=["out", "in", "global"])
    ap.add_argument("--anchor", default="px_heating_oil", type=str)

    ap.add_argument("--window_size", default=30, type=int)
    ap.add_argument("--horizon", default=5, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--train_ratio", default=0.68, type=float)
    ap.add_argument("--val_ratio", default=0.20, type=float)

    ap.add_argument("--gcn_hidden", default=32, type=int)
    ap.add_argument("--gru_hidden", default=64, type=int)
    ap.add_argument("--graph_hidden", default=32, type=int)
    ap.add_argument("--model_mode", default="prior_residual", type=str)

    ap.add_argument("--weight_col", default="w", type=str)
    ap.add_argument("--default_weight", default=0.0, type=float)
    ap.add_argument("--allow_self_loops", action="store_true")
    ap.add_argument("--no_symmetrize", action="store_true")

    ap.add_argument("--snapshot_pick", default="last", choices=["last", "first", "date"])
    ap.add_argument("--snapshot_date", default=None, type=str)

    ap.add_argument("--tau", default=0.05, type=float)
    ap.add_argument("--labels_json", default=None, type=str)   # optional: custom mapping
    ap.add_argument("--layout_json", default=None, type=str)   # optional: custom layout

    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 1) load data (from gp module)
    price_df, returns_df, macro_df, node_cols, macro_cols = gp.load_panel_from_two_files(
        args.panel_prices, args.panel_macro
    )

    # ---- find date column robustly ----
    date_col = None
    for cand in ["date", "Date", "DATE", "timestamp", "Timestamp", "time", "Time"]:
        if cand in price_df.columns:
            date_col = cand
            break

    # if not found, maybe date is in index
    if date_col is None:
        if price_df.index.name is not None and "date" in str(price_df.index.name).lower():
            date_col = "__index__"
        else:
            # last resort: if first column looks like datetime, treat it as date
            first = price_df.columns[0]
            try:
                pd.to_datetime(price_df[first].iloc[:5])
                date_col = first
            except Exception:
                raise KeyError(
                    f"Cannot find date column in price_df. Columns={list(price_df.columns)}"
                )

    # 2) dataset + splits (replicate common logic: split on returns length)
    full_ds = gp.GPMultiTaskDataset(
        returns_df=returns_df,
        price_df=price_df,
        macro_df=macro_df,
        window_size=args.window_size,
        horizon=args.horizon,
    )

    T = len(returns_df)
    train_end = int(T * args.train_ratio)
    val_end = int(T * (args.train_ratio + args.val_ratio))

    def idx_range(start, end):
        return list(range(start, end))

    train_idx = idx_range(0, train_end - args.window_size - args.horizon + 1)
    val_idx = idx_range(train_end - args.window_size - args.horizon + 1, val_end - args.window_size - args.horizon + 1)
    test_idx = idx_range(val_end - args.window_size - args.horizon + 1, T - args.window_size - args.horizon + 1)

    class SubsetWithDate(torch.utils.data.Dataset):
        def __init__(self, full_ds, indices):
            self.full_ds = full_ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            idx = self.indices[i]
            x, y = self.full_ds[idx]
            # window end date = idx + window_size - 1
            # price_df has 'date'
            if date_col == "__index__":
                date_end = price_df.index[idx + args.window_size - 1]
            else:
                date_end = price_df[date_col].iloc[idx + args.window_size - 1]
                date_end = pd.to_datetime(date_end).strftime("%Y-%m-%d")  # <-- 关键：转成 str
            return x, y, date_end

    test_ds = SubsetWithDate(full_ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3) A_mech (independent builder)
    A_mech_np = build_A_mech_from_edges(
        edges_csv=args.edges,
        node_cols=node_cols,
        weight_col=args.weight_col,
        allow_self=args.allow_self_loops,
        symmetrize=(not args.no_symmetrize),
        default_weight=args.default_weight,
    )
    A_mech_t = torch.tensor(A_mech_np, dtype=torch.float32, device=device)

    # 4) build model + load state_dict (your ckpt is a state_dict)
    # infer F_total
    sample_x, _ = full_ds[0]
    F_total = sample_x.shape[-1]

    model = gp.MechAware_GP_STGNN_MultiTask(
        num_nodes=len(node_cols),
        in_dim=F_total,
        mode=args.model_mode,
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        graph_hidden=args.graph_hidden,
        horizon=args.horizon,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys:", missing)
    if unexpected:
        print("[WARN] unexpected keys:", unexpected)

    # 5) snapshot
    snap_date, A_learn_np, gamma = pick_one_snapshot(
        model=model,
        loader=test_loader,
        A_mech_t=A_mech_t,
        device=device,
        pick=args.snapshot_pick,
        pick_date=args.snapshot_date,
    )

    A_hyb_np = gamma * A_mech_np + (1.0 - gamma) * A_learn_np

    # 6) export top-k edges
    rows = []
    for tag, A in [("mech", A_mech_np), ("learn", A_learn_np), ("hyb", A_hyb_np)]:
        es = topk_edges_from_adj(A, node_cols, anchor=args.anchor, k=args.topk, mode=args.mode)
        for r in es:
            r["tag"] = tag
            r["gamma"] = float(gamma)
            r["date_end"] = snap_date
            rows.append(r)

    df = pd.DataFrame(rows)

    mech_lookup = build_mech_lookup(args.edges)
    df = attach_mechanism(df, mech_lookup)

    out_csv = os.path.join(args.out_dir, f"top{args.topk}_triptych_{args.mode}_{args.anchor}_{snap_date}.csv")
    df.to_csv(out_csv, index=False)
    print("[SAVED]", out_csv)

    # 7) plot triptych (top-conf style)
    future_label = DEFAULT_FUTURE_LABEL
    econ_pos = DEFAULT_ECON_POS

    if args.labels_json:
        future_label = json.loads(args.labels_json)
    if args.layout_json:
        econ_pos = json.loads(args.layout_json)

    out_png = os.path.join(args.out_dir, f"triptych_{args.mode}_{args.anchor}_{snap_date}.png")
    out_pdf = os.path.join(args.out_dir, f"triptych_{args.mode}_{args.anchor}_{snap_date}.pdf")

    draw_triptych_from_df(df, out_png, out_pdf, future_label=future_label, econ_pos=econ_pos, tau=args.tau)
    print("[SAVED]", out_png)
    print("[SAVED]", out_pdf)


if __name__ == "__main__":
    main()

