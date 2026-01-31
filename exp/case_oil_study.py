#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oil chain interpretability case study (WTI -> RBOB / Heating Oil)

What it does:
- Load trained MechAware_GP_STGNN_MultiTask model checkpoint
- Rebuild A_mech from edges_candidates.csv
- Run on TEST split (same split logic as gp_mech_multitask_stgnn.py)
- Export per-window edge weights:
    A_mech, A_learn(t), delta(t), A_hyb(t), gamma
- Plot:
    (1) Edge weights over time (mech vs learn vs hyb)
    (2) Residual correction delta over time

Usage example:
python case_oil_study.py \
  --panel_prices panel_20_prices.csv \
  --panel_macro  panel_20_macro.csv \
  --edges edges_candidates_20.csv \
  --ckpt ckpt_panel20_h5_seed1.pt \
  --horizon 5 --window_size 30 \
  --out_dir out_case_oil

Notes:
- No need to modify gp_mech_multitask_stgnn.py.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Import your existing code (same folder or PYTHONPATH)
import gp_mech_multitask_stgnn as gp
import networkx as nx


from matplotlib.patches import Circle
import math

FUTURE_LABEL = {
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

ECON_POS = {
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

def draw_triptych_topconf(out_path, node_cols, edges_mech, edges_learn, edges_hyb, gamma,
                          tau=0.05, anchor=None):
    """
    Top-conf style triptych:
      - fixed economic layout
      - futures symbols
      - thresholded edges
      - line-style differences for mech vs learn vs hyb
    edges_*: list of (src_name, dst_name, w) in your internal node names (px_*)
    """
    import matplotlib.pyplot as plt

    def _draw(ax, edges, title, color, linestyle):
        ax.set_title(title, fontsize=12)
        ax.axis("off")

        # nodes
        for code, (x, y) in ECON_POS.items():
            ax.add_patch(Circle((x, y), 0.16, facecolor="white", edgecolor="black", lw=1.2, zorder=3))
            ax.text(x, y, code, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)

        # map edges to futures codes and filter
        kept = []
        for u, v, w in edges:
            su = FUTURE_LABEL.get(u, u)
            sv = FUTURE_LABEL.get(v, v)
            if (su in ECON_POS) and (sv in ECON_POS) and (abs(w) >= tau):
                kept.append((su, sv, float(w)))

        if not kept:
            ax.text(0, -0.2, f"No edges with |w| ≥ {tau:.2f}", ha="center", fontsize=10)
            return

        wmax = max(abs(w) for _, _, w in kept) or 1.0
        for su, sv, w in kept:
            (x1, y1), (x2, y2) = ECON_POS[su], ECON_POS[sv]
            lw_min, lw_max = 1.2, 2.8
            lw = lw_min + (lw_max - lw_min) * abs(w) / (wmax + 1e-6)

            ax.annotate(
                "",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", linewidth=lw, color=color,
                                linestyle=linestyle, shrinkA=12, shrinkB=12),
                zorder=2
            )
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mx, my, f"{w:.2f}",
                fontsize=9,
                ha="center", va="center",
                zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white",
                    ec="none",
                    alpha=0.85
                )
            )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=220)

    _draw(axes[0], edges_mech, r"Mechanism prior $A_{\mathrm{mech}}$", "black", "-")
    _draw(axes[1], edges_learn, r"Learned graph $A_{\mathrm{learn}}$", "gray", "--")
    _draw(axes[2], edges_hyb,  rf"Hybrid graph $A_{{hyb}}$ ($\gamma={gamma:.2f}$)", "#1f4fd8", "-")

    for ax in axes:
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-1.8, 1.8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def build_splits(full_ds, returns_len, window_size, horizon, train_ratio, val_ratio):
    """
    Copy the exact split logic from gp_mech_multitask_stgnn.py main().
    """
    T = returns_len
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

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


class SubsetWithDate(torch.utils.data.Dataset):
    """
    Wrap a Subset(full_ds, indices) but also return the window-end date.

    full_ds.__getitem__(i) internally uses t = valid_idx[i]
    so we can recover date_end = full_ds.dates[t].
    """
    def __init__(self, full_ds, indices):
        self.full_ds = full_ds
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        x_seq, y_seq = self.full_ds[i]
        t = int(self.full_ds.valid_idx[i])
        date_end = str(pd.to_datetime(self.full_ds.dates[t]).date())
        return x_seq, y_seq, date_end


@torch.no_grad()
def export_oil_edges(model, loader, A_mech, node_cols, out_csv, device,
                     node_wti="wti", node_rb="rbo_gas", node_ho="heating_oil"):
    """
    Export per-window weights for:
      WTI -> RBOB
      WTI -> Heating Oil
    """
    idx = {n: i for i, n in enumerate(node_cols)}
    missing = [n for n in [node_wti, node_rb, node_ho] if n not in idx]
    if missing:
        raise ValueError(
            f"Missing nodes in panel columns: {missing}. "
            f"Available nodes: {node_cols}"
        )

    i_wti = idx[node_wti]
    i_rb  = idx[node_rb]
    i_ho  = idx[node_ho]

    A_mech = A_mech.to(device)  # (N,N)
    model.eval()

    rows = []
    for x_seq, y_seq_true, date_end in loader:
        x_seq = x_seq.to(device)  # (B,L,N,F)
        y_seq_pred, A_learn, gamma = model(x_seq, A_mech)  # A_learn:(B,N,N)

        g = float(gamma.detach().item()) if torch.is_tensor(gamma) else float(gamma)
        B = x_seq.size(0)

        for b in range(B):
            # --- WTI -> RBOB
            mech_rb  = float(A_mech[i_wti, i_rb].item())
            learn_rb = float(A_learn[b, i_wti, i_rb].item())
            delta_rb = learn_rb - mech_rb
            hyb_rb   = g * mech_rb + (1.0 - g) * learn_rb

            # --- WTI -> Heating Oil
            mech_ho  = float(A_mech[i_wti, i_ho].item())
            learn_ho = float(A_learn[b, i_wti, i_ho].item())
            delta_ho = learn_ho - mech_ho
            hyb_ho   = g * mech_ho + (1.0 - g) * learn_ho

            rows.append({
                "date_end": pd.to_datetime(date_end[b]).strftime("%Y-%m-%d")
                           if isinstance(date_end, (list, tuple, np.ndarray, pd.Series))
                           else pd.to_datetime(date_end).strftime("%Y-%m-%d"),
                "gamma": g,

                "mech_wti_rb": mech_rb,
                "learn_wti_rb": learn_rb,
                "delta_wti_rb": delta_rb,
                "hyb_wti_rb": hyb_rb,

                "mech_wti_ho": mech_ho,
                "learn_wti_ho": learn_ho,
                "delta_wti_ho": delta_ho,
                "hyb_wti_ho": hyb_ho,
            })

    df = pd.DataFrame(rows)
    df["date_end"] = pd.to_datetime(df["date_end"])
    df = df.sort_values("date_end").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def plot_case(df, out_dir, prefix="oil_case"):
    os.makedirs(out_dir, exist_ok=True)

    # (1) Edge weights over time: mech vs learn vs hyb
    plt.figure()
    plt.plot(df["date_end"], df["mech_wti_rb"], label="mech WTI→RBOB")
    plt.plot(df["date_end"], df["learn_wti_rb"], label="learn WTI→RBOB")
    plt.plot(df["date_end"], df["hyb_wti_rb"], label="hyb WTI→RBOB")
    plt.legend()
    plt.xlabel("date_end")
    plt.ylabel("edge weight")
    plt.title("WTI→RBOB edge: mechanism vs learned vs hybrid")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_wti_rb_weights.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["date_end"], df["mech_wti_ho"], label="mech WTI→HO")
    plt.plot(df["date_end"], df["learn_wti_ho"], label="learn WTI→HO")
    plt.plot(df["date_end"], df["hyb_wti_ho"], label="hyb WTI→HO")
    plt.legend()
    plt.xlabel("date_end")
    plt.ylabel("edge weight")
    plt.title("WTI→HeatingOil edge: mechanism vs learned vs hybrid")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_wti_ho_weights.png"), dpi=200)
    plt.close()

    # (2) Residual correction delta over time
    plt.figure()
    plt.plot(df["date_end"], df["delta_wti_rb"], label="ΔA WTI→RBOB")
    plt.plot(df["date_end"], df["delta_wti_ho"], label="ΔA WTI→HO")
    plt.legend()
    plt.xlabel("date_end")
    plt.ylabel("delta = A_learn - A_mech")
    plt.title("Residual correction over time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_delta.png"), dpi=200)
    plt.close()

    # (3) gamma over time (optional sanity plot)
    plt.figure()
    plt.plot(df["date_end"], df["gamma"], label="gamma")
    plt.legend()
    plt.xlabel("date_end")
    plt.ylabel("gamma")
    plt.title("Fusion weight gamma over time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_gamma.png"), dpi=200)
    plt.close()




def _topk_edges_from_adj(A, node_cols, k=12, mode="global", anchor=None, include_self=False):
    """
    Return list of edges (u, v, w) sorted by |w| desc.
    mode:
      - "global": pick top-k edges from whole matrix
      - "out": pick top-k outgoing edges from anchor (u=anchor)
      - "in":  pick top-k incoming edges to anchor (v=anchor)
    """
    A = np.asarray(A)
    N = A.shape[0]
    idx = {n:i for i,n in enumerate(node_cols)}

    edges = []
    if mode == "global":
        for i in range(N):
            for j in range(N):
                if (not include_self) and (i == j):
                    continue
                w = float(A[i, j])
                if w == 0.0:
                    continue
                edges.append((node_cols[i], node_cols[j], w))
    else:
        if anchor is None:
            raise ValueError("anchor must be provided for mode out/in")
        if anchor not in idx:
            raise ValueError(f"anchor '{anchor}' not in node_cols")
        a = idx[anchor]
        if mode == "out":
            for j in range(N):
                if (not include_self) and (a == j):
                    continue
                w = float(A[a, j])
                if w == 0.0:
                    continue
                edges.append((node_cols[a], node_cols[j], w))
        elif mode == "in":
            for i in range(N):
                if (not include_self) and (i == a):
                    continue
                w = float(A[i, a])
                if w == 0.0:
                    continue
                edges.append((node_cols[i], node_cols[a], w))
        else:
            raise ValueError("mode must be global/out/in")

    edges = sorted(edges, key=lambda x: abs(x[2]), reverse=True)[:k]
    return edges


def _nodes_from_edges(edges, max_nodes=10):
    """Collect nodes appearing in edges; cap size for readability."""
    nodes = []
    for u, v, _ in edges:
        if u not in nodes:
            nodes.append(u)
        if v not in nodes:
            nodes.append(v)
        if len(nodes) >= max_nodes:
            break
    return nodes


def _draw_subgraph(ax, nodes, edges, title):
    """
    Draw directed weighted subgraph with fixed node set.
    Edge width proportional to |w|.
    """
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    for u, v, w in edges:
        if u in G and v in G:
            G.add_edge(u, v, weight=w)

    if len(G.nodes) == 0:
        ax.set_title(title)
        ax.axis("off")
        return

    # layout (deterministic-ish)
    pos = nx.spring_layout(G, seed=7)

    # node
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # edge widths
    ws = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    if len(ws) == 0:
        ax.set_title(title)
        ax.axis("off")
        return

    wmax = max(ws) if max(ws) > 0 else 1.0
    widths = [1.0 + 4.0 * (w / wmax) for w in ws]

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, width=widths, arrowsize=12)

    # edge labels (show weight)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)

    ax.set_title(title)
    ax.axis("off")


@torch.no_grad()
def pick_one_snapshot(model, loader, A_mech, device, pick="last", pick_date=None):
    """
    Return a single snapshot from test loader:
      date_end, A_learn (N,N), gamma (float)
    pick:
      - "last": last batch's last item
      - "first": first batch's first item
      - "date": match pick_date (YYYY-MM-DD) on date_end
    """
    A_mech = A_mech.to(device)
    model.eval()

    chosen = None
    for x_seq, y_seq_true, date_end in loader:
        x_seq = x_seq.to(device)
        y_pred, A_learn, gamma = model(x_seq, A_mech)
        g = float(gamma.detach().item()) if torch.is_tensor(gamma) else float(gamma)

        B = x_seq.size(0)
        # normalize date_end into list[str]
        if isinstance(date_end, (list, tuple, np.ndarray, pd.Series)):
            dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in date_end]
        else:
            # could be a tensor/list-like; fallback
            try:
                dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in list(date_end)]
            except Exception:
                dates = [pd.to_datetime(date_end).strftime("%Y-%m-%d")] * B

        if pick == "first":
            return dates[0], A_learn[0].detach().cpu().numpy(), g

        if pick == "date":
            if pick_date is None:
                raise ValueError("pick_date must be provided when pick='date'")
            for b in range(B):
                if dates[b] == pick_date:
                    return dates[b], A_learn[b].detach().cpu().numpy(), g

        # pick == "last": keep overwriting until end
        chosen = (dates[-1], A_learn[-1].detach().cpu().numpy(), g)

    if chosen is None:
        raise RuntimeError("Empty loader; cannot pick snapshot.")
    return chosen


def plot_case_triptych(out_dir, case_name, node_cols, A_mech, A_learn_snap, gamma,
                       k=12, pick_mode="out", anchor=None, max_nodes=10, tau=0.05):
    os.makedirs(out_dir, exist_ok=True)

    A_mech_np = A_mech.detach().cpu().numpy() if torch.is_tensor(A_mech) else np.asarray(A_mech)
    A_hyb = gamma * A_mech_np + (1.0 - gamma) * A_learn_snap

    edges_mech = _topk_edges_from_adj(A_mech_np, node_cols, k=k, mode=pick_mode, anchor=anchor)
    edges_learn = _topk_edges_from_adj(A_learn_snap, node_cols, k=k, mode=pick_mode, anchor=anchor)
    edges_hyb = _topk_edges_from_adj(A_hyb, node_cols, k=k, mode=pick_mode, anchor=anchor)

    out_path = os.path.join(out_dir, f"triptych_{case_name}_topconf.png")
    draw_triptych_topconf(
        out_path=out_path,
        node_cols=node_cols,
        edges_mech=edges_mech,
        edges_learn=edges_learn,
        edges_hyb=edges_hyb,
        gamma=gamma,
        tau=tau,
        anchor=anchor,
    )
    return out_path




def load_model_from_ckpt(node_cols, F_total, args, device):
    model = gp.MechAware_GP_STGNN_MultiTask(
        num_nodes=len(node_cols),
        in_dim=F_total,
        mode=args.mode,
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        graph_hidden=args.graph_hidden,
        horizon=args.horizon,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    # support both raw state_dict or {"state_dict": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_prices", required=True, type=str)
    ap.add_argument("--panel_macro", required=True, type=str)
    ap.add_argument("--edges", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)

    ap.add_argument("--out_dir", default="out_case_oil", type=str)
    ap.add_argument("--out_csv", default="oil_case_edges.csv", type=str)

    # must match training config
    ap.add_argument("--window_size", default=30, type=int)
    ap.add_argument("--horizon", default=5, type=int)
    ap.add_argument("--mode", default="prior_residual", type=str)

    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--train_ratio", default=0.68, type=float)
    ap.add_argument("--val_ratio", default=0.20, type=float)

    ap.add_argument("--weight_col", default="w", type=str)
    ap.add_argument("--default_weight", default=1.0, type=float)

    ap.add_argument("--gcn_hidden", default=32, type=int)
    ap.add_argument("--gru_hidden", default=64, type=int)
    ap.add_argument("--graph_hidden", default=32, type=int)

    ap.add_argument("--cpu", action="store_true")

    # node names for oil chain (override if your columns differ)
    ap.add_argument("--node_wti", default="px_wti", type=str)
    ap.add_argument("--node_rb", default="px_rbo_gas", type=str)
    ap.add_argument("--node_ho", default="px_heating_oil", type=str)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load panel data
    price_df, returns_df, macro_df, node_cols, macro_cols = gp.load_panel_from_two_files(
        args.panel_prices, args.panel_macro
    )

    print("[DEBUG] node_cols =", node_cols)
    print("[DEBUG] node_cols (count) =", len(node_cols))

    # 2) build A_mech (same as training)
    A_mech = gp.build_adjacency_from_edges(
        edges_path=args.edges,
        node_list=node_cols,
        weight_col=args.weight_col,
        default_weight=args.default_weight,
        self_loop=True,
        symmetrize=True,
    )

    # 3) dataset + split (same as training)
    full_ds = gp.GPMultiTaskDataset(
        returns_df=returns_df,
        price_df=price_df,
        macro_df=macro_df,
        window_size=args.window_size,
        horizon=args.horizon,
    )

    train_idx, val_idx, test_idx = build_splits(
        full_ds=full_ds,
        returns_len=len(returns_df),
        window_size=args.window_size,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    test_ds = SubsetWithDate(full_ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 4) build model & load ckpt
    sample_x, _ = full_ds[0]
    F_total = sample_x.shape[-1]
    model = load_model_from_ckpt(node_cols, F_total, args, device)

    # 5) export
    out_csv_path = os.path.join(args.out_dir, args.out_csv)
    df = export_oil_edges(
        model=model,
        loader=test_loader,
        A_mech=A_mech,
        node_cols=node_cols,
        out_csv=out_csv_path,
        device=device,
        node_wti=args.node_wti,
        node_rb=args.node_rb,
        node_ho=args.node_ho,
    )

    # 6) plots
    plot_case(df, out_dir=args.out_dir, prefix="oil_case")

    # 7) quick console summary (for your paper text)
    g_mean = df["gamma"].mean()
    g_std  = df["gamma"].std()
    d_rb   = df["delta_wti_rb"].abs().mean()
    d_ho   = df["delta_wti_ho"].abs().mean()
    print(f"[OK] Exported: {out_csv_path} (rows={len(df)})")
    print(f"[SUMMARY] gamma mean±std = {g_mean:.4f} ± {g_std:.4f}")
    print(f"[SUMMARY] mean |delta|: WTI→RBOB={d_rb:.4f}, WTI→HO={d_ho:.4f}")
    print(f"[FIG] saved to: {args.out_dir}/oil_case_*.png")

    # 7)
    # 6.5) ---- triptych subgraphs (A_mech vs A_learn vs A_hyb) ----
    # pick one snapshot from TEST: last window by default
    snap_date, A_learn_snap, g_snap = pick_one_snapshot(
        model=model,
        loader=test_loader,
        A_mech=A_mech,
        device=device,
        pick="last",         # or "first" / "date"
        pick_date=None       # if pick="date", set like "2022-03-08"
    )
    print(f"[SNAPSHOT] using date_end={snap_date}, gamma={g_snap:.4f}")

    # --- define 2-3 cases here (edit anchors to match your panel column names) ---
    cases = [
        ("oil_chain", args.node_wti),   # e.g., px_wti
        ("gasoline", args.node_rb),     # e.g., px_rbo_gas
        ("heating_oil", args.node_ho),  # e.g., px_heating_oil
    ]

    for case_name, anchor in cases:
        fig_path = plot_case_triptych(
            out_dir=args.out_dir,
            case_name=case_name,
            node_cols=node_cols,
            A_mech=A_mech,
            A_learn_snap=A_learn_snap,
            gamma=g_snap,
            k=12,
            pick_mode="out",   # "out" shows anchor -> others; can try "global"
            anchor=anchor,
            max_nodes=10,
        )
        print(f"[FIG] triptych saved: {fig_path}")



if __name__ == "__main__":
    main()
