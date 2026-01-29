#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/prepare_dataset.py  (v2: metadata -> edges, no external build script)

Goal
----
Keep `dataset/` minimal (panel + macro + metadata) and generate derived artifacts
(edges_candidates_*) reproducibly.

Expected inputs (repo convention)
--------------------------------
dataset/
  panel_20.csv, panel_30.csv, panel_40.csv, panel_50.csv
  panel_macro.csv
  metadata_20.csv, metadata_30.csv, metadata_40.csv, metadata_50.csv

Generated outputs
-----------------
dataset/derived/
  edges_candidates_20.csv, ...
  reports/prepare_panel20.json, ...

Metadata schema (robust, best-effort)
-------------------------------------
We support common column names; at minimum we try to use:
- price_col (preferred node id, must match panel column name)
- symbol (fallback node id)
- main_output_of (upstream input commodity; edges upstream -> downstream)
- substitution_group (connect commodities within group, undirected)
- sector (optional; can add weak within-sector edges)
- chain_level (optional; used for directional sanity)

If a column is missing, that rule is skipped.

Usage
-----
python scripts/prepare_dataset.py --dataset_dir dataset --panels 20 30 40 50
python scripts/prepare_dataset.py --dataset_dir dataset --panels 40 --force

Notes
-----
- This script writes *edge lists*, not adjacency matrices.
- Your `data_provider.build_adjacency_from_edges()` consumes the edge list
  and applies self-loop + symmetrization + D^{-1/2} A D^{-1/2} normalization.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from data_provider import build_adjacency_from_edges


# -----------------------------
# helpers
# -----------------------------
def _read_panel_node_cols(panel_csv: Path) -> List[str]:
    df = pd.read_csv(panel_csv)
    if df.shape[1] < 2:
        raise ValueError(f"Panel file has too few columns: {panel_csv}")
    # First column is date/index; the rest are node ids
    return list(df.columns[1:])

def _normalize_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None

def _node_id_map(meta: pd.DataFrame) -> Tuple[List[str], Dict[str,str]]:
    """
    Build node id list from metadata.
    Returns:
      node_ids: preferred ids (price_col if exists else symbol)
      alias_to_node: map from alternative identifiers -> node_id
    """
    col_price = _pick_col(meta, ["price_col", "price", "pricecol", "px_col", "px"])
    col_symbol = _pick_col(meta, ["symbol", "ticker", "contract", "id", "name"])

    node_ids = []
    alias_to = {}

    for _, r in meta.iterrows():
        sym = _normalize_str(r[col_symbol]) if col_symbol else ""
        px  = _normalize_str(r[col_price]) if col_price else ""

        node = px if px else sym
        if not node:
            continue

        node_ids.append(node)

        # aliases
        if sym:
            alias_to[sym] = node
            alias_to[sym.lower()] = node
        if px:
            alias_to[px] = node
            alias_to[px.lower()] = node
        # also accept "px_"+sym convention if user stores that in panel
        if sym:
            alias_to["px_" + sym] = node
            alias_to["px_" + sym.lower()] = node

    # unique preserve order
    seen = set()
    uniq = []
    for n in node_ids:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq, alias_to

def _resolve(alias_to_node: Dict[str,str], raw: str, panel_nodes: set[str]) -> str | None:
    raw = _normalize_str(raw)
    if not raw:
        return None
    # direct hit
    if raw in panel_nodes:
        return raw
    # alias -> node_id, then see if that is in panel nodes
    key = raw
    if key in alias_to_node:
        cand = alias_to_node[key]
        if cand in panel_nodes:
            return cand
        # panel might have px_ prefix
        if ("px_" + cand) in panel_nodes:
            return "px_" + cand
    # try lower
    key = raw.lower()
    if key in alias_to_node:
        cand = alias_to_node[key]
        if cand in panel_nodes:
            return cand
        if ("px_" + cand) in panel_nodes:
            return "px_" + cand
    # try add px_ prefix
    if ("px_" + raw) in panel_nodes:
        return "px_" + raw
    if ("px_" + raw.lower()) in panel_nodes:
        return "px_" + raw.lower()
    return None

def _make_edges_from_metadata(meta: pd.DataFrame, panel_nodes: List[str]) -> pd.DataFrame:
    """
    Construct candidate edges with weights.

    Rules (when columns exist):
      1) main_output_of: upstream -> downstream (w=2.0)
      2) substitution_group: fully connect within group, undirected (w=1.5)
      3) sector: weak within-sector complete graph (w=0.5, optional & can be dense)

    Returns edge dataframe columns: source, target, w
    """
    panel_set = set(panel_nodes)
    node_ids, alias_to = _node_id_map(meta)

    col_price = _pick_col(meta, ["price_col", "price_col_id", "price", "pricecol", "px_col", "px"])
    col_symbol = _pick_col(meta, ["symbol", "ticker", "contract", "id", "name"])
    col_main = _pick_col(meta, ["main_output_of", "main_input_of", "main_output", "input", "upstream"])
    col_sub = _pick_col(meta, ["substitution_group", "sub_group", "subgroup", "substitute_group"])
    col_sector = _pick_col(meta, ["sector", "category"])
    col_chain = _pick_col(meta, ["chain_level", "chain", "level"])

    # Choose the "self id" per row for edges
    def row_node(r) -> str | None:
        px = _normalize_str(r[col_price]) if col_price else ""
        sym = _normalize_str(r[col_symbol]) if col_symbol else ""
        raw = px if px else sym
        return _resolve(alias_to, raw, panel_set)

    edges = {}  # (u,v) -> weight (accumulate max)
    def add(u, v, w):
        if u is None or v is None or u == v:
            return
        key = (u, v)
        if key in edges:
            edges[key] = max(edges[key], float(w))
        else:
            edges[key] = float(w)

    # (1) main_output_of edges
    if col_main:
        for _, r in meta.iterrows():
            v = row_node(r)
            up_raw = _normalize_str(r[col_main])
            if not up_raw:
                continue
            u = _resolve(alias_to, up_raw, panel_set)
            # If chain_level exists, we can enforce direction sanity loosely, but don't over-restrict.
            add(u, v, 2.0)

    # (2) substitution group edges (undirected complete graph within group)
    if col_sub:
        groups = {}
        for _, r in meta.iterrows():
            g = _normalize_str(r[col_sub])
            n = row_node(r)
            if not g or n is None:
                continue
            groups.setdefault(g, []).append(n)
        for g, nodes in groups.items():
            nodes = list(dict.fromkeys(nodes))
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    add(nodes[i], nodes[j], 1.5)
                    add(nodes[j], nodes[i], 1.5)

    # (3) sector edges (optional, weak; can be dense—use with care)
    if col_sector:
        sectors = {}
        for _, r in meta.iterrows():
            s = _normalize_str(r[col_sector])
            n = row_node(r)
            if not s or n is None:
                continue
            sectors.setdefault(s, []).append(n)
        for s, nodes in sectors.items():
            nodes = list(dict.fromkeys(nodes))
            # To avoid O(N^2) blow-up in very large sectors, cap to 60 nodes.
            if len(nodes) > 60:
                nodes = nodes[:60]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    add(nodes[i], nodes[j], 0.5)
                    add(nodes[j], nodes[i], 0.5)

    df_edges = pd.DataFrame(
        [{"source": u, "target": v, "w": w} for (u, v), w in edges.items()]
    ).sort_values(["source", "target"]).reset_index(drop=True)

    return df_edges

def _edge_match_stats(edges: pd.DataFrame, node_cols: List[str]) -> dict:
    node_set = set(map(str, node_cols))
    src = edges["source"].astype(str).tolist()
    dst = edges["target"].astype(str).tolist()

    def hit(x: str) -> bool:
        return (x in node_set) or (("px_" + x) in node_set) or (x.startswith("px_") and (x[3:] in node_set))

    src_hit = sum(hit(s) for s in src)
    dst_hit = sum(hit(t) for t in dst)

    return {
        "n_edges": int(len(edges)),
        "src_hit_ratio": float(src_hit / max(len(src), 1)),
        "dst_hit_ratio": float(dst_hit / max(len(dst), 1)),
        "node_cols_example": node_cols[:5],
        "edges_head": edges.head(5).to_dict(orient="records"),
    }

def _adj_density(A: torch.Tensor) -> float:
    A = A.detach().cpu().numpy()
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    return float((np.abs(off) > 1e-12).mean())

def prepare_one_panel(dataset_dir: Path, panel: int, force: bool) -> Path:
    panel_csv = dataset_dir / f"panel_{panel}.csv"
    meta_csv  = dataset_dir / f"metadata_{panel}.csv"
    if not panel_csv.exists():
        raise FileNotFoundError(panel_csv)
    if not meta_csv.exists():
        raise FileNotFoundError(meta_csv)

    derived_dir = dataset_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    report_dir = derived_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    out_edges = derived_dir / f"edges_candidates_{panel}.csv"

    node_cols = _read_panel_node_cols(panel_csv)
    meta = pd.read_csv(meta_csv)

    if out_edges.exists() and not force:
        print(f"[SKIP] {out_edges} exists. Use --force to regenerate.")
        edges = pd.read_csv(out_edges)
    else:
        print(f"[GEN] panel={panel} metadata -> edges: {out_edges}")
        edges = _make_edges_from_metadata(meta, node_cols)
        edges.to_csv(out_edges, index=False)

    match = _edge_match_stats(edges, node_cols)

    # Build adjacency and compute density
    A = build_adjacency_from_edges(
        str(out_edges),
        node_cols,
        weight_col="w",
        default_weight=1.0,
        self_loop=True,
        symmetrize=True,
    )
    density = _adj_density(A)

    report = {
        "panel": panel,
        "panel_csv": str(panel_csv),
        "metadata_csv": str(meta_csv),
        "edges_csv": str(out_edges),
        "match": match,
        "A_mech": {
            "shape": list(A.shape),
            "finite": bool(torch.isfinite(A).all().item()),
            "symmetric": bool(torch.allclose(A, A.T, atol=1e-6).item()),
            "offdiag_density": density,
        },
        "notes": [
            "Edges are generated from metadata (main_output_of, substitution_group, sector).",
            "If hit ratios are low or density is ~0, ensure metadata uses the same node identifiers as panel columns (prefer metadata.price_col).",
            "Sector edges are weak (w=0.5) and capped to avoid very dense graphs.",
        ],
    }

    report_path = report_dir / f"prepare_panel{panel}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] panel={panel} src_hit={match['src_hit_ratio']:.3f} dst_hit={match['dst_hit_ratio']:.3f} density={density:.4f}")
    print(f"     report: {report_path}")
    return out_edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset")
    ap.add_argument("--panels", type=int, nargs="+", default=[20, 30, 40, 50])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    print("[INFO] dataset_dir:", dataset_dir.resolve())
    print("[INFO] panels:", args.panels)

    for p in args.panels:
        prepare_one_panel(dataset_dir, int(p), args.force)

if __name__ == "__main__":
    main()
