#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/prepare_dataset.py

Generate mechanism-prior candidate edges from per-panel metadata files, and
save derived artifacts to `dataset/derived/`.

Repo convention:
  dataset/
    panel_20.csv, panel_30.csv, panel_40.csv, panel_50.csv
    panel_macro.csv
    metadata_20.csv, metadata_30.csv, metadata_40.csv, metadata_50.csv
  dataset/derived/   (generated)
    edges_candidates_20.csv, ...

Sanity checks per panel:
- edge endpoint match ratio vs panel node columns (supports optional 'px_' prefix compatibility)
- adjacency off-diagonal density after building A_mech via data_provider.build_adjacency_from_edges

Outputs:
  dataset/derived/edges_candidates_{N}.csv
  dataset/derived/reports/prepare_panel{N}.json

Usage:
  python scripts/prepare_dataset.py --dataset_dir dataset --panels 20 30 40 50
  python scripts/prepare_dataset.py --dataset_dir dataset --panels 40 --force
"""

from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch

from data_provider import build_adjacency_from_edges


def _read_panel_node_cols(panel_csv: Path) -> List[str]:
    df = pd.read_csv(panel_csv)
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError(f"Panel file has too few columns: {panel_csv}")
    # First column is assumed to be date/index
    return cols[1:]


def _call_build_edges(build_script: Path, metadata_csv: Path, out_edges: Path) -> None:
    # Probe help to guess flags
    try:
        help_text = subprocess.check_output(["python", str(build_script), "-h"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        help_text = ""

    candidates = [
        ["--metadata", str(metadata_csv), "--out_edges", str(out_edges)],
        ["--metadata_path", str(metadata_csv), "--out_path", str(out_edges)],
        ["--meta", str(metadata_csv), "--out", str(out_edges)],
        ["--input", str(metadata_csv), "--output", str(out_edges)],
    ]

    def flags_exist(flags):
        if not help_text:
            return True
        for f in flags[::2]:
            if f not in help_text:
                return False
        return True

    last_err = None
    for flags in candidates:
        if not flags_exist(flags):
            continue
        try:
            subprocess.check_call(["python", str(build_script)] + flags)
            return
        except Exception as e:
            last_err = e

    # fallback positional
    try:
        subprocess.check_call(["python", str(build_script), str(metadata_csv), str(out_edges)])
        return
    except Exception as e:
        last_err = e

    raise RuntimeError(
        "Failed to call build_edges_from_metadata.py automatically. "
        "Run `python build_edges_from_metadata.py -h` and adjust flags in scripts/prepare_dataset.py.\n"
        f"Last error: {last_err}"
    )


def _edge_match_stats(edges_csv: Path, node_cols: List[str]) -> dict:
    edges = pd.read_csv(edges_csv)
    if not {"source", "target"}.issubset(edges.columns):
        if {"src", "dst"}.issubset(edges.columns):
            edges = edges.rename(columns={"src": "source", "dst": "target"})
        else:
            raise ValueError(f"Edges CSV missing (source,target) columns: {edges_csv}")

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


def prepare_one_panel(dataset_dir: Path, panel: int, build_script: Path, force: bool) -> Path:
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
    if out_edges.exists() and not force:
        print(f"[SKIP] {out_edges} exists. Use --force to regenerate.")
    else:
        print(f"[GEN] panel={panel} -> {out_edges}")
        _call_build_edges(build_script, meta_csv, out_edges)

    node_cols = _read_panel_node_cols(panel_csv)
    match = _edge_match_stats(out_edges, node_cols)

    A = build_adjacency_from_edges(
        str(out_edges),
        node_cols,
        weight_col="w",
        default_weight=1.0,
        symmetrize=True,
        add_self_loops=True,
        normalize=True,
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
        "notes": "If hit ratios are low or density is ~0, ensure edges use the same node identifiers as panel columns (prefer metadata.price_col).",
    }

    report_path = report_dir / f"prepare_panel{panel}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] panel={panel} src_hit={match['src_hit_ratio']:.3f} dst_hit={match['dst_hit_ratio']:.3f} density={density:.4f}")
    print(f"     report: {report_path}")
    return out_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset")
    ap.add_argument("--panels", type=int, nargs="+", default=[20, 30, 40, 50])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--build_script", type=str, default="build_edges_from_metadata.py")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    build_script = Path(args.build_script)
    if not build_script.exists():
        for cand in [Path("build_edges_from_metadata.py"), Path("scripts/build_edges_from_metadata.py"), Path("exp/scripts/build_edges_from_metadata.py")]:
            if cand.exists():
                build_script = cand
                break
    if not build_script.exists():
        raise FileNotFoundError(f"Cannot find build_edges_from_metadata.py at {args.build_script} or common locations.")

    print("[INFO] dataset_dir:", dataset_dir.resolve())
    print("[INFO] build_script:", build_script.resolve())
    print("[INFO] panels:", args.panels)

    for p in args.panels:
        prepare_one_panel(dataset_dir, int(p), build_script, args.force)

if __name__ == "__main__":
    main()
