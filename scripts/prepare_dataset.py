#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/prepare_dataset.py  (FINAL: subprocess CLI wrapper)

目标
- 不重写任何 edge 生成逻辑
- 只用你原来那条命令去生成 edges_candidates_{N}.csv
- 把派生产物写到 dataset/derived/，并输出简单 report

你的“权威生成方式”（prepare_dataset 只是批量调用）:
python build_edges_from_metadata.py \
  --metadata metadata_50.csv \
  --prices panel_50.csv \
  --macro panel_macro.csv \
  --out_edges edges_candidates_50.csv \
  --compute_weights

约定目录结构
dataset/
  panel_20.csv, panel_30.csv, panel_40.csv, panel_50.csv
  panel_macro.csv
  metadata_20.csv, metadata_30.csv, metadata_40.csv, metadata_50.csv
  derived/   (自动生成)
    edges_candidates_*.csv
    reports/prepare_panel*.json

用法（在 repo 根目录）:
  python scripts/prepare_dataset.py --dataset_dir dataset --panels 20 30 40 50 --compute_weights
  python scripts/prepare_dataset.py --dataset_dir dataset --panels 20 --compute_weights --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    # scripts/prepare_dataset.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def _run_cmd(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.check_call(cmd)


def _read_panel_node_cols(panel_csv: Path) -> List[str]:
    df = pd.read_csv(panel_csv)
    if df.shape[1] < 2:
        raise ValueError(f"Panel file has too few columns: {panel_csv}")
    return list(df.columns[1:])


def _edge_match_ratio(edges_csv: Path, node_cols: List[str]) -> dict:
    edges = pd.read_csv(edges_csv)
    if not {"source", "target"}.issubset(edges.columns):
        raise ValueError(f"Edges file missing columns source/target: {edges_csv}")

    node_set = set(map(str, node_cols))

    def hit(x: str) -> bool:
        x = str(x)
        return (x in node_set) or (("px_" + x) in node_set) or (x.startswith("px_") and (x[3:] in node_set))

    src_hit = float(sum(hit(x) for x in edges["source"]) / max(len(edges), 1))
    dst_hit = float(sum(hit(x) for x in edges["target"]) / max(len(edges), 1))
    w_unique = sorted(set(map(float, edges["w"].unique()))) if "w" in edges.columns else None

    return {
        "n_edges": int(len(edges)),
        "src_hit_ratio": src_hit,
        "dst_hit_ratio": dst_hit,
        "w_unique": w_unique,
    }


def _adj_density_from_edges(edges_csv: Path, node_cols: List[str]) -> Optional[float]:
    """
    可选 sanity check：用 data_provider.build_adjacency_from_edges 来看一下 A 的非对角稠密度。
    如果 repo 里没有 data_provider 或 import 失败，就跳过（不影响生成）。
    """
    try:
        # 确保 repo root 在 sys.path 里，避免脚本路径问题
        sys.path.insert(0, str(_repo_root()))
        from data_provider import build_adjacency_from_edges  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        print(f"[WARN] skip adjacency sanity check (cannot import data_provider/torch): {e}")
        return None

    A = build_adjacency_from_edges(
        str(edges_csv),
        node_cols,
        weight_col="w",
        default_weight=1.0,
        self_loop=True,
        symmetrize=True,
    )

    A_np = A.detach().cpu().numpy()
    off = A_np.copy()
    np.fill_diagonal(off, 0.0)
    dens = float((np.abs(off) > 1e-12).mean())
    return dens


def build_edges_via_cli(
    build_script: Path,
    metadata_csv: Path,
    prices_csv: Path,
    macro_csv: Path,
    out_edges: Path,
    compute_weights: bool,
) -> None:
    """
    关键：完全用 subprocess 调用 CLI，保证与你当时手工命令一致。
    """
    cmd = [
        "python",
        str(build_script),
        "--metadata", str(metadata_csv),
        "--prices", str(prices_csv),
        "--macro", str(macro_csv),
        "--out_edges", str(out_edges),
    ]
    if compute_weights:
        cmd.append("--compute_weights")

    _run_cmd(cmd)


def prepare_one_panel(
    dataset_dir: Path,
    panel: int,
    build_script: Path,
    compute_weights: bool,
    force: bool,
) -> Path:
    panel_csv = dataset_dir / f"panel_{panel}.csv"
    meta_csv = dataset_dir / f"metadata_{panel}.csv"
    macro_csv = dataset_dir / "panel_macro.csv"

    if not panel_csv.exists():
        raise FileNotFoundError(panel_csv)
    if not meta_csv.exists():
        raise FileNotFoundError(meta_csv)
    if not macro_csv.exists():
        raise FileNotFoundError(macro_csv)

    derived_dir = dataset_dir / "derived"
    report_dir = derived_dir / "reports"
    derived_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_edges = derived_dir / f"edges_candidates_{panel}.csv"

    if out_edges.exists() and not force:
        print(f"[SKIP] {out_edges} exists. Use --force to regenerate.")
    else:
        print(f"[GEN] panel={panel} -> {out_edges}")
        build_edges_via_cli(
            build_script=build_script,
            metadata_csv=meta_csv,
            prices_csv=panel_csv,
            macro_csv=macro_csv,
            out_edges=out_edges,
            compute_weights=compute_weights,
        )

    # reports / sanity checks (does not affect generation)
    node_cols = _read_panel_node_cols(panel_csv)
    match = _edge_match_ratio(out_edges, node_cols)
    dens = _adj_density_from_edges(out_edges, node_cols)

    report = {
        "panel": panel,
        "panel_csv": str(panel_csv),
        "metadata_csv": str(meta_csv),
        "macro_csv": str(macro_csv),
        "edges_csv": str(out_edges),
        "compute_weights": bool(compute_weights),
        "edge_stats": match,
        "A_offdiag_density": dens,
        "notes": [
            "Edges are generated by calling the original CLI of build_edges_from_metadata.py (no re-implementation).",
            "If you want bitwise-identical results across machines for statistical weights, pin python/numpy/statsmodels versions.",
        ],
    }
    report_path = report_dir / f"prepare_panel{panel}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    msg = f"[OK] panel={panel} edges={match['n_edges']} src_hit={match['src_hit_ratio']:.3f} dst_hit={match['dst_hit_ratio']:.3f}"
    if dens is not None:
        msg += f" density={dens:.4f}"
    print(msg)
    print(f"     report: {report_path}")

    return out_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset")
    ap.add_argument("--panels", type=int, nargs="+", default=[20, 30, 40, 50])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compute_weights", action="store_true")

    # build script location:
    # - if you keep it at repo root: build_edges_from_metadata.py
    # - or under scripts/: scripts/build_edges_from_metadata.py
    ap.add_argument(
        "--build_script",
        type=str,
        default="build_edges_from_metadata.py",
        help="Path to build_edges_from_metadata.py (repo root or scripts/...)",
    )

    args = ap.parse_args()
    dataset_dir = Path(args.dataset_dir)

    # resolve build script path relative to repo root
    repo_root = _repo_root()
    build_script = Path(args.build_script)
    if not build_script.is_absolute():
        # try as given (relative to cwd), then repo root, then repo_root/scripts
        candidates = [
            Path(args.build_script),
            repo_root / args.build_script,
            repo_root / "scripts" / Path(args.build_script).name,
        ]
        build_script = next((p for p in candidates if p.exists()), candidates[-1])

    if not build_script.exists():
        raise FileNotFoundError(
            f"Cannot find build script: {args.build_script}. Tried: {build_script}"
        )

    print("[INFO] repo_root:", repo_root)
    print("[INFO] dataset_dir:", dataset_dir.resolve())
    print("[INFO] build_script:", build_script.resolve())
    print("[INFO] panels:", args.panels)
    print("[INFO] compute_weights:", bool(args.compute_weights))

    for p in args.panels:
        prepare_one_panel(
            dataset_dir=dataset_dir,
            panel=int(p),
            build_script=build_script,
            compute_weights=bool(args.compute_weights),
            force=bool(args.force),
        )


if __name__ == "__main__":
    main()


