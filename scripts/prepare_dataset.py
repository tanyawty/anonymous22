# scripts/prepare_dataset.py
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import torch
import numpy as np

# 保证能 import repo 内模块
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_provider import build_adjacency_from_edges
from scripts.build_edges_from_metadata import main as build_edges_main


def adj_density(A):
    A = A.detach().cpu().numpy()
    np.fill_diagonal(A, 0.0)
    return float((np.abs(A) > 1e-12).mean())


def run_one(dataset_dir: Path, panel: int, force: bool, compute_weights: bool):
    prices = dataset_dir / f"panel_{panel}.csv"
    meta   = dataset_dir / f"metadata_{panel}.csv"
    macro  = dataset_dir / "panel_macro.csv"

    out_dir = dataset_dir / "derived"
    out_dir.mkdir(exist_ok=True)
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(exist_ok=True)

    out_edges = out_dir / f"edges_candidates_{panel}.csv"

    if out_edges.exists() and not force:
        print(f"[SKIP] {out_edges}")
    else:
        class Args: pass
        args = Args()
        args.metadata = str(meta)
        args.prices   = str(prices)
        args.macro    = str(macro)
        args.out_edges = str(out_edges)
        args.compute_weights = compute_weights

        # 原脚本的默认超参（完全一致）
        args.alpha1, args.alpha2, args.alpha3, args.alpha4 = 0.4, 0.2, 0.2, 0.2
        args.max_lag_corr = 10
        args.horizons = [1,5,10]
        args.granger_lag = 5

        build_edges_main(args)

    # sanity check（不改逻辑，只检查）
    df = pd.read_csv(prices)
    node_cols = list(df.columns[1:])
    A = build_adjacency_from_edges(str(out_edges), node_cols)

    report = {
        "panel": panel,
        "edges": len(pd.read_csv(out_edges)),
        "A_shape": list(A.shape),
        "A_density": adj_density(A),
        "finite": bool(torch.isfinite(A).all()),
        "symmetric": bool(torch.allclose(A, A.T, atol=1e-6))
    }

    rep_path = rep_dir / f"prepare_panel{panel}.json"
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"[OK] panel={panel} density={report['A_density']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--panels", nargs="+", type=int, default=[20,30,40,50])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compute_weights", action="store_true")
    args = ap.parse_args()

    for p in args.panels:
        run_one(Path(args.dataset_dir), p, args.force, args.compute_weights)


if __name__ == "__main__":
    main()

