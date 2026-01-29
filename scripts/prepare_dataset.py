#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_one(dataset_dir: Path, panel: int, force: bool, compute_weights: bool) -> Path:
    root = repo_root()

    # ✅ 全部用绝对路径，避免 cwd/相对路径差异
    meta = (root / dataset_dir / f"metadata_{panel}.csv").resolve()
    prices = (root / dataset_dir / f"panel_{panel}.csv").resolve()
    macro = (root / dataset_dir / "panel_macro.csv").resolve()

    out_dir = (root / dataset_dir / "derived").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_edges = (out_dir / f"edges_candidates_{panel}.csv").resolve()

    if out_edges.exists() and (not force):
        print(f"[SKIP] {out_edges}")
        return out_edges

    # ✅ 逐字复制你的“金标准命令”，只是路径改为绝对路径
    cmd = [
        "python",
        str((root / "scripts" / "build_edges_from_metadata.py").resolve()),
        "--metadata", str(meta),
        "--prices", str(prices),
        "--macro", str(macro),
        "--out_edges", str(out_edges),
    ]
    if compute_weights:
        cmd.append("--compute_weights")

    print("[CMD]", " ".join(cmd))

    # ✅ 强制在 repo 根目录运行，避免脚本内部相对路径依赖
    subprocess.check_call(cmd, cwd=str(root))

    return out_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset")
    ap.add_argument("--panels", type=int, nargs="+", default=[20, 30, 40, 50])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compute_weights", action="store_true")
    args = ap.parse_args()

    for p in args.panels:
        out = run_one(Path(args.dataset_dir), int(p), args.force, args.compute_weights)
        print(f"[OK] panel={p} -> {out}")


if __name__ == "__main__":
    main()

