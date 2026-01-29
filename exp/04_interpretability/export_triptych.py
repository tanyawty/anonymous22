#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 04b: Triptych export (A_mech / A_learn / A_hyb top-k)

Wrapper that runs `export_triptych_topk.py`.

Usage:
  python exp/04_interpretability/export_triptych.py --ckpt <ckpt> --edges <edges> --out_dir <dir>
"""
from __future__ import annotations
import argparse
import subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--panel_prices", required=True)
    ap.add_argument("--panel_macro", required=True)
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="results/interpretability/triptych")
    args = ap.parse_args()

    cmd = [
        sys.executable, "-u", "export_triptych_topk.py",
        "--panel_prices", args.panel_prices,
        "--panel_macro",  args.panel_macro,
        "--edges",        args.edges,
        "--ckpt",         args.ckpt,
        "--window_size",  str(args.window_size),
        "--horizon",      str(args.horizon),
        "--out_dir",      args.out_dir,
    ]
    print("[CMD]", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
