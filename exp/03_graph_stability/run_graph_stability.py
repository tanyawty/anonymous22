#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 03: Graph-structure stability

This script is a thin wrapper around the original `graph_stability_viz.py`,
but placed under `exp/03_graph_stability/` for reproducibility.

Usage:
  python exp/03_graph_stability/run_graph_stability.py --config exp/03_graph_stability/configs/stability_panel40_h5.yaml

It will save figures / metrics under the configured out_dir.
"""
from __future__ import annotations
import argparse
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("Please `pip install pyyaml` to use YAML configs.")

import subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--python", default=None)
    args = ap.parse_args()

    py = args.python or sys.executable
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    cmd = [
        py, "-u", "graph_stability_viz.py",
        "--panel_prices", cfg["panel_prices"],
        "--panel_macro",  cfg["panel_macro"],
        "--edges",        cfg["edges"],
        "--window_size",  str(cfg.get("window_size", 20)),
        "--horizon",      str(cfg.get("horizon", 5)),
        "--epochs",       str(cfg.get("epochs", 50)),
        "--batch_size",   str(cfg.get("batch_size", 64)),
        "--out_dir",      str(cfg.get("out_dir", "results/graph_stability")),
        "--mode_list",
    ] + [str(x) for x in cfg.get("mode_list", ["learn","prior_residual"])] + [
        "--seeds",
    ] + [str(s) for s in cfg.get("seeds", [1,2,3,4,5])]

    print("[CMD]", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
