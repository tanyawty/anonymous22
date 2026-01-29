#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 04a: Oil-chain interpretability case study

Wrapper that runs `case_oil_study.py` with a YAML config.

Usage:
  python exp/04_interpretability/run_oil_case.py --config exp/04_interpretability/configs/oil_case.yaml
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
        py, "-u", "case_oil_study.py",
        "--panel_prices", cfg["panel_prices"],
        "--panel_macro",  cfg["panel_macro"],
        "--edges",        cfg["edges"],
        "--ckpt",         cfg["ckpt"],
        "--window_size",  str(cfg.get("window_size", 30)),
        "--horizon",      str(cfg.get("horizon", 5)),
        "--out_dir",      str(cfg.get("out_dir", "results/interpretability/oil_case")),
    ]
    print("[CMD]", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
