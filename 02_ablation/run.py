#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 02: Ablations

Runs ablations defined in YAML and saves parsed metrics.json per seed.

Usage:
  python exp/02_ablation/run.py --config exp/02_ablation/configs/ablation_panel40_h5.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("Please `pip install pyyaml` to use YAML configs.")

from exp.common.runner import run_and_capture, parse_gp_stdout_metrics, save_json

def build_cmd(py, script, args: dict):
    cmd = [py, "-u", script]
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--python", default=None)
    ap.add_argument("--out_root", default="results/ablation", type=str)
    args = ap.parse_args()

    py = args.python or __import__("sys").executable
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    base_args = dict(
        panel_prices=cfg["panel_prices"],
        panel_macro=cfg["panel_macro"],
        edges=cfg.get("edges"),
        weight_col="w",
        default_weight=1.0,
        window_size=int(cfg["window_size"]),
        horizon=int(cfg["horizon"]),
        epochs=int(cfg.get("epochs", 50)),
        batch_size=int(cfg.get("batch_size", 64)),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        cpu=False,
    )
    out_root = Path(args.out_root)
    seeds = cfg["seeds"]
    ablations = cfg["ablations"]

    # infer panel N
    import re
    mm = re.search(r"panel_(\d+)", str(cfg["panel_prices"]))
    panelN = int(mm.group(1)) if mm else -1
    horizon = int(cfg["horizon"])

    for ab_name, ab_cfg in ablations.items():
        script = ab_cfg["script"]
        extra  = ab_cfg.get("args", {}) or {}
        for seed in seeds:
            run_dir = out_root / ab_name / f"panel{panelN}_h{horizon}" / f"seed{seed}"
            log_path = run_dir / "stdout.log"

            cmd_args = dict(base_args)
            cmd_args["seed"] = int(seed)
            cmd_args.update(extra)

            cmd = build_cmd(py, script, cmd_args)
            rc, stdout_text = run_and_capture(cmd, log_path)
            metrics = parse_gp_stdout_metrics(stdout_text)

            report = {
                "ablation": ab_name,
                "seed": int(seed),
                "return_code": int(rc),
                "horizon": horizon,
                "panel": panelN,
                "test_metrics": metrics,
                "args": cmd_args,
            }
            save_json(report, run_dir / "metrics.json")

if __name__ == "__main__":
    main()
