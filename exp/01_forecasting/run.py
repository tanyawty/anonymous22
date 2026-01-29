#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 01: Forecasting performance (ours vs baselines)

- Runs each method across seeds
- Saves per-run stdout logs + metrics.json (parsed from stdout when needed)

Usage:
  python exp/01_forecasting/run.py --config exp/01_forecasting/configs/panel40_h5.yaml

Outputs:
  results/forecasting/<method>/panel<N>_h<H>/seed<S>/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("Please `pip install pyyaml` to use YAML configs.")

from exp.common.runner import run_and_capture, parse_gp_stdout_metrics, save_json

def build_cmd(py: str, script: str, base_args: dict, extra_args: dict) -> list[str]:
    cmd = [py, "-u", script]
    # deterministic ordering
    for k, v in {**base_args, **extra_args}.items():
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
    ap.add_argument("--python", default=None, help="Python executable (default: current)")
    ap.add_argument("--out_root", default="results/forecasting", type=str)
    args = ap.parse_args()

    py = args.python or __import__("sys").executable
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    panel_prices = cfg["panel_prices"]
    panel_macro  = cfg["panel_macro"]
    edges        = cfg.get("edges")
    window_size  = int(cfg["window_size"])
    horizon      = int(cfg["horizon"])
    epochs       = int(cfg.get("epochs", 50))
    batch_size   = int(cfg.get("batch_size", 64))
    lr           = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    seeds        = cfg["seeds"]

    out_root = Path(args.out_root)
    methods = cfg["methods"]

    # Base args (used by our main script; baselines will ignore unknown args if they parse_known_args)
    base_args = dict(
        panel_prices=panel_prices,
        panel_macro=panel_macro,
        window_size=window_size,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )
    if edges:
        base_args["edges"] = edges
        base_args["weight_col"] = "w"
        base_args["default_weight"] = 1.0

    for method_name, mcfg in methods.items():
        script = mcfg["script"]
        extra = mcfg.get("args", {}) or {}
        for seed in seeds:
            run_dir = out_root / method_name / f"panel{cfg.get('panel_size','') or ''}_h{horizon}".replace("panel_h", "panel_h") / f"seed{seed}"
            # if panel_size not specified, infer N from filename like panel_40.csv
            if "panel_size" in cfg:
                panelN = int(cfg["panel_size"])
            else:
                import re
                mm = re.search(r"panel_(\d+)", str(panel_prices))
                panelN = int(mm.group(1)) if mm else -1
            run_dir = out_root / method_name / f"panel{panelN}_h{horizon}" / f"seed{seed}"

            log_path = run_dir / "stdout.log"

            cmd_args = dict(base_args)
            cmd_args["seed"] = int(seed)
            cmd_args.update(extra)

            cmd = build_cmd(py, script, cmd_args, {})
            rc, stdout_text = run_and_capture(cmd, log_path)

            # Metrics handling:
            metrics = None
            # If the method already writes metrics.json, prefer it.
            mj = run_dir / "metrics.json"
            if mj.exists():
                try:
                    metrics = json.loads(mj.read_text(encoding="utf-8")).get("test_metrics")
                except Exception:
                    metrics = None
            if metrics is None:
                metrics = parse_gp_stdout_metrics(stdout_text)

            report = {
                "method": method_name,
                "script": script,
                "seed": int(seed),
                "return_code": int(rc),
                "panel_prices": panel_prices,
                "panel_macro": panel_macro,
                "edges": edges,
                "window_size": window_size,
                "horizon": horizon,
                "test_metrics": metrics,
            }
            save_json(report, run_dir / "metrics.json")

if __name__ == "__main__":
    main()
