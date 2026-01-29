#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate forecasting results across seeds:
- Reads results/forecasting/<method>/panel<N>_h<H>/seed<S>/metrics.json
- Produces summary.csv and (optional) LaTeX table snippets.

Usage:
  python exp/01_forecasting/aggregate.py --results_root results/forecasting --panel 40 --horizon 5
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def mean_std_cv(x):
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=0))
    cv = float(s / (m + 1e-12) * 100.0)
    return m, s, cv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results/forecasting")
    ap.add_argument("--panel", type=int, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    args = ap.parse_args()

    root = Path(args.results_root)
    rows = []
    for method_dir in sorted(root.iterdir()):
        panel_dir = method_dir / f"panel{args.panel}_h{args.horizon}"
        if not panel_dir.exists():
            continue
        for seed_dir in sorted(panel_dir.glob("seed*/metrics.json")):
            j = json.loads(seed_dir.read_text(encoding="utf-8"))
            met = j.get("test_metrics", {})
            for task in ["PF", "MA", "GAP"]:
                if task in met:
                    rows.append({
                        "method": method_dir.name,
                        "seed": int(j.get("seed", -1)),
                        "task": task,
                        "MAE": met[task].get("MAE", np.nan),
                        "RMSE": met[task].get("RMSE", np.nan),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No results found. Did you run exp/01_forecasting/run.py first?")

    out_rows = []
    for (method, task), g in df.groupby(["method", "task"]):
        mae_m, mae_s, mae_cv = mean_std_cv(g["MAE"])
        rmse_m, rmse_s, rmse_cv = mean_std_cv(g["RMSE"])
        out_rows.append({
            "method": method,
            "task": task,
            "MAE_mean": mae_m, "MAE_std": mae_s, "MAE_CV%": mae_cv,
            "RMSE_mean": rmse_m, "RMSE_std": rmse_s, "RMSE_CV%": rmse_cv,
            "n_seeds": int(len(g)),
        })

    out = pd.DataFrame(out_rows).sort_values(["task", "method"])
    out_path = root / f"summary_panel{args.panel}_h{args.horizon}.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
