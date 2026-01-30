# exp/01_forecasting/run_ablation_sweep.py
# -*- coding: utf-8 -*-
"""
Ablation sweep runner for GPMechSTGNN experiments.

It runs exp/01_forecasting/run_gp_mech.py over:
  - panel sizes: 20/30/40/50
  - horizons: 5/10/15
  - modes: mech / learn / prior_residual
  - seeds: e.g. 1,2,3,4,5

Outputs:
  - one CSV per config in results/
  - merged summary CSV: results/gp_mech_ablation_all.csv

Usage (from repo root):
  python exp/01_forecasting/run_ablation_sweep.py \
    --panels 20,30,40,50 \
    --horizons 5,10,15 \
    --modes mech,learn,prior_residual \
    --seeds 1,2,3,4,5 \
    --window 20 --epochs 10 --batch 32 \
    --price_dir dataset \
    --macro_path dataset/panel_macro.csv \
    --edges_dir dataset/derived \
    --out_dir results
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import pandas as pd


def _csv_list(x: str):
    return [s.strip() for s in x.split(",") if s.strip()]


def _int_list(x: str):
    return [int(s.strip()) for s in x.split(",") if s.strip()]


def run_one(cmd, dry_run: bool = False):
    print("\n" + "=" * 80)
    print("[CMD]", " ".join(cmd))
    print("=" * 80)
    if dry_run:
        return 0
    p = subprocess.run(cmd, check=False)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=str, default="20,30,40,50")
    ap.add_argument("--horizons", type=str, default="5,10,15")
    ap.add_argument("--modes", type=str, default="mech,learn,prior_residual")
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")

    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)

    ap.add_argument("--price_dir", type=str, default="dataset")
    ap.add_argument("--macro_path", type=str, default="dataset/panel_macro.csv")
    ap.add_argument("--edges_dir", type=str, default="dataset/derived")

    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--python", type=str, default=sys.executable)

    ap.add_argument("--skip_existing", action="store_true", help="skip runs if output CSV exists")
    ap.add_argument("--dry_run", action="store_true", help="print commands only")

    args = ap.parse_args()

    panels = _int_list(args.panels)
    horizons = _int_list(args.horizons)
    modes = _csv_list(args.modes)
    seeds = ",".join(str(s) for s in _int_list(args.seeds))

    repo_root = Path(__file__).resolve().parents[2]  # .../anonymous22
    runner_py = repo_root / "exp" / "01_forecasting" / "run_gp_mech.py"
    if not runner_py.exists():
        raise FileNotFoundError(f"Cannot find {runner_py}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csv_paths = []
    failed = []

    for p in panels:
        price_path = Path(args.price_dir) / f"panel_{p}.csv"
        if not price_path.exists():
            print(f"[WARN] Missing price file: {price_path} -> skip panel {p}")
            continue

        # edges file only needed for mech / prior_residual
        edges_path = Path(args.edges_dir) / f"edges_candidates_{p}.csv"
        if not edges_path.exists():
            print(f"[WARN] Missing edges file: {edges_path} (needed for mech/prior_residual)")

        for h in horizons:
            for mode in modes:
                tag = "hybrid" if mode == "prior_residual" else mode
                out_csv = out_dir / f"gp_mech_stgnn_{tag}_panel{p}_h{h}.csv"

                if args.skip_existing and out_csv.exists():
                    print(f"[SKIP] exists: {out_csv}")
                    all_csv_paths.append(out_csv)
                    continue

                cmd = [
                    args.python, str(runner_py),
                    "--mode", mode,
                    "--price_path", str(price_path),
                    "--macro_path", str(args.macro_path),
                    "--window", str(args.window),
                    "--horizon", str(h),
                    "--epochs", str(args.epochs),
                    "--batch", str(args.batch),
                    "--seeds", seeds,
                    "--out_csv", str(out_csv),
                ]

                # only include edges_path when needed
                if mode in ("mech", "prior_residual"):
                    if edges_path.exists():
                        cmd += ["--edges_path", str(edges_path)]
                    else:
                        print(f"[WARN] edges missing for {mode}: {edges_path} -> this run will likely fail")

                code = run_one(cmd, dry_run=args.dry_run)
                if code != 0:
                    failed.append((p, h, mode, str(out_csv)))
                else:
                    all_csv_paths.append(out_csv)

    # merge results
    merged_path = out_dir / "gp_mech_ablation_all.csv"
    if not args.dry_run:
        frames = []
        for fp in all_csv_paths:
            if Path(fp).exists():
                try:
                    df = pd.read_csv(fp)
                    # helpful extra columns
                    df["panel"] = int(str(fp).split("panel")[1].split("_h")[0])
                    df["horizon"] = int(str(fp).split("_h")[1].split(".csv")[0])
                    frames.append(df)
                except Exception as e:
                    print(f"[WARN] Failed reading {fp}: {repr(e)}")

        if frames:
            big = pd.concat(frames, axis=0, ignore_index=True)
            big.to_csv(merged_path, index=False)
            print(f"\n[OK] Merged results saved to: {merged_path}")
        else:
            print("\n[WARN] No result CSVs found to merge.")

    if failed:
        print("\n[FAILED RUNS]")
        for p, h, mode, out_csv in failed:
            print(f"  panel={p} horizon={h} mode={mode} -> {out_csv}")
        sys.exit(1)

    print("\n[DONE] All scheduled runs finished.")


if __name__ == "__main__":
    main()
