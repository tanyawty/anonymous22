#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

GP_METRIC_RE = {
    "PF": re.compile(r"PF-task:\s*MAE\s*=\s*([0-9.eE+-]+)\s*RMSE\s*=\s*([0-9.eE+-]+)"),
    "MA": re.compile(r"MA-task:\s*MAE\s*=\s*([0-9.eE+-]+)\s*RMSE\s*=\s*([0-9.eE+-]+)"),
    "GAP": re.compile(r"GAP-task:\s*MAE\s*=\s*([0-9.eE+-]+)\s*RMSE\s*=\s*([0-9.eE+-]+)"),
}

def run_and_capture(cmd, log_path: Path) -> Tuple[int, str]:
    """
    Run a command, tee stdout to console, and write full stdout to log_path.
    Returns (return_code, full_stdout_text).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    chunks = []
    with log_path.open("w", encoding="utf-8") as f:
        for line in p.stdout:
            print(line, end="")
            f.write(line)
            chunks.append(line)
    rc = p.wait()
    return rc, "".join(chunks)

def parse_gp_stdout_metrics(stdout_text: str) -> Dict[str, Dict[str, float]]:
    """
    Parse gp_mech_multitask_stgnn.py stdout, expecting lines like:
      PF-task:  MAE=...  RMSE=...
      MA-task:  MAE=...  RMSE=...
      GAP-task: MAE=...  RMSE=...
    """
    out = {}
    for k, rx in GP_METRIC_RE.items():
        m = rx.search(stdout_text)
        if m:
            out[k] = {"MAE": float(m.group(1)), "RMSE": float(m.group(2))}
    return out

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
