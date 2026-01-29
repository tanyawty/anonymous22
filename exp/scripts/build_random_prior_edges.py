# build_random_prior_edges.py
import argparse
import numpy as np
import pandas as pd
import random


def build_random_prior_graph(
    nodes,
    num_edges,
    seed=42,
    allow_self_loops=False,
):
    rng = random.Random(seed)

    all_pairs = [
        (i, j)
        for i in nodes
        for j in nodes
        if allow_self_loops or i != j
    ]

    rng.shuffle(all_pairs)
    selected = all_pairs[:num_edges]

    rows = []
    for src, dst in selected:
        rows.append(
            dict(
                source=src,
                target=dst,
                template="T0_random_prior",
                rule="random_connection",
                lag_hint=rng.choice([1, 3, 5]),
                lag_hint_real=np.nan,
                notes="randomly connected prior edge",
                w=1.0,
                s_LP=np.nan,
                s_co=np.nan,
                s_gr=np.nan,
                s_corr=np.nan,
            )
        )

    return pd.DataFrame(rows)


def main(args):
    df = pd.read_csv(args.prices)

    # ===== 核心：只取资产列 =====
    date_cols = {"date", "Date", "DATE"}
    asset_cols = [c for c in df.columns if c not in date_cols]

    if len(asset_cols) < 2:
        raise ValueError("Not enough asset columns detected.")

    N = len(asset_cols)
    print(f"[INFO] Detected panel size = {N}")

    if args.edges_per_node is not None:
        num_edges = N * args.edges_per_node
    else:
        num_edges = args.num_edges

    print(f"[INFO] Generating {num_edges} random prior edges")

    edges = build_random_prior_graph(
        nodes=asset_cols,
        num_edges=num_edges,
        seed=args.seed,
        allow_self_loops=args.self_loop,
    )

    edges.to_csv(args.out_edges, index=False)
    print(f"[OK] Random prior graph saved to {args.out_edges}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out_edges", required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_edges", type=int, default=200)
    ap.add_argument("--edges_per_node", type=int, default=None)
    ap.add_argument("--self_loop", action="store_true")

    args = ap.parse_args()
    main(args)
