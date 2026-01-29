
"""
Graph-structure stability experiment + visualization for gp_mech_multitask_stgnn.py

What it does
- Trains the same model across multiple seeds for two graph modes:
    (1) learn            -> uses A_learn (purely learned graph)
    (2) prior_residual   -> uses A_hyb = gamma*A_mech + (1-gamma)*A_learn
- Extracts a representative graph for each seed by averaging A_learn across a probe loader (val/test)
- Computes cross-seed stability metrics:
    * normalized Frobenius drift
    * cosine similarity
    * top-k edge Jaccard
- Visualizes graphs (heatmaps) + optionally a top-k network plot (requires networkx)
- Saves everything under --out_dir

Usage (example)
python graph_stability_viz.py \
  --panel_prices panel_40.csv --panel_macro panel_macro.csv --edges edges_candidates_40.csv \
  --mode_list learn prior_residual --seeds 1 2 3 4 5 \
  --epochs 50 --batch_size 64 --window_size 20 --horizon 5 \
  --out_dir runs/graph_stability_panel40_h5

Notes
- This script DOES NOT modify your original model file.
- It imports gp_mech_multitask_stgnn.py as a module.
"""

from __future__ import annotations
import argparse
import itertools
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- import your existing model code (no edits needed) ---
import importlib.util

def import_model_module(model_py: str):
    spec = importlib.util.spec_from_file_location("gp_mech_multitask_stgnn", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model module from: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- metrics ----------------
def frobenius(A: np.ndarray, B: np.ndarray, normalize: bool = True, eps: float = 1e-12) -> float:
    diff = A - B
    d = float(np.linalg.norm(diff, ord="fro"))
    if not normalize:
        return d
    denom = float(np.linalg.norm(A, ord="fro")) + eps
    return float(d / denom)

def cosine_sim(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    a = A.reshape(-1)
    b = B.reshape(-1)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(num / den)

def topk_edge_set(A: np.ndarray, k: int, sym_upper: bool = True, remove_diag: bool = True) -> set[Tuple[int,int]]:
    N = A.shape[0]
    M = A.copy()

    if remove_diag:
        np.fill_diagonal(M, -np.inf)

    if sym_upper:
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        M = np.where(mask, M, -np.inf)

    flat_idx = np.argsort(M.reshape(-1))[::-1]  # desc
    edges = set()
    for idx in flat_idx:
        if len(edges) >= k:
            break
        i = idx // N
        j = idx % N
        if np.isfinite(M[i, j]):
            edges.add((int(i), int(j)))
    return edges

def jaccard(E1: set, E2: set, eps: float = 1e-12) -> float:
    inter = len(E1 & E2)
    uni = len(E1 | E2)
    return float(inter / (uni + eps))


# ---------------- plotting helpers ----------------
def save_heatmap(A: np.ndarray, title: str, path: str):
    plt.figure()
    plt.imshow(A, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_similarity_matrix(mat: np.ndarray, title: str, path: str, vmin=None, vmax=None):
    plt.figure()
    plt.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_topk_network(A: np.ndarray, k: int, title: str, path: str):
    try:
        import networkx as nx
    except Exception:
        # networkx not available; silently skip
        return False

    N = A.shape[0]
    E = topk_edge_set(A, k=k, sym_upper=True, remove_diag=True)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for (i, j) in E:
        w = float(A[i, j])
        G.add_edge(i, j, weight=w)

    # simple layout
    pos = nx.spring_layout(G, seed=1)

    plt.figure()
    nx.draw_networkx_nodes(G, pos, node_size=120)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


# ---------------- graph extraction ----------------
@torch.no_grad()
def extract_mean_graph(model, loader, A_mech: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, float]:
    """
    Extract representative learned graph by averaging A_learn over batches.
    Returns:
      A_learn_mean: (N,N) numpy
      gamma_mean: float (last/mean gamma; in learn mode usually 0)
    """
    model.eval()
    A_mech = A_mech.to(device)

    A_sum = None
    cnt = 0
    gammas: List[float] = []

    for x_seq, _ in loader:
        x_seq = x_seq.to(device)
        _, A_learn, gamma = model(x_seq, A_mech)

        A_batch = A_learn.mean(dim=0)  # (N,N)
        if A_sum is None:
            A_sum = A_batch.detach().cpu().clone()
        else:
            A_sum += A_batch.detach().cpu()
        cnt += 1

        if isinstance(gamma, torch.Tensor):
            gammas.append(float(gamma.detach().cpu().item()))
        else:
            gammas.append(float(gamma))

    A_learn_mean = (A_sum / max(cnt, 1)).numpy()
    gamma_mean = float(np.mean(gammas)) if gammas else 0.0
    return A_learn_mean, gamma_mean


def build_split_datasets(mod, price_df, returns_df, macro_df, node_cols, macro_cols, args):
    """
    Matches your original time split logic in gp_mech_multitask_stgnn.py main().
    """
    T = len(returns_df)
    train_end = int(T * args.train_ratio)
    val_end   = int(T * (args.train_ratio + args.val_ratio))

    ret_train = returns_df.iloc[:train_end]
    ret_val   = returns_df.iloc[train_end - args.window_size - 1:val_end]
    ret_test  = returns_df.iloc[val_end - args.window_size - 1:]

    price_train = price_df.loc[ret_train.index]
    price_val   = price_df.loc[ret_val.index]
    price_test  = price_df.loc[ret_test.index]

    macro_train = macro_df.loc[ret_train.index]
    macro_val   = macro_df.loc[ret_val.index]
    macro_test  = macro_df.loc[ret_test.index]

    ds_train = mod.GPMultiTaskDataset(ret_train, price_train, macro_train,
                                      window_size=args.window_size, horizon=args.horizon)
    ds_val = mod.GPMultiTaskDataset(ret_val, price_val, macro_val,
                                    window_size=args.window_size, horizon=args.horizon)
    ds_test = mod.GPMultiTaskDataset(ret_test, price_test, macro_test,
                                     window_size=args.window_size, horizon=args.horizon)

    return ds_train, ds_val, ds_test


@dataclass
class RunResult:
    seed: int
    mode: str
    gamma: float
    A: np.ndarray  # representative graph for this mode (Alearn or Ahyb)


def train_and_extract_one(mod, args, seed: int, mode: str, probe: str, out_dir: str) -> RunResult:
    # seed + device
    mod.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu")

    # data
    price_df, returns_df, macro_df, node_cols, macro_cols = mod.load_panel_from_two_files(
        args.panel_prices, args.panel_macro
    )

    # mech graph
    A_mech = mod.build_adjacency_from_edges(
        args.edges,
        node_cols,
        weight_col=args.weight_col,
        default_weight=args.default_weight
    ).float()

    # datasets & loaders
    ds_train, ds_val, ds_test = build_split_datasets(mod, price_df, returns_df, macro_df, node_cols, macro_cols, args)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    probe_loader = val_loader if probe == "val" else test_loader


    # --- model ---
    # 从dataset样本自动推断 in_dim（最稳，避免你以后改特征维度又崩）
    x0, _ = ds_train[0]  # x0 shape: (L, N, F)
    in_dim = int(x0.shape[-1])  # F

    model = mod.MechAware_GP_STGNN_MultiTask(
        num_nodes=len(node_cols),
        in_dim=in_dim,
        mode=mode,
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        graph_hidden=args.graph_hidden,
        horizon=args.horizon,
        out_dim=1,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_loss, _, _, _, _ = mod.train_one_epoch_seq(model, train_loader, A_mech, opt, device)
        va_loss, _, _, _, va_gamma = mod.eval_one_epoch_seq(model, val_loader, A_mech, device)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if args.log_every > 0 and (ep % args.log_every == 0 or ep == 1 or ep == args.epochs):
            print(f"[{mode} seed={seed}] ep={ep:03d} tr={tr_loss:.6f} va={va_loss:.6f} gamma={va_gamma:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # extract graph
    A_learn_mean, gamma_mean = extract_mean_graph(model, probe_loader, A_mech, device)

    A_mech_np = A_mech.numpy()
    if mode == "prior_residual":
        A_out = gamma_mean * A_mech_np + (1.0 - gamma_mean) * A_learn_mean
    elif mode == "mech":
        A_out = A_mech_np
    else:
        A_out = A_learn_mean

    # save per-seed artifacts
    seed_dir = os.path.join(out_dir, mode, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    np.save(os.path.join(seed_dir, "A.npy"), A_out)
    np.save(os.path.join(seed_dir, "A_learn_mean.npy"), A_learn_mean)
    np.save(os.path.join(seed_dir, "A_mech.npy"), A_mech_np)
    with open(os.path.join(seed_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed={seed}\nmode={mode}\nprobe={probe}\ngamma_mean={gamma_mean:.8f}\n")

    # visualizations for this seed
    save_heatmap(A_out, f"{mode} seed={seed} (probe={probe})", os.path.join(seed_dir, "heatmap.png"))
    if args.topk_network > 0:
        ok = save_topk_network(A_out, k=args.topk_network, title=f"{mode} seed={seed} topk={args.topk_network}",
                               path=os.path.join(seed_dir, f"topk_{args.topk_network}.png"))
        if (not ok) and (seed == args.seeds[0]):
            print("[WARN] networkx not available; skip top-k network plots.")

    return RunResult(seed=seed, mode=mode, gamma=gamma_mean, A=A_out)


def summarize_stability(results: List[RunResult], topk: int) -> Dict[str, object]:
    seeds = [r.seed for r in results]
    A_list = [r.A for r in results]
    n = len(A_list)

    # pairwise matrices
    fro_mat = np.zeros((n, n), dtype=float)
    cos_mat = np.zeros((n, n), dtype=float)
    jac_mat = np.zeros((n, n), dtype=float)

    fro_vals, cos_vals, jac_vals = [], [], []

    for i in range(n):
        for j in range(n):
            if i == j:
                fro_mat[i, j] = 0.0
                cos_mat[i, j] = 1.0
                jac_mat[i, j] = 1.0
            elif i < j:
                A = A_list[i]; B = A_list[j]
                f = frobenius(A, B, normalize=True)
                c = cosine_sim(A, B)
                E1 = topk_edge_set(A, k=topk, sym_upper=True, remove_diag=True)
                E2 = topk_edge_set(B, k=topk, sym_upper=True, remove_diag=True)
                jv = jaccard(E1, E2)

                fro_mat[i, j] = fro_mat[j, i] = f
                cos_mat[i, j] = cos_mat[j, i] = c
                jac_mat[i, j] = jac_mat[j, i] = jv

                fro_vals.append(f); cos_vals.append(c); jac_vals.append(jv)

    def ms(x): 
        return float(np.mean(x)) if x else float("nan"), float(np.std(x)) if x else float("nan")

    return {
        "seeds": seeds,
        "topk": int(topk),
        "fro_norm_mean": ms(fro_vals)[0],
        "fro_norm_std": ms(fro_vals)[1],
        "cos_mean": ms(cos_vals)[0],
        "cos_std": ms(cos_vals)[1],
        "jacc_mean": ms(jac_vals)[0],
        "jacc_std": ms(jac_vals)[1],
        "fro_mat": fro_mat,
        "cos_mat": cos_mat,
        "jac_mat": jac_mat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_py", type=str, default="gp_mech_multitask_stgnn.py",
                    help="Path to your original model script.")
    ap.add_argument("--panel_prices", type=str, default="panel_40.csv")
    ap.add_argument("--panel_macro",  type=str, default="panel_macro.csv")
    ap.add_argument("--edges",        type=str, default="edges_candidates_40.csv")
    ap.add_argument("--weight_col",   type=str, default="w")
    ap.add_argument("--default_weight", type=float, default=1.0)

    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--horizon",     type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio",   type=float, default=0.15)

    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--epochs",      type=int, default=50)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight_decay",type=float, default=1e-5)

    ap.add_argument("--gcn_hidden",  type=int, default=32)
    ap.add_argument("--gru_hidden",  type=int, default=64)
    ap.add_argument("--graph_hidden",type=int, default=32)
    ap.add_argument("--lambda_gap",  type=float, default=1.0)

    ap.add_argument("--mode_list", nargs="+", default=["learn", "prior_residual"],
                    choices=["learn", "mech", "prior_residual"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1,2,3,4,5])
    ap.add_argument("--probe", choices=["val", "test"], default="val",
                    help="Which split to use to extract representative graphs.")
    ap.add_argument("--topk_ratio", type=float, default=0.10,
                    help="Top-k ratio on upper-tri edges for Jaccard/network.")
    ap.add_argument("--topk_network", type=int, default=0,
                    help="If >0, additionally save a top-k network plot with this k.")
    ap.add_argument("--out_dir", type=str, default="graph_stability_out")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # import your module
    mod = import_model_module(args.model_py)

    # we need N to compute topk (load once)
    price_df, returns_df, macro_df, node_cols, macro_cols = mod.load_panel_from_two_files(
        args.panel_prices, args.panel_macro
    )
    N = len(node_cols)
    num_upper = N * (N - 1) // 2
    topk = max(1, int(args.topk_ratio * num_upper))

    # run
    all_summaries = {}
    for mode in args.mode_list:
        print(f"\n===== Running mode={mode} seeds={args.seeds} probe={args.probe} =====")
        results: List[RunResult] = []
        for sd in args.seeds:
            rr = train_and_extract_one(mod, args, seed=sd, mode=mode, probe=args.probe, out_dir=args.out_dir)
            results.append(rr)

        summ = summarize_stability(results, topk=topk)
        all_summaries[mode] = summ

        mode_dir = os.path.join(args.out_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # save similarity matrices
        save_similarity_matrix(summ["fro_mat"], f"{mode} normalized Frobenius drift", os.path.join(mode_dir, "pairwise_fro.png"))
        save_similarity_matrix(summ["cos_mat"], f"{mode} cosine similarity", os.path.join(mode_dir, "pairwise_cos.png"), vmin=-1, vmax=1)
        save_similarity_matrix(summ["jac_mat"], f"{mode} topk Jaccard (k={topk})", os.path.join(mode_dir, "pairwise_jacc.png"), vmin=0, vmax=1)

        # write summary txt
        with open(os.path.join(mode_dir, "stability_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"mode={mode}\nseeds={summ['seeds']}\nprobe={args.probe}\n")
            f.write(f"topk_ratio={args.topk_ratio}  topk={topk}\n")
            f.write(f"Fro(norm) mean±std = {summ['fro_norm_mean']:.6f} ± {summ['fro_norm_std']:.6f}\n")
            f.write(f"Cosine   mean±std = {summ['cos_mean']:.6f} ± {summ['cos_std']:.6f}\n")
            f.write(f"Jaccard  mean±std = {summ['jacc_mean']:.6f} ± {summ['jacc_std']:.6f}\n")

    # combined quick comparison table (txt)
    with open(os.path.join(args.out_dir, "COMPARE_ALEARN_vs_AHYB.txt"), "w", encoding="utf-8") as f:
        f.write("Graph stability (pairwise over seeds)\n")
        f.write(f"probe={args.probe}  topk_ratio={args.topk_ratio}  topk={topk}\n\n")
        for mode in args.mode_list:
            s = all_summaries[mode]
            f.write(f"[{mode}]\n")
            f.write(f"Fro(norm): {s['fro_norm_mean']:.6f} ± {s['fro_norm_std']:.6f}\n")
            f.write(f"Cosine:    {s['cos_mean']:.6f} ± {s['cos_std']:.6f}\n")
            f.write(f"Jaccard:   {s['jacc_mean']:.6f} ± {s['jacc_std']:.6f}\n\n")

    print("\nDone.")
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
