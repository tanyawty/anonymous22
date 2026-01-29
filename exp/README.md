# Experiments (`exp/`)

This folder contains **four experiment suites**, aligned with the paper:

1. **Forecasting performance** (ours vs baselines): `exp/01_forecasting/`
2. **Ablations** (learn / mech / prior_residual, random prior, etc.): `exp/02_ablation/`
3. **Graph-structure stability** (Frobenius / Cosine / Top-k Jaccard): `exp/03_graph_stability/`
4. **Interpretability case studies** (oil chain, triptych export): `exp/04_interpretability/`

All experiment scripts save outputs under `results/` with a unified structure.

> Tip: run each suite from the **repo root** so imports like `data_provider`, `models`, and `utils` work.
