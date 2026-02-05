# Experiments (`exp/`)

This folder contains **five experiment suites**, aligned with the paper:

1. **Forecasting performance** (ours vs baselines)
2. **Ablations** (learn / mech / prior_residual, random prior, etc.)
3. **Graph-structure stability** (Frobenius / Cosine / Top-k Jaccard):graph_stability_viz.py and export_triptych_topk.py
4. **Interpretability case studies** (oil chain, triptych export):case_oil_study.py
5. **Trading Validation** (Sharp Ratio) py file is in run_baslines(run_compare_sharpe_MAGN_v3.py)
```bash
# Interpretability case studies
python case_oil_study.py \
  --panel_prices panel_20.csv \
  --panel_macro  panel_macro.csv \
  --edges edges_candidates_20.csv \
  --ckpt ckpt_panel20_h5_seed1.pt \
  --window_size 30 --horizon 5 \
  --out_dir out_case_oil

# Graph-structure stability
python build_random_prior_edges.py \
  --prices panel_40.csv \  # change it to 20,30,40,50
  --num_edges 292 \  # this should match with the mech graph edges number,please change it
  --seed 42 \
  --out_edges edges_random_prior_40.csv  # change it to 20,30,40,50

for SEED in [1,2,3,4,5]:
  print(f"============= SEED={SEED} ==========")
  python graph_stability_viz.py \
  --model_py run.py \
  --panel_prices panel_40.csv \
  --panel_macro panel_macro.csv \
  --edges edges_random_prior_40.csv \
  --mode_list learn prior_residual \
  --seeds {SEED} \
  --epochs 50 \
  --out_dir graph_stability_panel40_h5_r \
  --probe test
```
