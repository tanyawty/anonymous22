# Mechanism‑Anchored Graph Learning for Stable Commodity Futures Forecasting

This repository provides the **official implementation** of the paper:

> **Mechanism‑Anchored Graph Learning for Stable Structural Modeling in Commodity Futures**
> *(submitted to KDD / IEEE ICBC)*

The project studies **structural stability and interpretability in graph‑based multivariate time‑series forecasting** under high‑noise financial environments, with a focus on **commodity futures markets**.

---

## 1. Motivation and Problem Statement

Graph neural networks (GNNs) and adaptive structure learning have become popular tools for multivariate time‑series forecasting. However, in **high‑noise, finite‑sample financial settings**, purely data‑driven graph learning suffers from a critical limitation:

> **Learned dependency graphs are highly unstable across random seeds, data perturbations, and asset universes**, even when predictive accuracy appears strong.

This instability undermines:

* reproducibility of empirical results,
* economic interpretability of learned relations, and
* credibility of structural explanations for decision‑making.

At the same time, **fixed prior graphs based on economic knowledge** provide interpretability and stability, but lack the flexibility to adapt to regime‑dependent market dynamics.

### Core Question

> How can we design a graph learning framework that **simultaneously balances predictive accuracy, structural stability, and economic interpretability** in noisy multi‑asset financial markets?

---

## 2. Key Idea

We reformulate financial graph learning as a **constrained structural learning problem**.

Instead of freely learning graph structures from data, we:

* define an **economically admissible structural hypothesis space**,
* anchor learning around a **stable mechanism‑based prior graph**, and
* allow **bounded, data‑driven residual adaptation** within this space.

Structural stability is treated as a **first‑class evaluation objective**, alongside predictive accuracy.

---

## 3. Method Overview: HPGN

We propose the **Hybrid‑Prior Graph Network (HPGN)**, a mechanism‑anchored adaptive graph learning framework.

### 3.1 Automatic Mechanism Graph Construction (AMGC)

AMGC constructs a **static, deterministic prior graph** from structured economic metadata.

The prior graph encodes **economically admissible relations**, including:

* supply‑chain dependencies (upstream → downstream),
* cross‑commodity substitution relationships,
* storability‑induced self‑dependencies,
* macroeconomic sensitivity,
* parity / arbitrage relations.

Key properties:

* no learnable parameters,
* invariant across random seeds and training runs,
* independent of forecasting tasks.

This prior defines the **structural anchor** for graph learning.

---

### 3.2 Adaptive Graph Learning

In parallel, a data‑driven adaptive graph is learned using an attention‑based mechanism over window‑level aggregated representations.

This adaptive graph captures:

* short‑term co‑movements,
* regime‑specific interactions,
* residual correlations not explained by long‑run economic structure.

On its own, this component is highly expressive but structurally unstable.

---

### 3.3 Mechanism‑Adaptive Graph Fusion

The final hybrid graph is formed via a **global gated fusion**:

[ A_{hyb} = \gamma A_{mech} + (1 - \gamma) A_{learn} ]

where:

* (A_{mech}) is the mechanism‑based prior graph,
* (A_{learn}) is the adaptive graph,
* (\gamma \in (0,1)) is a **global scalar gate**.

This design:

* explicitly bounds structural deviation from the mechanism anchor,
* limits degrees of freedom in graph learning,
* suppresses noise‑induced structural degeneration.

---

### 3.4 Spatio‑Temporal Modeling and Multi‑Task Learning

The hybrid graph is embedded into a shared spatio‑temporal GNN for **multi‑task forecasting**, including:

* **PF**: price forecasting,
* **MA**: moving‑average prediction,
* **GAP**: spread / volatility‑sensitive targets.

All tasks share the same learned structure, providing additional implicit regularization through cross‑task consistency.

---

## 4. Dataset

### 4.1 Asset Universe

We construct nested panels of commodity futures with:

* **Panel‑20 / 30 / 40 / 50 assets**,
* covering energy, metals, agriculture, softs, and livestock,
* sourced from CME, ICE, and CBOT.

Larger panels strictly contain smaller ones, enabling controlled scalability analysis.

The full sample spans **January 2015 – October 2025**.

### 4.2 Macroeconomic Variables

We incorporate five macroeconomic indicators from FRED:

* U.S. Dollar Index,
* CPI (YoY),
* Industrial Production,
* Federal Funds Rate,
* 10‑Year Treasury Yield.

Lower‑frequency series are forward‑filled to daily frequency.

### 4.3 Feature Engineering

Features include:

* log returns,
* volatility proxies,
* technical indicators,
* rolling normalization and clipping.

All preprocessing strictly respects temporal causality.

---

## 5. Experiments

### 5.1 Baselines

We compare against:

* statistical models (AR, VAR),
* deep temporal models (GRU, TCN, Transformer, PatchTST),
* graph‑based models (STGNN, FourierGNN),
* fixed‑prior graph variants.

### 5.2 Evaluation Metrics

Predictive performance:

* MAE, RMSE.

Structural stability (evaluated across random seeds):

* Frobenius distance,
* cosine similarity,
* Top‑k edge Jaccard overlap.

Stability metrics are **not optimized during training** and are treated as independent evaluation criteria.

---

## 6. Key Results

### Predictive Accuracy

HPGN achieves **best or near‑best average accuracy** across tasks and horizons, remaining competitive with strong temporal baselines.

### Structural Stability

Across all panel sizes and forecasting horizons:

* unconstrained graph learning exhibits severe structural instability,
* introducing a random prior improves stability but remains insufficient,
* **mechanism‑anchored hybrid learning consistently achieves the highest structural consistency**.

Notably, stability advantages persist as dimensionality increases.

### Interpretability

Case studies show that the hybrid graph:

* preserves economically meaningful long‑run dependencies,
* selectively amplifies relations during stress regimes,
* avoids absorbing transient noise into persistent structure.

---

## 7. Repository Structure

```text
anonymous22/
├── data_provider/        # Data loading and preprocessing
├── dataset/              # Dataset construction and metadata
├── layers/               # Custom neural network layers
├── models/               # Model architectures (HPGN and baselines)
├── exp/                  # Experiment runners and configurations
├── run_baselines/        # Scripts for baseline models
├── scripts/              # Utility and helper scripts
├── utils/                # Shared utilities
└── requirements.txt      # Dependencies
```

---

## 8. Installation

```bash
git clone https://github.com/tanyawty/anonymous22.git
cd anonymous22
pip install -r requirements.txt
```

Python >= 3.8 is recommended.

---

## 9. Running Experiments

Typical workflow:

```bash
# prepare datasets
PYTHONPATH="$PWD" python scripts/prepare_dataset.py --dataset_dir dataset --panels 20 30 40 50 --compute_weights --force
ls dataset/derived

# run main experiments
python exp/01_forecasting/run.py \
  --mode prior_residual \
  --price_path dataset/panel_20.csv \
  --macro_path dataset/panel_macro.csv \
  --edges_path dataset/derived/edges_candidates_20.csv \
  --window 20 --horizon 5 \
  --epochs 10 --batch 32 \
  --seeds 1,2,3,4,5 \
  --out_csv results/gp_mech_stgnn_mech_panel20_h5.csv

# run baselines
PRICE="dataset/panel_20.csv"
MACRO="dataset/panel_macro.csv"
WINDOW=20
HORIZON=5
EPOCHS=10
BATCH=32
SEEDS="1,2,3,4,5"

# # 1) STGNN
python run_baselines/run_baselines.py \
   --model stgnn \
   --price_path $PRICE --macro_path $MACRO \
   --window $WINDOW --horizon $HORIZON \
   --epochs $EPOCHS --batch $BATCH \
   --seeds $SEEDS \
   --out_csv results/stgnn.csv

# 2) FourierGNN
python run_baselines/run_baselines.py \
  --model fouriergnn \
  --price_path $PRICE --macro_path $MACRO \
  --window $WINDOW --horizon $HORIZON \
  --epochs $EPOCHS --batch $BATCH \
  --seeds $SEEDS \
  --out_csv results/fouriergnn.csv

# 3) PatchTST
python run_baselines/run_baselines.py \
  --model patchtst \
  --price_path $PRICE --macro_path $MACRO \
  --window $WINDOW --horizon $HORIZON \
  --epochs $EPOCHS --batch $BATCH \
  --seeds $SEEDS \
  --out_csv results/patchtst.csv

# 4) Classical baselines
for M in ["gru", "lstm", "tcn", "transformer", "mlp"]:
    print(f"=== Running {M} ===")
    python run_baselines/run_baselines.py \
      --model {M} \
      --price_path {PRICE} \
      --macro_path {MACRO} \
      --window {WINDOW} --horizon {HORIZON} \
      --epochs {EPOCHS} --batch {BATCH} \
      --seeds {SEEDS} \
      --out_csv results/{M}.csv
```

Refer to individual scripts for detailed arguments and configurations.

---

## 10. Reproducibility

* All reported results are averaged over **multiple random seeds**.
* Structural stability is evaluated independently from training.
* Mechanism graphs are fully deterministic and reproducible.

---

