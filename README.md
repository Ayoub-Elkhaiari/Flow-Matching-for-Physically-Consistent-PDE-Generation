# Flow Matching for Physically Consistent PDE Generation

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flow Matching](https://img.shields.io/badge/Flow%20Matching-Generative%20Modeling-8A2BE2)](https://arxiv.org/abs/2210.02747)
[![PDE](https://img.shields.io/badge/PDE-Burgers'%20Equation-4169E1)]()
[![Colab](https://img.shields.io/badge/Google%20Colab-Compatible-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-First%20Iteration%20%E2%80%94%20WIP-yellow)]()

**Research question:** do straight-line Optimal Transport paths in Flow Matching preserve physical consistency when generating PDE states, or does a lightweight physics-informed training penalty measurably improve it?

This project trains three generative models Flow Matching with OT paths, a diffusion baseline, and a physics-regularized Flow Matching variant to generate final-state solutions of the 1D viscous Burgers' equation from synthetically generated initial conditions. It goes beyond standard generation-quality reporting to ask whether the generated states respect an approximate physical conservation property, and whether Flow Matching's fewer-sampling-steps advantage over diffusion actually holds in practice.

> This is a first, single-seed, Colab-scale iteration on a synthetic 1D toy PDE. The pipeline is designed to extend to 2D systems, real PDE benchmarks (e.g., PDEBench), and multi-seed evaluation with additional compute see [Limitations](#️-limitations--future-work).

---

## ✨ Features

- Custom explicit finite-difference solver for the 1D viscous Burgers' equation, generating a synthetic dataset from randomized smooth initial conditions
- Three generative model variants trained under an identical architecture and training loop for a fair comparison:
  - **Baseline Flow Matching**  conditional Optimal Transport (straight-line) paths
  - **Diffusion baseline**  standard variance-preserving noising path, as a special case of the same framework
  - **Physics-regularized Flow Matching**  OT paths plus a soft penalty enforcing consistency of a conserved quantity (total field mass) along the training trajectory
- Simulation-free vector-field regression training (no ODE solving required during training)
- Euler-integrator sampling at two step counts (5 and 50) to directly test the Number-of-Function-Evaluations (NFE) efficiency claim from the Flow Matching paper
- Physical-consistency evaluation: comparing the distribution of a conserved quantity (mass/field integral) between real data and generated samples
- Generation-quality evaluation via nearest-real-sample MSE
- Fully self-contained, reproducible single notebook with fixed seeds

---

## 🧱 Tech Stack

### Modeling
- PyTorch
- Custom 1D convolutional vector-field network (shared architecture across all three variants)
- Custom Euler ODE integrator for sampling

### Physics / Data Generation
- Custom explicit finite-difference Burgers' equation solver (central differences, periodic domain)
- Randomized smooth initial conditions (superposition of sine modes)

### Environment
- Google Colab (GPU)

---

## 🏗️ Architecture

```text
Random smooth initial conditions
        ↓
Finite-difference Burgers' solver  →  synthetic dataset of PDE final states
        ↓
 ┌────────────────┬────────────────┬──────────────────────┐
 Baseline FM         Diffusion          Physics-regularized FM
 (OT path)           (VP noising path)  (OT path + mass penalty)
 └────────────────┴────────────────┴──────────────────────┘
        ↓
 Euler sampling (5-step / 50-step)
        ↓
 Generation quality (nearest-real MSE) + Physical consistency (mass distribution match)
```

---



## 🚀 Quick Start

### 1) Prerequisites
- Google account (for Colab)
- No local GPU required

### 2) Open in Colab
Open `flow_matching_burgers_pde_generation.ipynb` in Google Colab and run cells top to bottom. All data is generated synthetically inside the notebook  no external downloads are required.

### 3) Configuration
Key flags near the top of the notebook:

```python
n_grid = 128
n_samples = 3000        # number of synthetic Burgers' trajectories
train_fraction = 0.8
epochs = 50              # training epochs per model variant
lambda_phys = 0.1        # physics-penalty weight for the regularized FM variant
seed = 7
```

### 4) Outputs
All tables and figures are generated inline in the notebook. 

---

## 📊 Results

*(seed = 7, 3000 synthetic samples, 2400 train / 600 test, 50 epochs per model)*

### Example data: initial conditions and evolved Burgers' states

<img width="1187" height="495" alt="image" src="https://github.com/user-attachments/assets/015a8c33-c828-429a-80ad-0590733ce5b3" />


### Training loss curves

<img width="613" height="393" alt="image" src="https://github.com/user-attachments/assets/2ff06be6-bcd1-41a8-8177-7c985dc016b3" />


All three models converge smoothly with no instability. Baseline FM reaches the lowest training loss, Diffusion the highest, and Physics FM sits in between — consistent with Diffusion's harder curved-path target and Physics FM's extra penalty term competing with the main regression objective.

### Generation quality and sampling efficiency (nearest-real MSE, lower is better)

| Model | 5-step NFE | 50-step NFE |
|---|---|---|
| Baseline FM | 0.171 | 0.280 |
| Diffusion | 0.186 | 0.247 |
| Physics FM | **0.146** | **0.190** |

Physics FM produces the best generation quality at both step counts. Contrary to the standard Flow Matching claim, **quality did not improve with more sampling steps for any model** — all three got worse going from 5 to 50 Euler steps. This inversion persisted after scaling from 600 to 3000 training samples, suggesting it is a real property of this Euler-sampling setup rather than small-sample noise, and is flagged as an open question rather than explained away (see [Limitations](#️-limitations--future-work)).

<img width="622" height="398" alt="image" src="https://github.com/user-attachments/assets/6d1728c6-1e75-406e-baf4-f8a12826bf1f" />


### Physical consistency (mass/field-integral deviation from real data, lower is better)

| Model | 5-step | 50-step |
|---|---|---|
| Baseline FM | 0.054 | 0.072 |
| Diffusion | **0.050** | 0.081 |
| Physics FM | 0.051 | **0.058** |

Physics-regularized FM gives the clearest physical-consistency advantage in the more-accurate 50-step sampling regime, but does not clearly outperform the other models at fast 5-step sampling. The physics penalty's benefit is therefore step-count-dependent in this run rather than a uniform win, which is reported here as the honest finding rather than smoothed into an unqualified claim.

<img width="678" height="470" alt="image" src="https://github.com/user-attachments/assets/377f3d07-a191-4ff0-8de4-811abbebb8e0" />


### Example generated samples

<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/6dd0bad6-cd5a-4f34-8d4f-28a9eaa643ae" />


Generated states from all three models are visibly rougher than real Burgers' solutions at this model size and training budget the physical-consistency results above should be read as "closer to correct," not "correct."

---

## ⚠️ Limitations & Future Work

- Results are reported for a **single seed** (7); multi-seed variance estimation is left for future work.
- Only the **1D viscous Burgers' equation** is studied, with a hand-written first-order explicit finite-difference solver rather than a validated reference solver or real benchmark data (e.g., PDEBench); findings may not transfer to 2D systems (Navier-Stokes, climate/fluid data) without further validation.
- The model generates **unconditional final states** only — it is not conditioned on the initial condition, so it does not yet model the initial-condition-to-outcome mapping directly.
- The physics-consistency penalty enforces a single, simple constraint (total field mass/integral) rather than a full conservation law or PDE residual; it is a proof-of-concept regularizer, not a physically rigorous constraint.
- The 5→50 sampling-step quality inversion (all models perform worse with more Euler steps) is an open, unresolved finding. It survived a 5x increase in dataset size, which argues against pure sampling noise, but it has not been root-caused, candidate explanations include the fixed-step Euler integrator amplifying per-step error, or the generation-quality metric itself being unstable at this data/model scale. This should be investigated (e.g., with an adaptive-step or RK4 integrator) before drawing strong conclusions.
- Planned next steps (pending additional compute): condition generation on initial states, multi-seed runs with significance testing, stronger architectures (e.g., FNO or U-Net backbones), a proper PDE residual loss instead of the endpoint-mass penalty, and extension to 2D Navier-Stokes data from PDEBench.

---

## 📌 Notes

- All data used in this project is **synthetically generated inside the notebook** via a custom finite-difference Burgers' solver applied to randomized smooth initial conditions — no external dataset or benchmark data is used.
- This project was built as a portfolio piece demonstrating generative modeling (Flow Matching), physics-informed regularization, and rigorous, honest empirical reporting, intended to be extended into a fuller study.
