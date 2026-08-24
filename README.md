# MSMU_SST_CO: Edge-AI-based Selective Subtask Offloading in UAV-Supported MEC

This repository contains code, data-processing scripts, and experiment pipelines for an **Edge-AI-based selective subtask computation offloading framework** built on top of the **MSMU-CO** multi-server multi-user offloading dataset generator.[web:9]

The project:

- Uses **MSMU-CO** (MCMF-based multi-server multi-user computation offloading) as an **offline oracle** for optimal offloading decisions.[web:9]
- Mutates the original task-level dataset into a **subtask-level selective offloading** environment.
- Augments the environment with **UAV-supported MEC semantics** (UAV vs ground servers, 3D geometry, LoS).
- Trains a lightweight **Edge-AI model (MLP)** to predict selective offloading decisions from task, channel, and UAV context.
- Benchmarks against **naive baselines** (all-local, all-MCMF, random) and a strong **LR-style greedy optimization baseline** inspired by multi-server multi-user multi-task MEC literature.[web:76][web:11]

The code and data here reproduce the main experiments reported in the associated research manuscript.

---

## 1. Project Overview

### 1.1 Problem Setting

We consider a **multi-server multi-user MEC system** where each ground user generates computation tasks that can be processed either locally or offloaded to one of multiple edge servers (UAVs and ground MEC nodes). Each user task is decomposed into multiple subtasks (pre-processing, heavy compute, post-processing), enabling **selective subtask offloading** rather than all-or-nothing offload.

The goal is to **minimize a weighted sum of latency and energy** per user, while respecting server capacity constraints and leveraging UAV mobility and coverage. The cost function follows the MSMU-CO formulation with user-specific latency–energy trade-off weights.[web:9]

### 1.2 High-Level Pipeline

1. **Base data generation** from MSMU-CO (task-level multi-server multi-user offloading).
2. **Subtask mutation**: split each task into multiple subtasks and generate oracle selective offloading labels using the MSMU-CO cost model.
3. **UAV augmentation**: assign some servers as UAVs, generate 3D positions, distances, elevation, LoS probabilities, and UAV-aware link features.
4. **Edge-AI model training**: train an MLP classifier to predict local vs offload for each subtask from context features.
5. **Baseline evaluation**: compare against Oracle, All-Local, All-MCMF, Random, and an LR-greedy baseline.
6. **Result generation**: export CSV metrics, LaTeX tables, and publication-ready figures.

---

## 2. Repository Structure

> Note: exact paths may vary slightly; adjust this section to your actual tree after unzipping.

```text
MSMU_Comp_off/
  data/
    raw/             # base MSMU-CO dataset (task-level)
    subtask/         # subtask-level datasets + oracle labels
    uav_subtask/     # UAV-augmented subtask datasets
  src/               # main Python scripts (if separated)
    ...              # (data prep, env, models, eval)
  models/
    subtask_uav_mlp.pt    # trained Edge-AI MLP checkpoint
  output/
    results_metrics.csv   # per-user metrics per method
    summary_metrics.csv   # aggregated metrics
    summary_table.tex     # LaTeX table for paper
    latency_cdf.png
    energy_bar.png
    deadline_bar.png
    uav_vs_ground_latency.png
    README.md             # description of outputs
  MSMU-CO/           # (optional) cloned MSMU-CO repo or pointer
  build_subtask_dataset.py
  build_uav_dataset.py
  add_lr_greedy.py
  train_eval_mlp.py       # (or similar) training + evaluation script
  requirements.txt
  README.md               # (this file)
```

If MSMU-CO is not included as a submodule, see Section 3 for cloning instructions.

---

## 3. Dependencies and Setup

### 3.1 MSMU-CO Dataset Generator

This project depends on the MSMU-CO dataset generation tool:

- GitHub: https://github.com/qiyu3816/MSMU-CO  
- Description: A dataset generation tool based on **minimum-cost multi-commodity flow (MCMF)** for multi-server multi-user computation offloading.[web:9]

Clone MSMU-CO into this repo (if not already present):

```bash
git clone https://github.com/qiyu3816/MSMU-CO.git
```

or adjust paths inside `build_subtask_dataset.py` as needed.

### 3.2 Python Environment

Recommended environment:

- Python 3.10+ (adjust if needed)
- Key packages:
  - `numpy`, `pandas`, `scipy`
  - `torch` (PyTorch)
  - `pyyaml`
  - `matplotlib` / `seaborn` (or `plotly`) for plots

Install dependencies:

```bash
pip install -r requirements.txt
```

(or create a conda env and install manually.)

---

## 4. Reproducing the Pipeline

### 4.1 Step 1 – Base MSMU-CO Dataset (Task-Level)

Script: **(inside Computer this was Job 1)**

- Generates `data/raw/base_msmu_raw.parquet` by:
  - Importing MSMU-CO’s generation functions.
  - Sampling 800 scenarios across `U ∈ {10,20}` and `S ∈ {3,5}`.
  - Extracting user-level features, channel features, global parameters, and MCMF assignments.

If you need to regenerate:

```bash
python build_base_dataset.py   # if present, or see build_subtask_dataset.py header notes
```

(Adapt to the actual script/entry point present in the repo.)

### 4.2 Step 2 – Subtask Mutation + Oracle Labels

Script: `build_subtask_dataset.py`

This step:

- Loads `data/raw/base_msmu_raw.parquet`.
- Splits each user task into `K=3` subtasks (pre, heavy, post) with random weights.
- Uses the **MSMU-CO cost model** to evaluate three patterns per user:
  - P1: all subtasks offloaded to the MCMF anchor server.
  - P2: selective offload (heavy subtasks offloaded, light ones local).
  - P3: all local.
- Selects the min-cost pattern as the **oracle selective offloading decision** and assigns labels:
  - `exec_location_label` (0 = local, 1 = offload),
  - `exec_is_local`,
  - `exec_server_index`.
- Splits into train/val/test by `scenario_id`:
  - `data/subtask/train_subtask.parquet`
  - `data/subtask/val_subtask.parquet`
  - `data/subtask/test_subtask.parquet`

Run:

```bash
python build_subtask_dataset.py
```

### 4.3 Step 3 – UAV Augmentation

Script: `build_uav_dataset.py`

This step:

- Loads `data/subtask/*_subtask.parquet`.
- For each scenario:
  - Assigns 1–2 servers as UAVs; others are ground servers.
  - Samples 3D positions for all servers (UAV altitudes, ground at z=0).
  - Samples positions for all users.
- Derives UAV-aware features:
  - Distances, elevation angle, LoS probability, effective rate to the anchor server.
  - Scenario-level context: number of UAVs, minimum distance to any UAV, coverage indicators.
- Produces UAV-augmented subtask datasets:
  - `data/uav_subtask/train_uav.parquet`
  - `data/uav_subtask/val_uav.parquet`
  - `data/uav_subtask/test_uav.parquet`

Run:

```bash
python build_uav_dataset.py
```

### 4.4 Step 4 – Edge-AI MLP Training + Evaluation

Script: `train_eval_mlp.py` (or similar; adjust to your actual script name)

This step:

- Loads `data/uav_subtask/train_uav.parquet` and `val_uav.parquet`.
- Selects “honest” features (task/subtask/global/UAV context) and target `exec_location_label`.
- Trains an MLP classifier (21→256→128→1, ReLU+Dropout) with BCE loss and Adam optimizer.
- Saves best model to `models/subtask_uav_mlp.pt`.
- Evaluates on `test_uav.parquet` and compares against:
  - **oracle** (selective labels),
  - **all_local**,
  - **all_mcmf**,
  - **random**,
- Reuses the original MSMU-CO cost model to compute:
  - per-user latency, energy, cost,
  - deadline violation rate (deadline = 1.5× median oracle latency),
  - subtask-level classification metrics (acc, precision, recall, F1, AUC).

Outputs:

- `output/results_metrics.csv` (per-user metrics, one row per method × user).
- `output/summary_metrics.csv`.
- `output/summary_table.tex`.
- Plots under `output/` (`latency_cdf.png`, `energy_bar.png`, etc.).

Run:

```bash
python train_eval_mlp.py
```

### 4.5 Step 5 – LR-Greedy Baseline

Script: `add_lr_greedy.py`

This step:

- Loads `data/uav_subtask/test_uav.parquet` and `output/results_metrics.csv`.
- Implements an **LR-style greedy baseline**:
  - For each user, computes cost difference between all-local and all-offload-to-anchor patterns.
  - Sorts users by cost gain; greedily admits offloading while respecting server capacity.
- Evaluates LR-Greedy under the same cost model and adds its metrics to:
  - `output/results_metrics.csv` (new method `lr_greedy`).
  - `output/summary_metrics.csv`.
  - `output/summary_table.tex`.
- Regenerates plots including LR-Greedy.

Run:

```bash
python add_lr_greedy.py
```

---

## 5. Main Artifacts

After running the full pipeline, key artifacts are:

- **Model checkpoint**
  - `models/subtask_uav_mlp.pt` – trained Edge-AI MLP for selective subtask offloading.
- **Metrics**
  - `output/results_metrics.csv` – per-user, per-method metrics on the test set.
  - `output/summary_metrics.csv` – aggregated metrics (mean ± std) per method.
- **Tables**
  - `output/summary_table.tex` – LaTeX table summarizing latency, energy, cost, deadline violation, and accuracy per method.
- **Figures**
  - `output/latency_cdf.png` – CDF of user latency for all methods.
  - `output/energy_bar.png` – average energy per method.
  - `output/deadline_bar.png` – deadline violation rate per method.
  - `output/uav_vs_ground_latency.png` – stratified latency for UAV vs ground anchors.

These are intended to be used directly in the manuscript (with minor formatting tweaks).

---

## 6. How to Use This Repo for Your Own Research

You can adapt this pipeline to other offloading problems by:

- Replacing MSMU-CO with another base dataset or generator.
- Modifying `build_subtask_dataset.py` to implement different subtask structures or cost models.
- Changing `build_uav_dataset.py` to match new UAV mobility or channel models.
- Swapping `train_eval_mlp.py` for other Edge-AI models (GNNs, Transformers) or DRL agents.
- Adding new baselines in `src/baselines/` and integrating them into the evaluation scripts.

The core idea is to treat this repo as a **reference implementation** of a complete, reproducible offloading pipeline: dataset, environment, model, baselines, and result generation.

---

## 7. Citation

If you use this code or environment in your own work, please cite:

- The original MSMU-CO dataset/tool.[web:9]  
- Your own paper built on top of this repo (once published).

---

## 8. Contact

For questions, issues, or collaboration:

- GitHub: [@suhelahamed](https://github.com/suhelahamed)
- suhelahamed@ieee.org
