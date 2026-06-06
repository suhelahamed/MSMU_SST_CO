# Edge-AI Selective Subtask Offloading — Results

Produced by `../train_edge_ai.py`. Trains an MLP to predict per-subtask
offloading decisions on the UAV-augmented MSMU dataset, then evaluates against
baselines using the SAME cost model as `../build_subtask_dataset.py`.

## Model
Compact MLP: 21 features → 256 → 128 → 1 logit, ReLU + Dropout(0.15),
`BCEWithLogitsLoss` (pos_weight for class balance), Adam lr=1e-3, batch 1024,
early stopping (patience 5) on val loss. Best model: `../models/subtask_uav_mlp.pt`.

## Important: leakage guard
In this dataset the oracle label `exec_location_label` is **perfectly determined**
by whether the user has an MCMF anchor (`exec_server_index >= 0`): all of a
user's subtasks share the anchor decision (the selective P2 pattern occurs for
only ~2% of users). The assigned-server geometry features are NaN iff the row is
local, so they leak the label 1:1 — including them yields a trivial 100% accuracy.

To make this an honest prediction task, the model is trained WITHOUT the
anchor-derived features (`exec_server_index` indicator, `mcmf_resource_ratio`,
and all `assigned_*`/`server_*` geometry). It predicts from task, subtask,
channel (`best_channel_gain`), and scenario-level UAV context features
(`num_uav_servers`, `min_dist_to_any_uav`, `has_uav_in_coverage`). Those excluded
columns are still used for the COST evaluation (which needs the true anchor).

## Test results (1,780 users, 5,340 subtasks)
- Subtask accuracy 0.842, precision 0.793, recall 0.870, F1 0.830, AUC 0.926.
- Per-user means: Edge-AI cost 33.19 vs Oracle 30.44, All-Local 40.89,
  All-MCMF 30.44, Random 35.71.
- Deadline (1.5× median oracle latency = 3.26 s) violation: Edge-AI 23.5% vs
  All-Local 52.9%, Random 40.3%, Oracle/All-MCMF ~21.2%.
- All-MCMF ≈ Oracle because the oracle is dominated by full-offload patterns.

## Files
- `results_metrics.csv` — per (user × method) latency/energy/cost/deadline rows
  (8,900 = 1,780 users × 5 methods), with `anchor_is_uav` flag.
- `summary_metrics.csv` — per-method aggregate means/stds.
- `summary_table.tex` — paper-ready LaTeX table.
- `training_history.csv` — per-epoch train/val loss and val accuracy.
- `latency_cdf.png`, `energy_bar.png`, `deadline_bar.png`,
  `uav_vs_ground_latency.png` — high-res figures.

## Baselines
- All-Local: every subtask local.
- All-MCMF: all subtasks → s_star (local if s_star=-1).
- Random: per-subtask coin flip (local if s_star=-1).
- Oracle: the dataset's `exec_location_label` (upper bound).
