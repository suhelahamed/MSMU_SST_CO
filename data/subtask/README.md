# Subtask-Level MSMU Dataset with Oracle Selective-Offloading Labels

Built from `../raw/base_msmu_raw.parquet` (MSMU-CO MCMF base) by
`../../build_subtask_dataset.py`. No UAV features, no ML training.

## What this is
Each base (scenario, user) macro task is decomposed into **K = 3 subtasks**.
For every user we evaluate candidate offloading patterns over its subtasks and
pick the minimum-cost one as the **oracle**, then label each subtask
local vs. offload-to-`s_star` (the user's MCMF-assigned server).

## Files
- `train_subtask.parquet`, `val_subtask.parquet`, `test_subtask.parquet`
- `all_subtask.parquet` — the full table (convenience)

## Split
By **scenario_id** (no leakage): 560 / 120 / 120 scenarios
→ 25,170 / 5,490 / 5,340 subtask rows (36,000 total = 12,000 users × 3).

## Subtask decomposition
- K=3 normalized random weights `w_k` (seed 42).
- `subtask_data_size = w_k * task_data_size`, `subtask_cpu_cycles = w_k * required_cpu_cycles`
  (verified to sum back to the user totals).
- `subtask_type` by CPU-cycle rank: heaviest→1 (heavy-compute), mid→0 (pre-proc), light→2 (post-proc).

## Cost model (for the oracle)
For each subtask, with channel gain `h` (best_channel_gain), bandwidth `B`,
noise `N0`, transmit power `P_tx`, local freq `f_l`, server capacity `F_t`,
server share = `mcmf_resource_ratio` (else `1/num_users`):

- Uplink rate: `R = B*log2(1 + SINR)`, `SINR = P_tx*h^2 / (N0 + P_tx*h^2)`.
- Offloaded subtask: `T = data/R + setup(0.2s) + cycles/(F_t*share)`,
  energy `= P_tx*(data/R) + P_tx*0.2`.
- Local subtask: `T = cycles/f_l`, energy `= kappa*f_l^2*cycles`.
- USER latency = Σ T, USER energy = Σ energy.
- **USER cost = alpha_u·latency + (1-alpha_u)·energy**, where
  `alpha_u = alpha_latency_weight` from the base data (matches MSMU-CO's
  per-user latency/energy objective).

Candidate patterns (each subtask ∈ {local, s_star}):
- **P1**: all subtasks → s_star (all local if s_star = -1)
- **P2**: heaviest ⌈K/2⌉ subtasks → s_star, rest local
- **P3**: all subtasks local

Oracle = argmin cost over the candidates.

## Labels (consistent representation)
- `exec_location_label`: **0 = local, 1 = offload to s_star** (primary class).
- `exec_is_local` (bool) and `exec_server_index` (server idx, -1 if local) — redundant, kept for convenience.
- `oracle_pattern`: which candidate won (P1/P2/P3).

Label distribution: ~55.7% local, ~44.3% offload.
Oracle patterns: P1 (full offload) 35,217 subtasks; P2 (selective) 783 subtasks.
P3/local dominates because users with `s_star = -1` keep everything local and,
when a server is the anchor, full offload usually wins on energy — selectivity
(P2) emerges for latency-sensitive users with small low-compute subtasks.

## Schema
Identifiers: `scenario_id, num_servers, num_users, user_id, subtask_id`.
User features: `task_data_size, required_cpu_cycles, local_cpu_freq, alpha_latency_weight, best_channel_gain, mcmf_assigned_server, mcmf_resource_ratio`.
Subtask features: `subtask_weight, subtask_data_size, subtask_cpu_cycles, subtask_type`.
Global/server params: `F_t, B, N0, P_tx, kappa`.
Labels: `oracle_pattern, exec_is_local, exec_server_index, exec_location_label`.

## Loading
```python
import pandas as pd
train = pd.read_parquet("train_subtask.parquet")
```
