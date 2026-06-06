# Base MSMU Multi-Server Multi-User Offloading Dataset

Generated from the MSMU-CO repository (https://github.com/qiyu3816/MSMU-CO),
which provides the dataset generation tool for the
[IEEE Dataport: Dataset for Multi-Server Multi-User Computation Offloading](https://ieee-dataport.org/documents/dataset-multi-server-multi-user-computation-offloading)
(DOI: https://dx.doi.org/10.21227/386p-7h83).

Offloading decisions are computed with the original repo's **max-flow-min-cost
(MCMF)** heuristic solver (`solve_once`).

## Files
- `base_msmu_raw.parquet` — the dataset (long format, one row per (scenario, user)).
- `base_msmu_config.json` — global/server resource parameters used by the generator.

## Generation settings
- Scenario regimes: **U ∈ {10, 20} users**, **S ∈ {3, 5} servers** (4 regimes).
- **200 scenarios per regime → 800 scenarios total**, **12,000 rows**.
- Random seed: 42 (reproducible). Script: `../../generate_base_dataset.py`.

## Schema (long format — each row = one user in one scenario)

| Column | Meaning |
|---|---|
| `scenario_id` | Unique scenario index (0–799) |
| `num_servers` | Number of edge servers S in the scenario |
| `num_users` | Number of users U in the scenario |
| `user_id` | User index within the scenario (0 … U-1) |
| `task_data_size` | Task input data size (bits) |
| `required_cpu_cycles` | CPU cycles required (= data_size × 3e3) |
| `local_cpu_freq` | Local device CPU frequency (cycles/s) |
| `alpha_latency_weight` | Latency-vs-energy tradeoff weight α ∈ [0,1] (β = 1-α) |
| `num_connected_servers` | How many servers this user can reach |
| `best_channel_gain` | Best channel gain across the user's connected servers |
| `channel_gain_to_assigned` | Channel gain to the MCMF-assigned server (NaN if local) |
| `local_exec_satisfies_delay` | 1 if local execution meets the delay bound θ |
| **`mcmf_assigned_server`** | **MCMF-optimal server index. `-1` = executed locally (no offload).** |
| `mcmf_resource_ratio` | Resource-allocation ratio at the assigned server (NaN if local) |
| `scenario_mcmf_total_cost` | MCMF-optimal total cost for the whole scenario |
| `F_t_server_cpu_capacity` | Server CPU capacity = 3.36e10 cycles/s |
| `B_bandwidth_hz` | Channel bandwidth = 8e7 Hz (80 MHz) |
| `N0_noise_power` | Noise power for SINR = 7.96159e-13 |
| `P_t_transmit_power_w` | Transmit power = 0.3 W |
| `P_I_compute_power_w` | Compute/idle power proxy = 0.15 W |
| `kappa_energy_coeff` | Local-compute energy coefficient = 1e-28 |
| `theta_max_delay_s` | Max tolerable delay = 2.0 s |
| `input_size_*`, `f_local_*` | Distribution params (mu/sigma/low/up) for task size & local CPU |

Server-level resource parameters (CPU capacity, bandwidth, noise, power) are
constant across servers in the MSMU-CO generator, so they are attached to every
row and also stored in `base_msmu_config.json`.

## Notes
- The original `generate_graph` could request up to 4 server connections per
  user even when S < 4; for S = 3 this was capped to the feasible count
  (see `generate_base_dataset.py`). All other generation logic is unchanged.
- ~45% of users are offloaded (assigned to a server); the rest execute locally.

## Loading
```python
import pandas as pd
df = pd.read_parquet("base_msmu_raw.parquet")
```
