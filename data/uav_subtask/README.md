# UAV-Augmented Subtask MSMU Dataset

Built from `../subtask/{train,val,test}_subtask.parquet` by
`../../build_uav_dataset.py`. Adds UAV-supported MEC geometry/link features.
**No ML training. Existing labels are unchanged.**

## Files
- `train_uav.parquet`, `val_uav.parquet`, `test_uav.parquet` (40 columns each;
  25 original subtask columns + 15 new UAV columns).
- `server_positions.parquet` — per (scenario_id, server_index) geometry.
- `user_positions.parquet` — per (scenario_id, user_id) geometry.

Splits keep the same scenario partition (560 / 120 / 120 scenarios →
25,170 / 5,490 / 5,340 subtask rows). Labels
(`exec_location_label`, `exec_is_local`, `exec_server_index`) are byte-for-byte
identical to the input (verified).

## Geometry generation (seed 20260606, region = [0,1000] m square)
- **UAV servers**: per scenario, if `num_servers >= 2` pick 1–2 servers uniformly
  at random as UAVs; if `num_servers == 1`, that server is a UAV w.p. 0.5. Here
  every scenario ended up with 1 or 2 UAVs.
- **Server positions**: `(server_x, server_y)` uniform in [0,1000] m.
  UAV altitude `server_z ∈ [80,150]` m; ground servers `server_z = 0`.
- **User positions**: `(user_x, user_y)` uniform in [0,1000] m, `user_z = 0`.

## Per-subtask link features (to the assigned/anchor server `exec_server_index`)
Computed only for **offloaded** subtasks (`exec_server_index >= 0`); for local
subtasks (`-1`) all assigned-server features are `NaN` and
`assigned_is_uav_server = 0`.

- `assigned_dist_2d = sqrt(dx^2 + dy^2)`
- `assigned_dist_3d = sqrt(dx^2 + dy^2 + dz^2)`
- `assigned_elevation_angle = arctan2(dz, dist_2d + eps)` — **radians**.
- `assigned_is_uav_server` — 1 if the anchor server is a UAV, else 0.
- `assigned_los_prob` — line-of-sight probability:
  - UAV anchor: sigmoid in elevation (urban-micro, 3GPP/ITU style)
    `los = 1 / (1 + A·exp(-B·(θ_deg − A)))`, with `A=9.61, B=0.16`.
  - Ground anchor: `1.0` (terrestrial link treated as deterministic LoS).
  - Local: `NaN`.
- `assigned_effective_rate` — uplink rate via a free-space-like model:
  - channel gain `h = β0 / dist_3d^n` with `β0=1e-3, n=2.2`,
  - LoS blend `h_eff = h · (los + (1−los)·0.2)` (NLoS share attenuated ×0.2),
  - `SINR = P_tx·h_eff / N0`, `R = B·log2(1 + SINR)` (B, N0 from base config,
    `P_tx = 0.3 W`). Range observed ≈ 0.31–1.68 Gbps.

## Scenario-level UAV context (all rows)
- `num_uav_servers` — number of UAV servers in the scenario (1 or 2 here).
- `min_dist_to_any_uav` — 3D distance from the user to the nearest UAV server in
  its scenario (regardless of the oracle anchor).
- `has_uav_in_coverage` — 1 if `min_dist_to_any_uav <= 600 m`, else 0
  (~71% of rows = 1).

## Interpretation notes
- These features describe UAV availability/geometry; they do **not** change the
  oracle offloading decision, which remains anchored to the original MCMF server.
- Use `assigned_*` features as link descriptors for the chosen server, and the
  scenario-level context (`num_uav_servers`, `min_dist_to_any_uav`,
  `has_uav_in_coverage`) as availability signals for later modeling.

## Loading
```python
import pandas as pd
train = pd.read_parquet("train_uav.parquet")
```
