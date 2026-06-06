"""
Mutate the base MSMU dataset into a SUBTASK-LEVEL dataset with oracle labels
for selective subtask offloading (no UAV, no ML training).

Pipeline
--------
1. Load base (scenario, user) rows from data/raw/base_msmu_raw.parquet.
2. Decompose each macro task into K=3 subtasks via normalized random weights.
3. For each user, evaluate 3 candidate offloading patterns (P1/P2/P3) over its
   subtasks using a documented latency+energy cost model, pick the min-cost
   pattern as the ORACLE, and label each subtask local(0) vs server s_star(1).
4. Split by scenario_id into 70/15/15 train/val/test and save Parquet.

COST MODEL (documented)
-----------------------
For each subtask k of a user with local CPU freq f_l, server CPU capacity F_t,
channel gain h (best_channel_gain proxy), bandwidth B, noise N0, tx power P_tx:

  Uplink SINR / rate (offloaded subtasks) -- matches the MSMU-CO PHY model,
  where the received signal power saturates the link (realistic sub-Gbps rate):
      SINR = P_tx * h^2 / (N0 + P_tx * h^2)
      R    = B * log2(1 + SINR)
  Per-offload signaling/setup OVERHEAD (so offloading tiny subtasks is not free):
      T_setup = OVERHEAD_S seconds, E_setup = P_tx * T_setup joules, added once
      per offloaded subtask. This is what makes SELECTIVE offloading meaningful:
      light subtasks may be cheaper to keep local than to pay the offload setup.
  Transmission time:
      T_tx = subtask_data_size / R + T_setup   (0 if executed locally)
  Execution time:
      local  : T_exec = subtask_cpu_cycles / f_l
      remote : T_exec = subtask_cpu_cycles / (F_t * server_share)
               where server_share = mcmf_resource_ratio if available else 1/num_users
  Per-subtask completion time:
      t_k = T_tx + T_exec
  Energy:
      local  : E = kappa * f_l^2 * subtask_cpu_cycles      (CPU energy model)
      remote : E = P_tx * (subtask_data_size / R) + E_setup (transmit energy)

  USER latency = SUM_k t_k     (sequential subtask execution per user)
  USER energy  = SUM_k E_k
  USER cost    = alpha_u * latency + beta_u * energy
     where alpha_u = alpha_latency_weight (from the base MCMF data) and
     beta_u = 1 - alpha_u. This matches the MSMU-CO objective, which weights
     latency vs energy per user. Latency-sensitive users (high alpha_u) prefer
     keeping low-compute subtasks local to avoid the offload setup latency,
     which is what produces genuinely SELECTIVE oracle patterns.

The oracle pattern is argmin over {P1, P2, P3} of USER cost.
  P1: all subtasks -> s_star  (all local if s_star == -1)
  P2: heaviest subtasks (top ceil(K/2) by cpu cycles) -> s_star, rest local
      (all local if s_star == -1)
  P3: all subtasks local
"""

import os
import json
import math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw", "base_msmu_raw.parquet")
CFG = os.path.join(BASE, "data", "raw", "base_msmu_config.json")
OUT = os.path.join(BASE, "data", "subtask")

K = 3                  # fixed number of subtasks per macro task
ALPHA, BETA = 1.0, 1.0 # cost weights (latency, energy)
OVERHEAD_S = 0.20      # per-offload signaling/setup latency (s) -> selectivity
SEED = 42

# subtask type heuristic for K=3: by CPU-cycle fraction rank
# largest fraction -> 1 (heavy-compute), then 0 (pre-processing), 2 (post-processing)
SUBTASK_TYPES = {0: 0, 1: 1, 2: 2}  # placeholder; assigned by rank below


def uplink_rate(B, P_tx, h, N0):
    # MSMU-CO PHY: received power saturates the link -> realistic bounded rate
    sinr = P_tx * (h ** 2) / (N0 + P_tx * (h ** 2))
    return B * np.log2(1.0 + sinr)


def pattern_cost(offload_mask, sub_cycles, sub_data, f_l, F_t, R, P_tx,
                 kappa, server_share, alpha_u):
    """Return (total_cost, total_latency, total_energy) for a boolean
    offload mask over the K subtasks, weighted by the user's alpha/beta."""
    beta_u = 1.0 - alpha_u
    total_lat = 0.0
    total_en = 0.0
    for k in range(len(sub_cycles)):
        if offload_mask[k]:
            t_comm = sub_data[k] / R if R > 0 else float("inf")
            t_tx = t_comm + OVERHEAD_S
            t_exec = sub_cycles[k] / (F_t * server_share)
            t_k = t_tx + t_exec
            e_k = P_tx * t_comm + P_tx * OVERHEAD_S
        else:
            t_tx = 0.0
            t_exec = sub_cycles[k] / f_l
            t_k = t_exec
            e_k = kappa * (f_l ** 2) * sub_cycles[k]
        total_lat += t_k
        total_en += e_k
    return alpha_u * total_lat + beta_u * total_en, total_lat, total_en


def main():
    rng = np.random.default_rng(SEED)
    cfg = json.load(open(CFG))
    P_tx = cfg["P_t_transmit_power_w"]
    kappa = cfg["kappa_energy_coeff"]

    df = pd.read_parquet(RAW)

    rows = []
    for _, u in df.iterrows():
        D = float(u["task_data_size"])
        C = float(u["required_cpu_cycles"])
        f_l = float(u["local_cpu_freq"])
        F_t = float(u["F_t_server_cpu_capacity"])
        B = float(u["B_bandwidth_hz"])
        N0 = float(u["N0_noise_power"])
        h = float(u["best_channel_gain"])
        s_star = int(u["mcmf_assigned_server"])
        ratio = u["mcmf_resource_ratio"]
        num_users = int(u["num_users"])

        # --- subtask decomposition ---
        w = rng.random(K)
        w = w / w.sum()
        sub_data = w * D
        sub_cycles = w * C

        # subtask type by CPU-cycle fraction rank (heaviest -> heavy-compute=1)
        order = np.argsort(-sub_cycles)  # descending
        sub_type = np.empty(K, dtype=int)
        # rank 0 (heaviest)=1 heavy, rank1=0 pre, rank2=2 post (works for K=3)
        type_by_rank = [1, 0, 2] if K == 3 else None
        for rank, idx in enumerate(order):
            sub_type[idx] = (type_by_rank[rank] if type_by_rank
                             else (1 if rank == 0 else 0))

        # server resource share for remote execution
        server_share = float(ratio) if (s_star != -1 and pd.notna(ratio) and ratio > 0) \
            else (1.0 / max(num_users, 1))

        R = uplink_rate(B, P_tx, h, N0)

        # --- candidate offload patterns ---
        if s_star == -1:
            # no anchored server -> only "all local" is feasible as offload target
            patterns = {
                "P1_all_to_sstar": np.zeros(K, dtype=bool),  # reduces to local
                "P3_all_local": np.zeros(K, dtype=bool),
            }
        else:
            heavy_idx = np.argsort(-sub_cycles)[: math.ceil(K / 2)]
            p2 = np.zeros(K, dtype=bool)
            p2[heavy_idx] = True
            patterns = {
                "P1_all_to_sstar": np.ones(K, dtype=bool),
                "P2_heavy_to_sstar": p2,
                "P3_all_local": np.zeros(K, dtype=bool),
            }

        alpha_u = float(u["alpha_latency_weight"])
        best_name, best_mask, best_cost = None, None, float("inf")
        for name, mask in patterns.items():
            cost, lat, en = pattern_cost(mask, sub_cycles, sub_data, f_l, F_t,
                                         R, P_tx, kappa, server_share, alpha_u)
            if cost < best_cost:
                best_cost, best_name, best_mask = cost, name, mask

        # --- emit subtask rows with oracle labels ---
        for k in range(K):
            is_off = bool(best_mask[k]) and (s_star != -1)
            rows.append({
                "scenario_id": int(u["scenario_id"]),
                "num_servers": int(u["num_servers"]),
                "num_users": num_users,
                "user_id": int(u["user_id"]),
                "subtask_id": k,
                # user-level features
                "task_data_size": D,
                "required_cpu_cycles": C,
                "local_cpu_freq": f_l,
                "alpha_latency_weight": float(u["alpha_latency_weight"]),
                "best_channel_gain": h,
                "mcmf_assigned_server": s_star,
                "mcmf_resource_ratio": float(ratio) if pd.notna(ratio) else np.nan,
                # subtask-level features
                "subtask_weight": float(w[k]),
                "subtask_data_size": float(sub_data[k]),
                "subtask_cpu_cycles": float(sub_cycles[k]),
                "subtask_type": int(sub_type[k]),
                # global/server params
                "F_t": F_t, "B": B, "N0": N0,
                "P_tx": P_tx, "kappa": kappa,
                # oracle labels
                "oracle_pattern": best_name,
                "exec_is_local": (not is_off),
                "exec_server_index": (s_star if is_off else -1),
                # 0 = local, 1 = offload to s_star
                "exec_location_label": (1 if is_off else 0),
            })

    sub = pd.DataFrame(rows)

    # ---- split by scenario_id (70/15/15) ----
    scen_ids = np.array(sorted(sub["scenario_id"].unique()))
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(scen_ids)
    n = len(scen_ids)
    n_tr = int(round(0.70 * n))
    n_va = int(round(0.15 * n))
    train_ids = set(scen_ids[:n_tr])
    val_ids = set(scen_ids[n_tr:n_tr + n_va])
    test_ids = set(scen_ids[n_tr + n_va:])

    train = sub[sub["scenario_id"].isin(train_ids)].reset_index(drop=True)
    val = sub[sub["scenario_id"].isin(val_ids)].reset_index(drop=True)
    test = sub[sub["scenario_id"].isin(test_ids)].reset_index(drop=True)

    os.makedirs(OUT, exist_ok=True)
    paths = {}
    for name, d in [("train", train), ("val", val), ("test", test)]:
        p = os.path.join(OUT, f"{name}_subtask.parquet")
        d.to_parquet(p, index=False)
        paths[name] = p

    # save the full table too (convenience) + a small metadata json
    sub.to_parquet(os.path.join(OUT, "all_subtask.parquet"), index=False)

    # ---------------- REPORT ----------------
    print("=" * 72)
    print("SUBTASK-LEVEL DATASET — REPORT")
    print("=" * 72)
    print(f"K (subtasks per user): {K}   |   cost = alpha_u*latency + (1-alpha_u)*energy")
    print(f"Total subtask rows: {len(sub)}  (= {df.shape[0]} users x {K})")
    print()
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"[{name:5}] scenarios={d['scenario_id'].nunique():4d}  "
              f"users={d.groupby('scenario_id')['num_users'].first().sum():6d}  "
              f"subtasks={len(d):6d}")
    print()
    print("subtask_data_size  (min/mean/max): "
          f"{sub['subtask_data_size'].min():.3e} / "
          f"{sub['subtask_data_size'].mean():.3e} / "
          f"{sub['subtask_data_size'].max():.3e}")
    print("subtask_cpu_cycles (min/mean/max): "
          f"{sub['subtask_cpu_cycles'].min():.3e} / "
          f"{sub['subtask_cpu_cycles'].mean():.3e} / "
          f"{sub['subtask_cpu_cycles'].max():.3e}")
    print()
    lab = sub["exec_location_label"].value_counts(normalize=True).sort_index()
    print("Label distribution (exec_location_label):")
    print(f"  local   (0): {lab.get(0, 0)*100:5.1f}%  ({(sub['exec_location_label']==0).sum()} subtasks)")
    print(f"  offload (1): {lab.get(1, 0)*100:5.1f}%  ({(sub['exec_location_label']==1).sum()} subtasks)")
    print()
    print("Oracle pattern distribution:")
    print(sub["oracle_pattern"].value_counts().to_string())
    print()
    print("COST MODEL: per-subtask completion time t_k = T_tx + T_exec, with")
    print("  T_tx = data/R + setup(0.2s), R = B*log2(1+SINR),")
    print("  SINR = P_tx*h^2/(N0+P_tx*h^2); local exec = cycles/f_l,")
    print("  remote exec = cycles/(F_t*server_share). USER latency = sum_k t_k,")
    print("  USER energy = sum_k E_k (local: kappa*f_l^2*cycles; remote: P_tx*comm+setup).")
    print("  USER cost = alpha_u*latency + (1-alpha_u)*energy. Oracle=argmin over P1/P2/P3.")
    print()
    print("Saved:")
    for name, p in paths.items():
        print(f"  {name:5}: {p}")
    print(f"  all  : {os.path.join(OUT, 'all_subtask.parquet')}")


if __name__ == "__main__":
    main()
