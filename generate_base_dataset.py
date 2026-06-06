"""
Generate a BASE multi-server multi-user (MSMU) computation-offloading dataset.

This script reuses the original MSMU-CO generation logic
(https://github.com/qiyu3816/MSMU-CO) by importing its core functions, and
produces a clean, documented LONG-FORMAT table where each row is a
(scenario, user) pair.

For each user it records:
  - user-level features: task data size, required CPU cycles, local CPU freq,
    latency/energy tradeoff weight (alpha)
  - channel features: channel gain to the assigned server, and the best
    channel gain among all servers the user is connected to
  - the MCMF-optimal offloading decision: the server index assigned to the
    user (-1 == executed locally / no offload), plus the resource-allocation
    ratio for that assignment.

Server-level / global resource parameters (CPU capacity F_t, bandwidth B,
noise N0, transmit power, etc.) are constant across servers in the MSMU-CO
generator and are attached to every row plus saved to a sidecar config.

Scenario regimes: U in {10, 20}, S in {3, 5}  -> 4 regimes.
"""

import os
import sys
import json
import random

import numpy as np
import pandas as pd
import networkx as nx

# --- make MSMU-CO importable ---
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MSMU-CO")
sys.path.insert(0, REPO_DIR)

from generate_explorative_data import (  # noqa: E402
    solve_once,
    range_random,
)


def generate_graph(server_num, user_num):
    """Bipartite user->server connectivity graph.

    Mirrors MSMU-CO's generate_graph but caps the number of servers a user
    connects to at the available server count (the original picks up to 4 even
    when server_num < 4, which raises a sampling error for S in {3}).
    """
    G = nx.DiGraph()
    servers = ['s' + str(i) for i in range(server_num)]
    users = ['u' + str(i) for i in range(user_num)]
    G.add_nodes_from(servers, bipartite=0)
    G.add_nodes_from(users, bipartite=1)

    choices = [1, 2, 3, 4]
    weights = [0.1, 0.75, 0.1, 0.05]
    # restrict to feasible connection counts (<= server_num)
    feas = [(c, w) for c, w in zip(choices, weights) if c <= server_num]
    feas_choices = [c for c, _ in feas]
    feas_weights = [w for _, w in feas]

    for user in users:
        num_servers = random.choices(feas_choices, weights=feas_weights, k=1)[0]
        for server in random.sample(servers, num_servers):
            G.add_edge(user, server)

    for server in servers:
        if G.in_degree(server) == 0:
            random_weights = [i / user_num for i in range(user_num)]
            node = random.choices([i for i in range(user_num)], weights=random_weights, k=1)[0]
            G.add_edge(users[node], server)

    return G

# ---------------------------------------------------------------------------
# Global / server-level resource parameters (identical to MSMU-CO generator)
# ---------------------------------------------------------------------------
F_t = 33.6e9        # server CPU capacity (cycles/s)
theta = 2.0         # max tolerable delay (s)
P_t = 0.3           # transmit power (W)
P_I = 0.15          # idle/compute power proxy (W)
kappa = 1e-28       # energy coefficient for local computation
B = 8e7             # channel bandwidth = 80 MHz
N0 = 7.96159e-13    # noise power (xi**2) for SINR

is_mu, is_sigma, is_low, is_up = 0.65e7, 0.3e7, 0.2e7, 1.1e7   # input size dist
fl_mu, fl_sigma, fl_low, fl_up = 6.4e9, 5e9, 1e9, 10e9         # local CPU freq dist

GLOBAL_CFG = {
    "F_t_server_cpu_capacity": F_t,
    "theta_max_delay_s": theta,
    "P_t_transmit_power_w": P_t,
    "P_I_compute_power_w": P_I,
    "kappa_energy_coeff": kappa,
    "B_bandwidth_hz": B,
    "N0_noise_power": N0,
    "input_size_mu": is_mu, "input_size_sigma": is_sigma,
    "input_size_low": is_low, "input_size_up": is_up,
    "f_local_mu": fl_mu, "f_local_sigma": fl_sigma,
    "f_local_low": fl_low, "f_local_up": fl_up,
}


def generate_scenario(server_num, user_num):
    """Replicate one MSMU-CO sample and return per-user structured records."""
    G = generate_graph(server_num, user_num)

    adj_mat = nx.adjacency_matrix(G).toarray()
    points = np.concatenate(
        (np.ones(server_num, dtype=np.int8), np.zeros(user_num, dtype=np.int8))
    )
    edges = np.array(
        [[list(G.nodes).index(u), list(G.nodes).index(v)] for u, v in G.edges()]
    )

    # ---- per-user raw features ----
    input_sizes = range_random(is_mu, is_sigma, user_num, is_low, is_up)
    required_cycles = input_sizes * 3e3
    f_locals = range_random(fl_mu, fl_sigma, user_num, fl_low, fl_up)
    alphas = np.random.rand(user_num)
    betas = 1 - alphas

    tau_locals = required_cycles / f_locals
    epsilon_locals = kappa * (f_locals ** 2) * input_sizes
    cost_locals_user = alphas * tau_locals + betas * epsilon_locals
    local_sat = np.where(theta > tau_locals, 1, 0)

    # ---- per-edge channel gains -> SINR -> uplink rate ----
    hs = np.random.rand(edges.shape[0])
    hs_squares = np.array(
        [np.sum(hs[np.where(edges[:, 1] == edges[i, 1])[0]] ** 2)
         for i in range(edges.shape[0])]
    )
    sinr = P_t * (hs ** 2) / (N0 + P_t * hs_squares)
    r_us = B * np.log2(1 + sinr)

    # expand user features along edges (matching MSMU-CO solver inputs)
    tmp_indices = edges[:, 0] - server_num
    cost_locals = cost_locals_user[tmp_indices]
    local_sat_e = local_sat[tmp_indices]
    input_sizes_e = input_sizes[tmp_indices]
    required_cycles_e = required_cycles[tmp_indices]
    alphas_e = alphas[tmp_indices]
    betas_e = 1.0 - alphas_e

    trans_costs = alphas_e * input_sizes_e / r_us + betas_e * P_t * input_sizes_e / r_us
    offload_total_costs = alphas_e * required_cycles_e / F_t + betas_e * P_I * required_cycles_e / F_t

    trans_sat = np.where(theta > input_sizes_e / r_us, 1, 0)
    offload_sat = np.where(theta > input_sizes_e / r_us + required_cycles_e / F_t, 1, 0)
    least_ratios = required_cycles_e / (theta - input_sizes_e / r_us) / F_t
    least_ratios = least_ratios * trans_sat * offload_sat

    # ---- MCMF-optimal solve ----
    res_cost, res_edges, res_adj, res_ratios = solve_once(
        server_num, user_num, adj_mat, points, edges,
        local_sat_e, offload_sat, cost_locals, trans_costs,
        offload_total_costs, least_ratios,
    )

    # map: user_index (0..user_num-1) -> assigned server, ratio
    assigned_server = {u: -1 for u in range(user_num)}   # -1 = local execution
    assigned_ratio = {u: np.nan for u in range(user_num)}
    if isinstance(res_edges, np.ndarray) and res_edges.size > 0:
        for k, (src, dst) in enumerate(res_edges):
            u = int(src) - server_num   # solver uses global node index
            assigned_server[u] = int(dst)
            if isinstance(res_ratios, np.ndarray) and k < res_ratios.size:
                assigned_ratio[u] = float(res_ratios[k])

    # best channel gain per user across all connected servers
    best_gain = {u: np.nan for u in range(user_num)}
    gain_to_assigned = {u: np.nan for u in range(user_num)}
    for ei in range(edges.shape[0]):
        u = int(edges[ei, 0]) - server_num
        s = int(edges[ei, 1])
        g = float(hs[ei])
        if np.isnan(best_gain[u]) or g > best_gain[u]:
            best_gain[u] = g
        if assigned_server[u] == s:
            gain_to_assigned[u] = g

    # number of servers each user is connected to
    conn_count = {u: 0 for u in range(user_num)}
    for ei in range(edges.shape[0]):
        conn_count[int(edges[ei, 0]) - server_num] += 1

    records = []
    for u in range(user_num):
        records.append({
            "user_id": u,
            "task_data_size": float(input_sizes[u]),
            "required_cpu_cycles": float(required_cycles[u]),
            "local_cpu_freq": float(f_locals[u]),
            "alpha_latency_weight": float(alphas[u]),
            "num_connected_servers": int(conn_count[u]),
            "best_channel_gain": float(best_gain[u]),
            "channel_gain_to_assigned": float(gain_to_assigned[u]),
            "mcmf_assigned_server": int(assigned_server[u]),
            "mcmf_resource_ratio": float(assigned_ratio[u]),
            "local_exec_satisfies_delay": int(local_sat[u]),
        })
    return records, float(res_cost)


def main():
    random.seed(42)
    np.random.seed(42)

    user_regimes = [10, 20]
    server_regimes = [3, 5]

    # ~625 scenarios per regime * 4 = 2500 total -> keep moderate at 200/regime = 800
    scenarios_per_regime = 200   # 4 regimes -> 800 scenarios total

    all_rows = []
    scenario_id = 0
    for S in server_regimes:
        for U in user_regimes:
            print(f"Generating {scenarios_per_regime} scenarios for S={S}, U={U} ...")
            for _ in range(scenarios_per_regime):
                recs, gt_cost = generate_scenario(S, U)
                for r in recs:
                    r.update({
                        "scenario_id": scenario_id,
                        "num_servers": S,
                        "num_users": U,
                        "scenario_mcmf_total_cost": gt_cost,
                        **GLOBAL_CFG,
                    })
                    all_rows.append(r)
                scenario_id += 1

    df = pd.DataFrame(all_rows)

    # column ordering
    lead = ["scenario_id", "num_servers", "num_users", "user_id"]
    feat = ["task_data_size", "required_cpu_cycles", "local_cpu_freq",
            "alpha_latency_weight", "num_connected_servers",
            "best_channel_gain", "channel_gain_to_assigned",
            "local_exec_satisfies_delay"]
    decision = ["mcmf_assigned_server", "mcmf_resource_ratio",
                "scenario_mcmf_total_cost"]
    cfg = list(GLOBAL_CFG.keys())
    df = df[lead + feat + decision + cfg]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, "base_msmu_raw.parquet")
    df.to_parquet(parquet_path, index=False)

    # also drop a human-readable config sidecar
    with open(os.path.join(out_dir, "base_msmu_config.json"), "w") as f:
        json.dump(GLOBAL_CFG, f, indent=2)

    # -------------------- DOCUMENTATION OUTPUT --------------------
    n_scenarios = df["scenario_id"].nunique()
    n_rows = len(df)
    print("\n" + "=" * 70)
    print("BASE MSMU DATASET — SUMMARY")
    print("=" * 70)
    print(f"Total scenarios : {n_scenarios}")
    print(f"Total rows      : {n_rows}  (one row per (scenario, user) pair)")
    print(f"Total users     : {int(df['num_users'].groupby(df['scenario_id']).first().sum())}")
    print(f"Table shape     : {df.shape}")
    print("\nScenarios per (S,U) regime:")
    print(df.groupby(['num_servers', 'num_users'])['scenario_id'].nunique().to_string())

    print("\nMain columns:")
    desc = {
        "scenario_id": "unique scenario index",
        "num_servers": "number of edge servers S in this scenario",
        "num_users": "number of users U in this scenario",
        "user_id": "user index within the scenario (0..U-1)",
        "task_data_size": "task input data size (bits)",
        "required_cpu_cycles": "CPU cycles required for the task",
        "local_cpu_freq": "local device CPU frequency (cycles/s)",
        "alpha_latency_weight": "latency-vs-energy tradeoff weight (0..1)",
        "num_connected_servers": "how many servers this user can reach",
        "best_channel_gain": "best channel gain across connected servers",
        "channel_gain_to_assigned": "channel gain to the MCMF-assigned server (NaN if local)",
        "local_exec_satisfies_delay": "1 if local execution meets delay bound",
        "mcmf_assigned_server": "MCMF-optimal server index (-1 = local execution)",
        "mcmf_resource_ratio": "resource allocation ratio at assigned server",
        "scenario_mcmf_total_cost": "MCMF-optimal total cost for the scenario",
    }
    for k, v in desc.items():
        print(f"  - {k:<30} {v}")
    print("  - <global config columns>     server/global params: "
          + ", ".join(GLOBAL_CFG.keys()))

    offloaded = (df["mcmf_assigned_server"] >= 0).mean() * 100
    print(f"\nShare of users offloaded (vs local): {offloaded:.1f}%")
    print(f"\nSaved dataset  -> {parquet_path}")
    print(f"Saved config   -> {os.path.join(out_dir, 'base_msmu_config.json')}")


if __name__ == "__main__":
    main()
