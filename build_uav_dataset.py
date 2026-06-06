"""
Augment the subtask-level MSMU datasets with UAV-supported MEC semantics.

No ML training. Existing labels (exec_location_label, exec_is_local,
exec_server_index) are NOT modified.

Pipeline
--------
B. Per scenario, designate some server indices as UAVs (rest ground MEC) and
   assign 3D positions in a [0,1000] m^2 region (UAV altitude 80-150 m, ground 0).
C. Per (scenario, user), assign 2D ground positions (z=0).
D. For each subtask, join the user position and its assigned server (the
   oracle anchor, exec_server_index); compute 3D/2D distance, elevation angle,
   a UAV LoS probability, and an effective uplink rate.
E. Add scenario-level UAV context (num_uav_servers, min distance to any UAV,
   coverage flag).
F. Save train/val/test UAV-augmented parquet.

All geometry uses a single global RNG seed for reproducibility. Server and user
position tables are built ONCE over the union of all scenarios so the same
scenario gets identical geometry across splits (splits are disjoint by
scenario_id anyway, but this keeps it deterministic and join-safe).
"""

import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(BASE, "data", "subtask")
OUT = os.path.join(BASE, "data", "uav_subtask")
CFG = os.path.join(BASE, "data", "raw", "base_msmu_config.json")

SEED = 20260606
REGION = 1000.0          # square region side (meters)
UAV_Z_LOW, UAV_Z_HIGH = 80.0, 150.0   # UAV altitude range (m)
R_COVERAGE = 600.0       # UAV coverage radius for has_uav_in_coverage (m, 3D)
EPS = 1e-9

# Urban-micro UAV LoS probability (sigmoid in elevation angle), ITU/3GPP-style:
#   los_prob = 1 / (1 + A * exp(-B*(theta_deg - A)))
LOS_A, LOS_B = 9.61, 0.16     # typical urban environment parameters

# Path-loss / channel model constants (free-space-like, documented in README)
PL_REF_GAIN = 1e-3       # reference channel power gain at 1 m (beta_0)
PL_EXP = 2.2             # path-loss exponent
EXTRA_NLOS_ATTEN = 0.2   # multiplicative gain penalty applied to the NLoS share


def build_server_table(scen_meta, rng):
    """scen_meta: dict scenario_id -> num_servers. Returns long DataFrame."""
    rows = []
    for sid, nsrv in scen_meta.items():
        # choose UAV servers
        if nsrv >= 2:
            k = rng.integers(1, 3)            # 1 or 2 UAVs
            k = min(k, nsrv)
            uav_idx = set(rng.choice(nsrv, size=k, replace=False).tolist())
        else:  # nsrv == 1
            uav_idx = {0} if rng.random() < 0.5 else set()
        for s in range(nsrv):
            is_uav = 1 if s in uav_idx else 0
            x = rng.uniform(0, REGION)
            y = rng.uniform(0, REGION)
            z = rng.uniform(UAV_Z_LOW, UAV_Z_HIGH) if is_uav else 0.0
            rows.append({
                "scenario_id": sid, "server_index": s,
                "is_uav_server": is_uav,
                "server_x": x, "server_y": y, "server_z": z,
            })
    return pd.DataFrame(rows)


def build_user_table(user_meta, rng):
    """user_meta: list of (scenario_id, user_id). Returns long DataFrame."""
    rows = []
    for sid, uid in user_meta:
        rows.append({
            "scenario_id": sid, "user_id": uid,
            "user_x": rng.uniform(0, REGION),
            "user_y": rng.uniform(0, REGION),
            "user_z": 0.0,
        })
    return pd.DataFrame(rows)


def los_probability(elev_deg):
    return 1.0 / (1.0 + LOS_A * np.exp(-LOS_B * (elev_deg - LOS_A)))


def main():
    cfg = json.load(open(CFG))
    P_tx = cfg["P_t_transmit_power_w"]

    splits = {nm: pd.read_parquet(os.path.join(SUB, f"{nm}_subtask.parquet"))
              for nm in ["train", "val", "test"]}
    full = pd.concat(splits.values(), ignore_index=True)

    # ---- B. server table (over union of scenarios) ----
    scen_meta = (full.groupby("scenario_id")["num_servers"].first().to_dict())
    rng = np.random.default_rng(SEED)
    srv = build_server_table(scen_meta, rng)

    # ---- C. user table ----
    user_meta = (full[["scenario_id", "user_id"]].drop_duplicates()
                 .sort_values(["scenario_id", "user_id"])
                 .itertuples(index=False, name=None))
    usr = build_user_table(list(user_meta), rng)

    # ---- E. scenario-level UAV context, precomputed per (scenario,user) ----
    num_uav = (srv.groupby("scenario_id")["is_uav_server"].sum()
               .rename("num_uav_servers").reset_index())

    # min distance from each user to any UAV server in its scenario
    uav_srv = srv[srv["is_uav_server"] == 1]
    u_join = usr.merge(uav_srv, on="scenario_id", how="left")
    u_join["d3"] = np.sqrt(
        (u_join["server_x"] - u_join["user_x"]) ** 2
        + (u_join["server_y"] - u_join["user_y"]) ** 2
        + (u_join["server_z"] - u_join["user_z"]) ** 2
    )
    min_uav = (u_join.groupby(["scenario_id", "user_id"])["d3"]
               .min().rename("min_dist_to_any_uav").reset_index())

    def augment(df):
        d = df.copy()
        # join user position
        d = d.merge(usr, on=["scenario_id", "user_id"], how="left")
        # join assigned server (exec_server_index); -1 will not match -> NaN
        d = d.merge(
            srv.rename(columns={"server_index": "exec_server_index"}),
            on=["scenario_id", "exec_server_index"], how="left",
        )

        remote = d["exec_server_index"] >= 0

        # ---- D. geometry to assigned server ----
        dx = d["server_x"] - d["user_x"]
        dy = d["server_y"] - d["user_y"]
        dz = d["server_z"] - d["user_z"]
        dist_2d = np.sqrt(dx ** 2 + dy ** 2)
        dist_3d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        elev = np.arctan2(dz, dist_2d + EPS)          # radians
        elev_deg = np.degrees(elev)

        d["assigned_dist_2d"] = np.where(remote, dist_2d, np.nan)
        d["assigned_dist_3d"] = np.where(remote, dist_3d, np.nan)
        d["assigned_elevation_angle"] = np.where(remote, elev, np.nan)  # radians

        # assigned_is_uav_server: 0 for local (consistent), else server flag
        d["assigned_is_uav_server"] = np.where(
            remote, d["is_uav_server"].fillna(0).astype(int), 0
        )

        # ---- D.3 LoS probability ----
        # UAV server: sigmoid LoS in elevation. Ground server: LoS=1.0
        # (terrestrial link, treated as deterministic LoS). Local: NaN.
        los = np.where(
            remote & (d["assigned_is_uav_server"] == 1),
            los_probability(elev_deg),
            np.where(remote, 1.0, np.nan),
        )
        d["assigned_los_prob"] = los

        # ---- D.4 effective rate ----
        # Channel power gain: free-space-like h = beta0 / dist^exp, blended by
        # LoS prob (NLoS share is extra-attenuated):
        #   h_eff = h_base * (los + (1-los)*EXTRA_NLOS_ATTEN)
        # SINR = P_tx * h_eff / N0 ; R = B * log2(1 + SINR)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_base = PL_REF_GAIN / np.power(np.maximum(dist_3d, 1.0), PL_EXP)
        los_factor = np.where(np.isnan(los), 1.0, los + (1 - np.nan_to_num(los)) * EXTRA_NLOS_ATTEN)
        h_eff = h_base * los_factor
        sinr = P_tx * h_eff / d["N0"]
        rate = d["B"] * np.log2(1.0 + sinr)
        d["assigned_effective_rate"] = np.where(remote, rate, np.nan)

        # ---- E. global UAV context ----
        d = d.merge(num_uav, on="scenario_id", how="left")
        d = d.merge(min_uav, on=["scenario_id", "user_id"], how="left")
        d["num_uav_servers"] = d["num_uav_servers"].fillna(0).astype(int)
        d["has_uav_in_coverage"] = (
            d["min_dist_to_any_uav"] <= R_COVERAGE
        ).astype(int)

        # tidy: rename joined server pos cols (already server_x/y/z), drop helper
        d = d.rename(columns={"is_uav_server": "_srv_is_uav"}).drop(
            columns=["_srv_is_uav"]
        )
        return d

    aug = {nm: augment(df) for nm, df in splits.items()}

    os.makedirs(OUT, exist_ok=True)
    # also persist the raw geometry tables for reuse
    srv.to_parquet(os.path.join(OUT, "server_positions.parquet"), index=False)
    usr.to_parquet(os.path.join(OUT, "user_positions.parquet"), index=False)

    paths = {}
    for nm in ["train", "val", "test"]:
        p = os.path.join(OUT, f"{nm}_uav.parquet")
        aug[nm].to_parquet(p, index=False)
        paths[nm] = p

    # ------------------- REPORT -------------------
    print("=" * 74)
    print("UAV-AUGMENTED SUBTASK DATASET — REPORT")
    print("=" * 74)
    for nm in ["train", "val", "test"]:
        d = aug[nm]
        print(f"[{nm:5}] shape={d.shape}  scenarios={d.scenario_id.nunique()}  "
              f"subtasks={len(d)}")
    alld = pd.concat(aug.values(), ignore_index=True)
    rem = alld[alld["exec_server_index"] >= 0]

    print("\nNew UAV feature ranges (over offloaded/remote subtasks):")
    def rng_line(col, deg=False):
        s = rem[col].dropna()
        if deg:
            s = np.degrees(s)
        print(f"  {col:28s} min={s.min():.4f}  mean={s.mean():.4f}  max={s.max():.4f}"
              + ("  (deg)" if deg else ""))
    rng_line("assigned_dist_3d")
    rng_line("assigned_dist_2d")
    rng_line("assigned_elevation_angle", deg=True)
    rng_line("assigned_los_prob")
    rng_line("assigned_effective_rate")
    mu = alld["min_dist_to_any_uav"].dropna()
    print(f"  {'min_dist_to_any_uav':28s} min={mu.min():.4f}  mean={mu.mean():.4f}  max={mu.max():.4f}")

    print("\nAmong OFFLOADED subtasks, anchor-server type:")
    vc = rem["assigned_is_uav_server"].value_counts().sort_index()
    tot = len(rem)
    print(f"  ground (0): {vc.get(0,0)} ({vc.get(0,0)/tot*100:.1f}%)")
    print(f"  UAV    (1): {vc.get(1,0)} ({vc.get(1,0)/tot*100:.1f}%)")
    print(f"\nnum_uav_servers per scenario distribution:")
    print(srv.groupby('scenario_id')['is_uav_server'].sum().value_counts().sort_index().to_string())
    print(f"\nhas_uav_in_coverage (R={R_COVERAGE} m) over all rows: "
          f"{alld['has_uav_in_coverage'].mean()*100:.1f}% =1")

    print("\nLabel integrity (unchanged):")
    for nm in ["train", "val", "test"]:
        a, b = splits[nm], aug[nm]
        same = (a["exec_location_label"].values == b["exec_location_label"].values).all() \
            and (a["exec_server_index"].values == b["exec_server_index"].values).all()
        print(f"  {nm}: labels identical = {same}")

    print("\nSaved:")
    for nm, p in paths.items():
        print(f"  {nm:5}: {p}")
    print(f"  geometry: {os.path.join(OUT,'server_positions.parquet')}, "
          f"{os.path.join(OUT,'user_positions.parquet')}")


if __name__ == "__main__":
    main()
