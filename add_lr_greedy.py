"""
Literature-inspired LR/DDLO-style GREEDY offloading baseline ("lr_greedy").

Per scenario, decide each user's all-local vs all-offload(to s_star) pattern by
greedily admitting the highest-cost-reduction (gain) users to their anchor
server subject to a server capacity constraint, then re-cost everything with the
SAME cost model used to generate the oracle labels (build_subtask_dataset.py).

Does NOT retrain the MLP or modify datasets/labels. Appends a new method to
output/results_metrics.csv and regenerates summary + LaTeX + plots.
"""

import os
import json
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "uav_subtask")
OUTPUT = os.path.join(BASE, "output")
CFG = os.path.join(BASE, "data", "raw", "base_msmu_config.json")
SEED = 42
CAPACITY_THRESHOLD = 1.2   # server admissible total resource share
T_NORM = 1.0               # normalization horizon for resource demand (s)

# ---- import EXACT cost model ----
spec = importlib.util.spec_from_file_location(
    "bsd", os.path.join(BASE, "build_subtask_dataset.py"))
bsd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsd)
uplink_rate = bsd.uplink_rate
pattern_cost = bsd.pattern_cost

cfg = json.load(open(CFG))
P_tx = cfg["P_t_transmit_power_w"]
kappa = cfg["kappa_energy_coeff"]

# ======================================================================
# Load
# ======================================================================
te = pd.read_parquet(os.path.join(DATA, "test_uav.parquet"))
existing = pd.read_csv(os.path.join(OUTPUT, "results_metrics.csv"))

n_users_csv = (existing["method_name"] == "oracle").sum()
n_users_te = te.groupby(["scenario_id", "user_id"]).ngroups
print(f"Test users: csv={n_users_csv}  test_uav={n_users_te}  match={n_users_csv == n_users_te}")

# deadline: same definition as Job 4 = 1.5x median oracle latency
DEADLINE = 1.5 * float(existing[existing.method_name == "oracle"]["latency"].median())
print(f"Deadline threshold (1.5x median oracle latency): {DEADLINE:.4f} s")


def user_params(g):
    """Extract per-user constants needed by the cost model."""
    g = g.sort_values("subtask_id")
    s_star = int(g["exec_server_index"].max())   # -1 if all local, else server idx
    has_anchor = s_star != -1
    f_l = float(g["local_cpu_freq"].iloc[0])
    F_t = float(g["F_t"].iloc[0])
    B = float(g["B"].iloc[0])
    N0 = float(g["N0"].iloc[0])
    h = float(g["best_channel_gain"].iloc[0])
    alpha_u = float(g["alpha_latency_weight"].iloc[0])
    nu = int(g["num_users"].iloc[0])
    anchor_is_uav = int(g["assigned_is_uav_server"].max()) if has_anchor else 0
    sub_cycles = g["subtask_cpu_cycles"].values
    sub_data = g["subtask_data_size"].values
    R = uplink_rate(B, P_tx, h, N0)
    ratio = g["mcmf_resource_ratio"].dropna()
    share = float(ratio.iloc[0]) if (has_anchor and len(ratio) and ratio.iloc[0] > 0) \
        else 1.0 / max(nu, 1)
    return dict(g=g, s_star=s_star, has_anchor=has_anchor, f_l=f_l, F_t=F_t,
                R=R, share=share, alpha_u=alpha_u, nu=nu,
                anchor_is_uav=anchor_is_uav,
                sub_cycles=sub_cycles, sub_data=sub_data)


# ======================================================================
# B/C. Greedy LR-style decision per scenario
# ======================================================================
K = len(te["subtask_id"].unique())
rows = []                # per-user metric rows for lr_greedy
pred_labels_all = []     # (index, subtask predicted label) aligned to te order
te = te.reset_index(drop=True)
te["_lr_pred"] = 0       # default local

for sid, scen in te.groupby("scenario_id"):
    # gather per-user info
    infos = {}
    for uid, g in scen.groupby("user_id"):
        infos[uid] = user_params(g)

    server_load = {}     # s_star -> accumulated resource demand

    # candidate offloaders: has_anchor and positive gain
    candidates = []
    for uid, p in infos.items():
        n = len(p["sub_cycles"])
        local_mask = np.zeros(n, dtype=bool)
        offl_mask = (np.ones(n, dtype=bool) if p["has_anchor"]
                     else np.zeros(n, dtype=bool))
        c_local, _, _ = pattern_cost(local_mask, p["sub_cycles"], p["sub_data"],
                                     p["f_l"], p["F_t"], p["R"], P_tx, kappa,
                                     p["share"], p["alpha_u"])
        c_offl, _, _ = pattern_cost(offl_mask, p["sub_cycles"], p["sub_data"],
                                    p["f_l"], p["F_t"], p["R"], P_tx, kappa,
                                    p["share"], p["alpha_u"])
        gain = c_local - c_offl
        # resource demand (dimensionless): cycles / (F_t * T_norm); fall back to share
        rd = float(np.sum(p["sub_cycles"]) / (p["F_t"] * T_NORM))
        if not np.isfinite(rd) or rd <= 0:
            rd = p["share"]
        p["gain"], p["rd"], p["chosen_offload"] = gain, rd, False
        if p["has_anchor"] and gain > 0:
            candidates.append(uid)

    # greedy: descending gain, admit if capacity allows
    candidates.sort(key=lambda u: infos[u]["gain"], reverse=True)
    for uid in candidates:
        p = infos[uid]
        s = p["s_star"]
        load = server_load.get(s, 0.0)
        if load + p["rd"] <= CAPACITY_THRESHOLD:
            p["chosen_offload"] = True
            server_load[s] = load + p["rd"]
        # else stays local

    # emit per-user costs + per-subtask predicted labels
    for uid, p in infos.items():
        g = p["g"]
        n = len(p["sub_cycles"])
        offload = p["chosen_offload"]
        mask = (np.ones(n, dtype=bool) if offload else np.zeros(n, dtype=bool))
        cost, lat, en = pattern_cost(mask, p["sub_cycles"], p["sub_data"],
                                     p["f_l"], p["F_t"], p["R"], P_tx, kappa,
                                     p["share"], p["alpha_u"])
        rows.append({
            "method_name": "lr_greedy", "scenario_id": sid, "user_id": uid,
            "anchor_is_uav": p["anchor_is_uav"],
            "has_anchor": int(p["has_anchor"]),
            "latency": lat, "energy": en, "cost": cost,
            "deadline_violation": int(lat > DEADLINE),
        })
        # subtask labels in this group's index order
        lab = 1 if offload else 0
        for idx in g.index:
            te.at[idx, "_lr_pred"] = lab

lr = pd.DataFrame(rows)

# ======================================================================
# Subtask-level classification metrics (lr_greedy vs oracle label)
# ======================================================================
y_true = te["exec_location_label"].astype(int).values
y_pred = te["_lr_pred"].astype(int).values
acc = (y_true == y_pred).mean()
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0)
try:
    auc = roc_auc_score(y_true, y_pred)   # hard labels -> AUC == balanced sep
except ValueError:
    auc = float("nan")
print(f"\nlr_greedy subtask metrics: acc={acc:.4f} prec={prec:.4f} "
      f"rec={rec:.4f} f1={f1:.4f} auc={auc:.4f}")

# ======================================================================
# D. Merge into results_metrics.csv
# ======================================================================
# drop any prior lr_greedy rows for idempotency
existing = existing[existing.method_name != "lr_greedy"]
merged = pd.concat([existing, lr[existing.columns]], ignore_index=True)
merged.to_csv(os.path.join(OUTPUT, "results_metrics.csv"), index=False)

# ---- summary metrics ----
ORDER = ["oracle", "model", "all_local", "all_mcmf", "random", "lr_greedy"]
SUB_ACC = {"oracle": 1.0, "model": None, "lr_greedy": acc}
# preserve model accuracy from existing summary if present
prev_summ_path = os.path.join(OUTPUT, "summary_metrics.csv")
prev_acc = {}
if os.path.exists(prev_summ_path):
    ps = pd.read_csv(prev_summ_path)
    for _, rr in ps.iterrows():
        if pd.notna(rr.get("subtask_accuracy")):
            prev_acc[rr["method"]] = rr["subtask_accuracy"]

summ_rows = []
for m in ORDER:
    s = merged[merged.method_name == m]
    if len(s) == 0:
        continue
    sub_acc = SUB_ACC.get(m, None)
    if sub_acc is None:
        sub_acc = prev_acc.get(m, np.nan)
    summ_rows.append({
        "method": m,
        "latency_mean": s.latency.mean(), "latency_std": s.latency.std(),
        "energy_mean": s.energy.mean(), "energy_std": s.energy.std(),
        "cost_mean": s.cost.mean(), "cost_std": s.cost.std(),
        "deadline_viol_rate": s.deadline_violation.mean(),
        "subtask_accuracy": sub_acc,
    })
summ = pd.DataFrame(summ_rows)
summ.to_csv(prev_summ_path, index=False)

# ---- LaTeX table ----
pretty = {"oracle": "Oracle", "model": "Edge-AI (MLP)", "all_local": "All-Local",
          "all_mcmf": "All-MCMF", "random": "Random", "lr_greedy": "LR-Greedy"}
lines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Test-set performance of selective subtask offloading methods "
    r"(per-user means $\pm$ std; UAV-augmented MSMU dataset).}",
    r"\label{tab:offloading_results}",
    r"\begin{tabular}{lccccc}", r"\hline",
    r"Method & Latency (s) & Energy (J) & Cost & Deadline Viol. & Subtask Acc. \\",
    r"\hline",
]
for _, r in summ.iterrows():
    a = "--" if pd.isna(r.subtask_accuracy) else f"{r.subtask_accuracy:.3f}"
    lines.append(
        f"{pretty[r.method]} & {r.latency_mean:.3f} $\\pm$ {r.latency_std:.3f} & "
        f"{r.energy_mean:.3f} $\\pm$ {r.energy_std:.3f} & "
        f"{r.cost_mean:.3f} $\\pm$ {r.cost_std:.3f} & "
        f"{r.deadline_viol_rate:.3f} & {a} \\\\")
lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
with open(os.path.join(OUTPUT, "summary_table.tex"), "w") as f:
    f.write("\n".join(lines))

# ======================================================================
# E. Regenerate plots (all methods including lr_greedy)
# ======================================================================
COLORS = {"oracle": "#2ca02c", "model": "#1f77b4", "all_local": "#d62728",
          "all_mcmf": "#ff7f0e", "random": "#9467bd", "lr_greedy": "#8c564b"}

# latency CDF
plt.figure(figsize=(7, 5), dpi=150)
for m in ["oracle", "model", "all_local", "all_mcmf", "random", "lr_greedy"]:
    s = np.sort(merged[merged.method_name == m]["latency"].values)
    cdf = np.arange(1, len(s) + 1) / len(s)
    plt.plot(s, cdf, label=pretty[m], color=COLORS[m], lw=2)
plt.xlabel("Per-user latency (s)"); plt.ylabel("CDF")
plt.title("User Latency CDF"); plt.legend(); plt.grid(alpha=0.3); plt.xlim(left=0)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "latency_cdf.png")); plt.close()

# energy bar
plt.figure(figsize=(8, 5), dpi=150)
means = [merged[merged.method_name == m]["energy"].mean() for m in ORDER]
stds = [merged[merged.method_name == m]["energy"].std() for m in ORDER]
plt.bar([pretty[m] for m in ORDER], means, yerr=stds, capsize=4,
        color=[COLORS[m] for m in ORDER])
plt.ylabel("Avg per-user energy (J)"); plt.title("Average Energy per User")
plt.xticks(rotation=20); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "energy_bar.png")); plt.close()

# deadline bar
plt.figure(figsize=(8, 5), dpi=150)
dv = [merged[merged.method_name == m]["deadline_violation"].mean() for m in ORDER]
plt.bar([pretty[m] for m in ORDER], dv, color=[COLORS[m] for m in ORDER])
plt.ylabel("Deadline violation rate"); plt.title("Deadline Violation Rate")
plt.xticks(rotation=20); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "deadline_bar.png")); plt.close()

# uav vs ground latency: Edge-AI and LR-Greedy, offloaded-anchor users
plt.figure(figsize=(7, 5), dpi=150)
style = {"model": ("Edge-AI", "-"), "lr_greedy": ("LR-Greedy", "--")}
uav_col = {1: "#1f77b4", 0: "#ff7f0e"}
for m, (lab, ls) in style.items():
    sub = merged[(merged.method_name == m) & (merged.has_anchor == 1)]
    for flag, fl_lab in [(1, "UAV"), (0, "Ground")]:
        s = np.sort(sub[sub.anchor_is_uav == flag]["latency"].values)
        if len(s):
            cdf = np.arange(1, len(s) + 1) / len(s)
            plt.plot(s, cdf, ls, color=uav_col[flag], lw=2,
                     label=f"{lab} – {fl_lab} (n={len(s)})")
plt.xlabel("Per-user latency (s)"); plt.ylabel("CDF")
plt.title("UAV vs Ground Anchor Latency"); plt.legend(fontsize=8)
plt.grid(alpha=0.3); plt.xlim(left=0)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "uav_vs_ground_latency.png")); plt.close()

# ======================================================================
# F. Report
# ======================================================================
def mstat(m, col):
    s = merged[merged.method_name == m][col]
    return s.mean(), s.std()

print("\n" + "=" * 70)
print("LR-GREEDY FINAL REPORT")
print("=" * 70)
print(f"Subtask: acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} "
      f"f1={f1:.4f} auc={auc:.4f}")
lm, ls = mstat("lr_greedy", "latency")
em, es = mstat("lr_greedy", "energy")
cm, cs = mstat("lr_greedy", "cost")
dv_lr = merged[merged.method_name == "lr_greedy"]["deadline_violation"].mean()
print(f"User-level: latency={lm:.3f}±{ls:.3f}  energy={em:.3f}±{es:.3f}  "
      f"cost={cm:.3f}±{cs:.3f}  deadline_viol={dv_lr:.3f}")
print()
print(summ.to_string(index=False,
      columns=["method", "latency_mean", "energy_mean", "cost_mean",
               "deadline_viol_rate", "subtask_accuracy"]))

c_local = mstat("all_local", "cost")[0]
c_rand = mstat("random", "cost")[0]
c_oracle = mstat("oracle", "cost")[0]
c_model = mstat("model", "cost")[0]
print(f"\nLR-Greedy cost vs All-Local: {(1 - cm/c_local)*100:.1f}% lower; "
      f"vs Random: {(1 - cm/c_rand)*100:.1f}% lower; "
      f"vs Oracle: {(cm/c_oracle - 1)*100:.1f}% higher; "
      f"vs Edge-AI: {(cm/c_model - 1)*100:+.1f}%.")

print("\nUpdated paths:")
for p in ["results_metrics.csv", "summary_metrics.csv", "summary_table.tex",
          "latency_cdf.png", "energy_bar.png", "deadline_bar.png",
          "uav_vs_ground_latency.png"]:
    print(f"  {os.path.join(OUTPUT, p)}")
