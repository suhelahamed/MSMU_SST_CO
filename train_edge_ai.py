"""
Edge-AI selective subtask offloading: train an MLP classifier on the
UAV-augmented dataset, evaluate vs baselines using the SAME cost model as
build_subtask_dataset.py, and emit paper-ready CSV / LaTeX / plots.

Does NOT modify datasets or labels. Predictions are scored against
exec_location_label and re-costed with the original generator cost model.
"""

import os
import json
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "uav_subtask")
CFG = os.path.join(BASE, "data", "raw", "base_msmu_config.json")
MODELS = os.path.join(BASE, "models")
OUTPUT = os.path.join(BASE, "output")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- import the EXACT cost model from build_subtask_dataset.py ----
spec = importlib.util.spec_from_file_location(
    "bsd", os.path.join(BASE, "build_subtask_dataset.py"))
bsd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsd)
uplink_rate = bsd.uplink_rate
pattern_cost = bsd.pattern_cost          # (mask, cycles, data, f_l, F_t, R, P_tx, kappa, share, alpha_u)
K = bsd.K
OVERHEAD_S = bsd.OVERHEAD_S

cfg = json.load(open(CFG))
P_tx = cfg["P_t_transmit_power_w"]
kappa = cfg["kappa_energy_coeff"]

# ======================================================================
# B. LOAD + FEATURE MATRICES
# ======================================================================
ID_COLS = ["scenario_id", "user_id", "subtask_id", "num_users"]
LABEL_COLS = ["exec_location_label", "exec_is_local", "exec_server_index"]
# columns that are non-numeric / bookkeeping and should not be raw features
DROP_EXTRA = ["oracle_pattern", "mcmf_assigned_server"]

# LEAKAGE GUARD: the oracle label is (in this dataset) perfectly determined by
# whether the user has an MCMF anchor (exec_server_index >= 0). The
# assigned-server geometry features are NaN iff the row is local, so they leak
# the label one-to-one. To make this a GENUINE prediction problem, we exclude
# those anchor-derived features and the is_local indicator from the model
# inputs. The model must decide offloading from task / subtask / channel /
# UAV-context features only. These columns are still used for COST evaluation.
LEAK_COLS = [
    "server_x", "server_y", "server_z",
    "assigned_dist_2d", "assigned_dist_3d", "assigned_elevation_angle",
    "assigned_los_prob", "assigned_effective_rate",
    "assigned_is_uav_server", "mcmf_resource_ratio",
]

# assigned-server features that are NaN for local rows -> fill 0 (eval only)
NAN_FILL_COLS = [
    "server_x", "server_y", "server_z",
    "assigned_dist_2d", "assigned_dist_3d", "assigned_elevation_angle",
    "assigned_los_prob", "assigned_effective_rate", "mcmf_resource_ratio",
]


def load_split(name):
    return pd.read_parquet(os.path.join(DATA, f"{name}_uav.parquet"))


train_df = load_split("train")
val_df = load_split("val")
test_df = load_split("test")

print(f"Loaded: train={train_df.shape} val={val_df.shape} test={test_df.shape}")
print(f"Device: {DEVICE}")

# build feature list from train columns (exclude leakage features)
all_cols = list(train_df.columns)
feature_cols = [c for c in all_cols
                if c not in ID_COLS + LABEL_COLS + DROP_EXTRA + LEAK_COLS]
# keep only numeric/bool
num_feats = [c for c in feature_cols
             if pd.api.types.is_numeric_dtype(train_df[c])
             or pd.api.types.is_bool_dtype(train_df[c])]


def make_features(df):
    d = df.copy()
    feats = num_feats
    X = d[feats].astype(float).values
    y = d["exec_location_label"].astype(float).values
    return X, y, feats


Xtr, ytr, FEATS = make_features(train_df)
Xva, yva, _ = make_features(val_df)
Xte, yte, _ = make_features(test_df)

scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xva_s = scaler.transform(Xva)
Xte_s = scaler.transform(Xte)
print(f"Feature dim: {Xtr_s.shape[1]}  | n_features used: {len(FEATS)}")
print(f"Label balance (train): mean={ytr.mean():.3f}")

# dataloaders
def make_loader(X, y, bs, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


train_loader = make_loader(Xtr_s, ytr, 1024, True)
val_loader = make_loader(Xva_s, yva, 2048, False)

# ======================================================================
# C. MODEL
# ======================================================================
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


model = MLP(Xtr_s.shape[1]).to(DEVICE)
# class imbalance handling via pos_weight
pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                     dtype=torch.float32, device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ======================================================================
# D. TRAIN
# ======================================================================
MAX_EPOCHS, PATIENCE = 30, 5
best_val, best_state, wait = float("inf"), None, 0
history = []
MODEL_PATH = os.path.join(MODELS, "subtask_uav_mlp.pt")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_loader.dataset)

    model.eval()
    va_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logit = model(xb)
            va_loss += criterion(logit, yb).item() * xb.size(0)
            pred = (torch.sigmoid(logit) >= 0.5).float()
            correct += (pred == yb).sum().item()
            n += xb.size(0)
    va_loss /= len(val_loader.dataset)
    va_acc = correct / n
    history.append({"epoch": epoch, "train_loss": tr_loss,
                    "val_loss": va_loss, "val_accuracy": va_acc})
    print(f"epoch {epoch:2d}  train_loss={tr_loss:.4f}  "
          f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")

    if va_loss < best_val - 1e-5:
        best_val, best_state, wait = va_loss, {k: v.cpu().clone()
                                               for k, v in model.state_dict().items()}, 0
        torch.save({"state_dict": best_state, "features": FEATS,
                    "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
                   MODEL_PATH)
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
            break

pd.DataFrame(history).to_csv(os.path.join(OUTPUT, "training_history.csv"), index=False)
model.load_state_dict(best_state)
model.eval()

# ======================================================================
# E. EVALUATION
# ======================================================================
# model predictions on test
with torch.no_grad():
    logits = model(torch.tensor(Xte_s, dtype=torch.float32, device=DEVICE))
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
pred_label = (probs >= 0.5).astype(int)

subtask_acc = (pred_label == yte.astype(int)).mean()
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
prec, rec, f1, _ = precision_recall_fscore_support(
    yte.astype(int), pred_label, average="binary", zero_division=0)
try:
    auc = roc_auc_score(yte.astype(int), probs)
except ValueError:
    auc = float("nan")
print(f"\nTest subtask-level accuracy (model vs oracle): {subtask_acc:.4f}")
print(f"  precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} auc={auc:.4f}")

# attach predictions back to test rows
te = test_df.copy().reset_index(drop=True)
te["pred_label"] = pred_label

# random baseline mask (per subtask, tied to s_star feasibility)
rng = np.random.default_rng(SEED)
te["rand_label"] = rng.integers(0, 2, size=len(te))


def server_share_of(row):
    s_star = int(row["exec_server_index"])
    ratio = row["mcmf_resource_ratio"]
    nu = int(row["num_users"])
    if s_star != -1 and pd.notna(ratio) and ratio > 0:
        return float(ratio)
    return 1.0 / max(nu, 1)


# ---- per-user cost evaluation reusing the EXACT cost model ----
# We need the original best_channel_gain-based rate R used at label time.
methods = ["oracle", "model", "all_local", "all_mcmf", "random"]
rows = []

# group test by user
grp = te.groupby(["scenario_id", "user_id"], sort=False)
for (sid, uid), g in grp:
    g = g.sort_values("subtask_id")
    s_star = int(g["exec_server_index"].iloc[0]) if (g["exec_location_label"] > 0).any() \
        else int(g["exec_server_index"].iloc[0])
    # s_star is constant within user (anchor). For oracle local users it's -1.
    s_star = int(g["exec_server_index"].max())  # -1 if all local, else server idx
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
    # server share: use mcmf ratio if available, else 1/num_users
    ratio = g["mcmf_resource_ratio"].dropna()
    share = float(ratio.iloc[0]) if (has_anchor and len(ratio) and ratio.iloc[0] > 0) else 1.0 / max(nu, 1)

    # masks per method (only meaningful if has_anchor; else all local)
    def mask_from(labels):
        m = np.asarray(labels, dtype=bool)
        if not has_anchor:
            m = np.zeros(len(labels), dtype=bool)
        return m

    masks = {
        "oracle": mask_from(g["exec_location_label"].values),
        "model": mask_from(g["pred_label"].values),
        "all_local": np.zeros(len(g), dtype=bool),
        "all_mcmf": (np.ones(len(g), dtype=bool) if has_anchor else np.zeros(len(g), dtype=bool)),
        "random": mask_from(g["rand_label"].values),
    }

    for mname, m in masks.items():
        cost, lat, en = pattern_cost(m, sub_cycles, sub_data, f_l, F_t, R,
                                     P_tx, kappa, share, alpha_u)
        rows.append({
            "method_name": mname, "scenario_id": sid, "user_id": uid,
            "anchor_is_uav": anchor_is_uav, "has_anchor": int(has_anchor),
            "latency": lat, "energy": en, "cost": cost,
        })

res = pd.DataFrame(rows)

# ---- deadline: 1.5x the median ORACLE latency (per-user threshold = global) ----
oracle_lat = res[res.method_name == "oracle"]["latency"]
DEADLINE = 1.5 * float(oracle_lat.median())
res["deadline_violation"] = (res["latency"] > DEADLINE).astype(int)
print(f"Deadline threshold (1.5x median oracle latency): {DEADLINE:.4f} s")

res.to_csv(os.path.join(OUTPUT, "results_metrics.csv"), index=False)

# ======================================================================
# F. SUMMARY TABLE + PLOTS
# ======================================================================
def summarize(df):
    out = []
    for m in methods:
        s = df[df.method_name == m]
        out.append({
            "method": m,
            "latency_mean": s.latency.mean(), "latency_std": s.latency.std(),
            "energy_mean": s.energy.mean(), "energy_std": s.energy.std(),
            "cost_mean": s.cost.mean(), "cost_std": s.cost.std(),
            "deadline_viol_rate": s.deadline_violation.mean(),
        })
    return pd.DataFrame(out)


summ = summarize(res)
summ["subtask_accuracy"] = np.nan
summ.loc[summ.method == "model", "subtask_accuracy"] = subtask_acc
summ.loc[summ.method == "oracle", "subtask_accuracy"] = 1.0
summ.to_csv(os.path.join(OUTPUT, "summary_metrics.csv"), index=False)

# LaTeX table
pretty = {"oracle": "Oracle", "model": "Edge-AI (MLP)", "all_local": "All-Local",
          "all_mcmf": "All-MCMF", "random": "Random"}
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
    acc = "--" if np.isnan(r.subtask_accuracy) else f"{r.subtask_accuracy:.3f}"
    lines.append(
        f"{pretty[r.method]} & {r.latency_mean:.3f} $\\pm$ {r.latency_std:.3f} & "
        f"{r.energy_mean:.3f} $\\pm$ {r.energy_std:.3f} & "
        f"{r.cost_mean:.3f} $\\pm$ {r.cost_std:.3f} & "
        f"{r.deadline_viol_rate:.3f} & {acc} \\\\")
lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
with open(os.path.join(OUTPUT, "summary_table.tex"), "w") as f:
    f.write("\n".join(lines))

# ---- Plots ----
COLORS = {"oracle": "#2ca02c", "model": "#1f77b4", "all_local": "#d62728",
          "all_mcmf": "#ff7f0e", "random": "#9467bd"}

# latency CDF
plt.figure(figsize=(7, 5), dpi=150)
for m in ["oracle", "model", "all_local", "all_mcmf"]:
    s = np.sort(res[res.method_name == m]["latency"].values)
    cdf = np.arange(1, len(s) + 1) / len(s)
    plt.plot(s, cdf, label=pretty[m], color=COLORS[m], lw=2)
plt.xlabel("Per-user latency (s)"); plt.ylabel("CDF")
plt.title("User Latency CDF"); plt.legend(); plt.grid(alpha=0.3)
plt.xlim(left=0)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "latency_cdf.png")); plt.close()

# energy bar
plt.figure(figsize=(7, 5), dpi=150)
ms = methods
means = [res[res.method_name == m]["energy"].mean() for m in ms]
stds = [res[res.method_name == m]["energy"].std() for m in ms]
plt.bar([pretty[m] for m in ms], means, yerr=stds, capsize=4,
        color=[COLORS[m] for m in ms])
plt.ylabel("Avg per-user energy (J)"); plt.title("Average Energy per User")
plt.xticks(rotation=20); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "energy_bar.png")); plt.close()

# deadline bar
plt.figure(figsize=(7, 5), dpi=150)
dv = [res[res.method_name == m]["deadline_violation"].mean() for m in ms]
plt.bar([pretty[m] for m in ms], dv, color=[COLORS[m] for m in ms])
plt.ylabel("Deadline violation rate"); plt.title("Deadline Violation Rate")
plt.xticks(rotation=20); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "deadline_bar.png")); plt.close()

# uav vs ground latency (model), only offloaded-anchor users
plt.figure(figsize=(7, 5), dpi=150)
mdl = res[(res.method_name == "model") & (res.has_anchor == 1)]
for uav_flag, lab, col in [(1, "UAV anchor", "#1f77b4"), (0, "Ground anchor", "#ff7f0e")]:
    s = np.sort(mdl[mdl.anchor_is_uav == uav_flag]["latency"].values)
    if len(s):
        cdf = np.arange(1, len(s) + 1) / len(s)
        plt.plot(s, cdf, label=f"{lab} (n={len(s)})", color=col, lw=2)
plt.xlabel("Per-user latency (s)"); plt.ylabel("CDF")
plt.title("Edge-AI: UAV vs Ground Anchor Latency"); plt.legend(); plt.grid(alpha=0.3)
plt.xlim(left=0)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "uav_vs_ground_latency.png")); plt.close()

# ======================================================================
# G. REPORT
# ======================================================================
print("\n" + "=" * 70)
print("FINAL TEST-SET REPORT")
print("=" * 70)
print(f"Model subtask accuracy: {subtask_acc:.4f}")
print(f"Deadline = {DEADLINE:.4f} s (1.5x median oracle latency)\n")
print(summ.to_string(index=False,
      columns=["method", "latency_mean", "energy_mean", "cost_mean",
               "deadline_viol_rate", "subtask_accuracy"]))

print("\nUAV vs Ground (model, offloaded-anchor users):")
for uav_flag, lab in [(1, "UAV"), (0, "Ground")]:
    s = mdl[mdl.anchor_is_uav == uav_flag]
    if len(s):
        print(f"  {lab:6s} n={len(s):4d}  lat={s.latency.mean():.3f}  "
              f"en={s.energy.mean():.3f}  cost={s.cost.mean():.3f}")

print("\nSaved:")
for p in [MODEL_PATH,
          os.path.join(OUTPUT, "results_metrics.csv"),
          os.path.join(OUTPUT, "summary_metrics.csv"),
          os.path.join(OUTPUT, "summary_table.tex"),
          os.path.join(OUTPUT, "training_history.csv"),
          os.path.join(OUTPUT, "latency_cdf.png"),
          os.path.join(OUTPUT, "energy_bar.png"),
          os.path.join(OUTPUT, "deadline_bar.png"),
          os.path.join(OUTPUT, "uav_vs_ground_latency.png")]:
    print(f"  {p}")
