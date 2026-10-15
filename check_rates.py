"""
check_rates.py - report the uplink rates used for the oracle labels and costs.

Reads data/subtask/all_subtask.parquet and recomputes, for every subtask row,
the rate used by build_subtask_dataset.py (function uplink_rate):

    R = B * log2(1 + x),   x = P_tx * h**2 / (N0 + P_tx * h**2)

with h = best_channel_gain. Prints the number of rows, the range of h, the range
of R in Mbit/s and the share of rows whose rate is below 99.9 % of B.
No file is written. See README section 5.3a.
"""

import os

import numpy as np
import pandas as pd

from build_subtask_dataset import uplink_rate

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "data", "subtask", "all_subtask.parquet")


def main():
    df = pd.read_parquet(PATH, columns=["best_channel_gain", "B", "N0", "P_tx"])
    h = df["best_channel_gain"].to_numpy(dtype=float)
    B = df["B"].to_numpy(dtype=float)
    N0 = df["N0"].to_numpy(dtype=float)
    P_tx = df["P_tx"].to_numpy(dtype=float)
    R = uplink_rate(B, P_tx, h, N0)

    print(f"rows: {len(df)}")
    print(f"B (Hz): {np.unique(B)}   N0 (W): {np.unique(N0)}   P_tx (W): {np.unique(P_tx)}")
    print(f"best_channel_gain h: min {h.min():.6g}  median {np.median(h):.6g}  max {h.max():.6g}")
    print(f"rate R (Mbit/s): min {R.min() / 1e6:.6f}  median {np.median(R) / 1e6:.6f}  max {R.max() / 1e6:.6f}")
    print(f"share of rows with R < 99.9 % of B: {np.mean(R < 0.999 * B):.6f}")


if __name__ == "__main__":
    main()
