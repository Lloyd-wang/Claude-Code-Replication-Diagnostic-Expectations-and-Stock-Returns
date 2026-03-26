"""
Replicate Figure 3 from BGLS (2019): Evolution of LTG.

Reads descriptive.parquet and produces Figure3.png showing the evolution
of realized earnings growth around portfolio formation for high-LTG (decile 10)
and low-LTG (decile 1) cohorts, with bootstrap confidence intervals.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_parquet("replication/output/descriptive.parquet")

# ── Step 1: require GROW_F0 and GROW_L3 both non-missing, yr <= 2015 ──
df = df.loc[df["GROW_F0"].notna() & df["GROW_L3"].notna() & (df["yr"] <= 2015)].copy()

# ── Step 2: keep only yr <= max yr where GROW_F3 is non-missing ────────
max_yr_f3 = df.loc[df["GROW_F3"].notna(), "yr"].max()
df = df.loc[df["yr"] <= max_yr_f3].copy()

# ── Step 3: filter to 3-year cohorts starting 1981 ────────────────────
cohort_years = list(range(1981, 2016, 3))  # 1981, 1984, ..., 2014
df = df.loc[df["yr"].isin(cohort_years)].copy()

# ── Step 4: rename GROW columns to GROW1–GROW8 ────────────────────────
rename_map = {
    "GROW_L3": "GROW1",  # t = -3
    "GROW_L2": "GROW2",  # t = -2
    "GROW_L1": "GROW3",  # t = -1
    "GROW_F0": "GROW4",  # t =  0
    "GROW_F1": "GROW5",  # t = +1
    "GROW_F2": "GROW6",  # t = +2
    "GROW_F3": "GROW7",  # t = +3
    "GROW_F4": "GROW8",  # t = +4
}
df = df.rename(columns=rename_map)

# ── Step 5: bootstrap means for HLTG (LTG==10) and LLTG (LTG==1) ─────
grow_cols = [f"GROW{i}" for i in range(1, 8)]  # GROW1–GROW7 (t=-3 to +3)
rng = np.random.RandomState(1234)
n_boot = 1000

results = {}
for label, ltg_val in [("HLTG", 10), ("LLTG", 1)]:
    sub = df.loc[df["LTG"] == ltg_val].copy()
    means = []
    ci_lo = []
    ci_hi = []
    for col in grow_cols:
        vals = sub[col].dropna().values
        boot_means = np.array(
            [vals[rng.randint(0, len(vals), size=len(vals))].mean() for _ in range(n_boot)]
        )
        means.append(boot_means.mean())
        ci_lo.append(np.percentile(boot_means, 2.5))
        ci_hi.append(np.percentile(boot_means, 97.5))
    results[label] = {"mean": means, "ci_lo": ci_lo, "ci_hi": ci_hi}

# ── Step 6: plot ───────────────────────────────────────────────────────
x = list(range(-3, 4))  # t = -3, -2, -1, 0, +1, +2, +3

fig, ax = plt.subplots(figsize=(8, 5))

# HLTG (red)
ax.plot(x, results["HLTG"]["mean"], color="red", linewidth=1.5, label="HLTG (decile 10)")
ax.plot(x, results["HLTG"]["ci_lo"], color="red", linewidth=0.7, linestyle="dotted")
ax.plot(x, results["HLTG"]["ci_hi"], color="red", linewidth=0.7, linestyle="dotted")

# LLTG (blue)
ax.plot(x, results["LLTG"]["mean"], color="blue", linewidth=1.5, label="LLTG (decile 1)")
ax.plot(x, results["LLTG"]["ci_lo"], color="blue", linewidth=0.7, linestyle="dotted")
ax.plot(x, results["LLTG"]["ci_hi"], color="blue", linewidth=0.7, linestyle="dotted")

ax.set_xlabel("Years relative to formation")
ax.set_ylabel("Earnings growth (%)")
ax.set_title("Figure 3: Evolution of LTG")
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("replication/output/Figure3.png", dpi=150)
plt.close(fig)

print("Saved replication/output/Figure3.png")
