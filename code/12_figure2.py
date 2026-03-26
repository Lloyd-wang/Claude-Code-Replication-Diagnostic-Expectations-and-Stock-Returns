"""
Replication of Figure 2 from BGLS (2019): Evolution of EPS.
Follows JFsubmission.do lines 123-229.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_parquet("replication/output/descriptive.parquet")

# Step 1: Filter to 3-year cohorts
cohort_years = list(range(1981, 2015, 3))  # 1981,1984,...,2014 (matching Stata 1981(3)2015)
df = df[df["yr"].isin(cohort_years)].copy()

# Step 2: Require eps_L3, eps_L2, eps_L1, eps_F0 all non-missing
df = df.dropna(subset=["eps_L3", "eps_L2", "eps_L1", "eps_F0"])

# Step 3: Compute EPS ratios normalised to t=-3 (only when eps_L3 > 0)
mask_pos = df["eps_L3"] > 0

df["eps_1"] = np.where(mask_pos, 1.0, np.nan)
df["eps_2"] = np.where(mask_pos, df["eps_L2"] / df["eps_L3"], np.nan)
df["eps_3"] = np.where(mask_pos, df["eps_L1"] / df["eps_L3"], np.nan)
df["eps_4"] = np.where(mask_pos, df["eps_F0"] / df["eps_L3"], np.nan)
df["eps_5"] = np.where(mask_pos, df["eps_F1"] / df["eps_L3"], np.nan)
df["eps_6"] = np.where(mask_pos, df["eps_F2"] / df["eps_L3"], np.nan)
df["eps_7"] = np.where(mask_pos, df["eps_F3"] / df["eps_L3"], np.nan)
df["eps_8"] = np.where(mask_pos, df["eps_F4"] / df["eps_L3"], np.nan)

# Restrict to years where eps_F4 is available (Stata: su yr if eps_8<., keep if yr<=r(max))
max_yr_eps8 = df.loc[df["eps_F4"].notna() & (df["eps_L3"] > 0), "yr"].max()
if max_yr_eps8 is not None:
    df = df[df["yr"] <= max_yr_eps8].copy()

# Step 4: Winsorize eps ratios at 5/95 by year (Stata: bysort yr: pctile, p(05)/p(95))
eps_ratio_cols = [f"eps_{k}" for k in range(1, 9)]  # eps_1..eps_8
for col in eps_ratio_cols:
    lo = df.groupby("yr")[col].transform(lambda x: x.quantile(0.05))
    hi = df.groupby("yr")[col].transform(lambda x: x.quantile(0.95))
    df[col] = df[col].clip(lower=lo, upper=hi)

# ── Bootstrap function ─────────────────────────────────────────────────────
def bootstrap_mean_ci(series, n_boot=200, seed=1234, alpha=0.05):
    """Return (mean, ci_low, ci_high) via bootstrap percentile method."""
    vals = series.dropna().values
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    n = len(vals)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = vals[idx].mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return vals.mean(), lo, hi


# ── Step 5: Bootstrap means for HLTG and LLTG at each time point ──────────
eps_cols = [f"eps_{k}" for k in range(1, 8)]  # eps_1..eps_7  (t=-3..+3)
time_labels = list(range(-3, 4))  # -3,-2,-1,0,1,2,3

results = {}
for label, (ltg_val, ltg_name) in enumerate([(1, "LLTG"), (10, "HLTG")]):
    sub = df[df["LTG"] == ltg_val]
    means, ci_lo, ci_hi = [], [], []
    for col in eps_cols:
        m, lo, hi = bootstrap_mean_ci(sub[col], n_boot=200, seed=1234)
        means.append(m)
        ci_lo.append(lo)
        ci_hi.append(hi)
    # Step 7: At t=-3 (index 0), set CI bounds to 1.0
    ci_lo[0] = 1.0
    ci_hi[0] = 1.0
    results[ltg_name] = {"mean": means, "ci_lo": ci_lo, "ci_hi": ci_hi}

# ── Step 6: Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

x = time_labels

# LLTG – blue
ax.plot(x, results["LLTG"]["mean"], color="blue", linewidth=1.5, label="LLTG")
ax.plot(x, results["LLTG"]["ci_lo"], color="blue", linewidth=0.8, linestyle=":")
ax.plot(x, results["LLTG"]["ci_hi"], color="blue", linewidth=0.8, linestyle=":")

# HLTG – red
ax.plot(x, results["HLTG"]["mean"], color="red", linewidth=1.5, label="HLTG")
ax.plot(x, results["HLTG"]["ci_lo"], color="red", linewidth=0.8, linestyle=":")
ax.plot(x, results["HLTG"]["ci_hi"], color="red", linewidth=0.8, linestyle=":")

ax.set_xlabel("Years relative to formation")
ax.set_ylabel("EPS (normalised to t = -3)")
ax.set_title("Figure 2: Evolution of EPS")
ax.legend()
ax.set_xticks(time_labels)

fig.tight_layout()
fig.savefig("replication/output/Figure2.png", dpi=150)
plt.close(fig)

print("Saved replication/output/Figure2.png")
