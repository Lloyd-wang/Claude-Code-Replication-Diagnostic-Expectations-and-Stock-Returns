"""
Produce Figures 7 and 8 from BGLS (2019).

Figure 7: Kernel density of 5-year realized GROSS EPS growth for HLTG vs Not-HLTG.
          Following Kernels_JF.do: s = (eps_F5/eps_F0)^(1/5), cap at 2, centered ~1.
          Vertical lines at means of each distribution.
Figure 8: For HLTG only, two distributions:
          (1) Realized gross EPS growth = same HLTG density from Figure 7
          (2) Expected LTG = 1 + GROW_F0/100
          Vertical lines at means (realized ~1.11 vs expected ~1.39).

Methodology follows Kernels_JF.do from the original BGLS replication code.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Load data ────────────────────────────────────────────────────────────────
# Stata Kernels_JF.do: "drop eps; rename ib_* eps_*" → uses COMPUSTAT ib per share,
# NOT the original eps_* from descriptive.dta.
desc = pd.read_parquet("replication/output/descriptive.parquet",
                       columns=["PERMNO", "yr", "LTG", "GROW_F0"])

# Build ib_F0 and ib_F5 from compustat_eps_panel (COMPUSTAT ib per share)
ib = pd.read_parquet("replication/output/compustat_eps_panel.parquet")
ib_f0 = ib.rename(columns={"eps": "ib_F0"})
ib_f5 = ib.copy()
ib_f5["yr"] = ib_f5["yr"] - 5  # shift: eps at yr+5 becomes ib_F5 at yr
ib_f5 = ib_f5.rename(columns={"eps": "ib_F5"})

df = desc.merge(ib_f0[["PERMNO", "yr", "ib_F0"]], on=["PERMNO", "yr"], how="left")
df = df.merge(ib_f5[["PERMNO", "yr", "ib_F5"]], on=["PERMNO", "yr"], how="left")
print(f"Merged ib_F0: {df['ib_F0'].notna().sum():,}, ib_F5: {df['ib_F5'].notna().sum():,} / {len(df):,}")

# ── Compute GROSS growth rate: s = (ib_F5/ib_F0)^(1/5) ─────────────────
# Stata: gen eps_g=(eps_F5/eps_F0)^(1/5) if eps_F0>0  (after renaming ib→eps)
mask_eps = df["ib_F0"] > 0
df.loc[mask_eps, "eps_g"] = (df.loc[mask_eps, "ib_F5"] / df.loc[mask_eps, "ib_F0"]) ** (1 / 5)

# ── Winsorize at 1/99 percentiles (pooled, matching Stata) ────────────────
# Stata: egen min=pctile(eps_g), p(01); egen max=pctile(eps_g), p(99)
for var in ["eps_g"]:
    p01 = df[var].quantile(0.01)
    p99 = df[var].quantile(0.99)
    df.loc[df[var] < p01, var] = p01
    df.loc[(df[var] > p99) & df[var].notna(), var] = p99

# ── Keep 5-year cohorts ──────────────────────────────────────────────────────
# Stata: forvalues x=1981(5)2015 { replace keep=yr if yr==`x' }
# Then: su yr if rmc_g<., keep if yr<=r(max), keep if keep<.
cohorts = list(range(1981, 2016, 5))  # 1981, 1986, ..., 2011, (2016 excluded)
max_yr = df.loc[df["eps_g"].notna(), "yr"].max()
df = df[df["yr"] <= max_yr].copy()
df = df[df["yr"].isin(cohorts)].copy()

# ── Cap at 2 (Stata: replace s=2 if s>2 & s<.) ──────────────────────────────
# s is the GROSS growth rate, NOT net
df["s"] = df["eps_g"].copy()
df.loc[(df["s"] > 2.0) & df["s"].notna(), "s"] = 2.0

# ── Expected LTG as gross growth rate ─────────────────────────────────────────
# Stata: generate l=1+GROW_F0/100
df["ltg_gross"] = 1 + df["GROW_F0"] / 100.0

# ── Define groups ────────────────────────────────────────────────────────────
hltg = df["LTG"] == 10
noth = (df["LTG"] >= 1) & (df["LTG"] <= 9)

bw = 0.15

def epanechnikov_kde(data, bw, grid):
    """Epanechnikov KDE matching Stata's kdensity exactly.
    K(u) = 3/4 * (1 - u^2) for |u| <= 1, f(x) = 1/(n*h) * sum(K((x-xi)/h))
    """
    data = np.asarray(data, dtype=float).reshape(-1, 1)  # (n, 1)
    grid = np.asarray(grid, dtype=float).reshape(1, -1)  # (1, m)
    u = (grid - data) / bw  # (n, m)
    k = np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0)
    return k.sum(axis=0) / (len(data) * bw)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7: HLTG vs Not-HLTG realized gross EPS growth
# Stata: kdensity s if LTG==10, bw(.15) ...
#        kdensity s if LTG>=01 & LTG<=09, bw(.15) ...
# ══════════════════════════════════════════════════════════════════════════════
s_hltg = df.loc[hltg, "s"].dropna()
s_noth = df.loc[noth, "s"].dropna()

# Stata: n(50) grid from data range, then filters to [0.5, 2.0]
# Use 200 points for visually smooth curves
x_grid = np.linspace(0.5, 2.0, 200)

den_hltg = epanechnikov_kde(s_hltg, bw, x_grid)
den_noth = epanechnikov_kde(s_noth, bw, x_grid)

m_H = s_hltg.mean()
m_N = s_noth.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_grid, den_hltg, color="blue", linewidth=1.5, label="HLTG")
ax.plot(x_grid, den_noth, color="orange", linewidth=1.5, label="Not HLTG")
ax.axvline(m_H, color="blue", linewidth=0.5)
ax.axvline(m_N, color="orange", linewidth=0.5)
ax.set_xlabel("Growth in EPS", fontsize=11)
ax.set_ylabel("Density Function", fontsize=11)
ax.legend(fontsize=9)
ax.set_title("Figure 7: Kernel Density of 5-Year Realized EPS Growth", fontsize=12)
fig.tight_layout()
fig.savefig("replication/output/Figure7.png", dpi=150)
plt.close(fig)
print(f"Saved Figure7.png (HLTG mean={m_H:.2f}, Not-HLTG mean={m_N:.2f})")
print(f"  HLTG peak density: {den_hltg.max():.2f}, Not-HLTG peak density: {den_noth.max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 8: For HLTG only - realized growth vs expected LTG
# Stata: kdensity s if LTG==10 → HLTG (already computed above)
#        generate l=1+GROW_F0/100; kdensity l if LTG==10 → IBESHi
# ══════════════════════════════════════════════════════════════════════════════
ltg_hltg = df.loc[hltg, "ltg_gross"].dropna()

# Stata also caps l at 2: replace l=2 if l>2 & l<.
ltg_hltg = ltg_hltg.clip(upper=2.0)

m_G = s_hltg.mean()  # mean realized growth for HLTG
m_L = ltg_hltg.mean()  # mean expected LTG for HLTG

# Stata: Figure 8 uses abs(Z)<2 → grid from ~0.5 to 2.0
x_grid2 = np.linspace(0.5, 2.0, 200)

den_growth = epanechnikov_kde(s_hltg, bw, x_grid2)
den_ltg = epanechnikov_kde(ltg_hltg, bw, x_grid2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_grid2, den_growth, color="blue", linewidth=1.5, label="Growth EPS")
ax.plot(x_grid2, den_ltg, color="red", linewidth=1.5, label="LTG")
ax.axvline(m_G, color="blue", linewidth=0.5)
ax.axvline(m_L, color="red", linewidth=0.5)
ax.set_xlabel("Growth in EPS, LTG", fontsize=11)
ax.set_ylabel("Density Function", fontsize=11)
ax.legend(fontsize=9)
ax.set_title("Figure 8: Realized vs Expected Growth (HLTG)", fontsize=12)
fig.tight_layout()
fig.savefig("replication/output/Figure8.png", dpi=150)
plt.close(fig)
print(f"Saved Figure8.png (Realized mean={m_G:.2f}, Expected LTG mean={m_L:.2f})")
print(f"  Growth EPS peak: {den_growth.max():.2f}, LTG peak: {den_ltg.max():.2f}")
