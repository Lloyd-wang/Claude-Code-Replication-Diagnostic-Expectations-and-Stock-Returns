"""
Figure 4: Forecast Errors of LTG in Predicting Future EPS Growth (BGLS 2019)

Replicates JFsubmission.do lines 307-391.
Builds ib_* from fresh COMPUSTAT annual data following earnings.do:
  ib_per_share = ib / ((shrout/1000) * cfacshr)
then creates leads/lags by fyear, merges with descriptive.parquet by (PERMNO, yr).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = '/workspaces/Dividend-growth/data'
OUT_DIR  = 'BGLS2019_rep/replication/output'

# ============================================================================
# Step 1: Build ib_* from COMPUSTAT annual (following earnings.do)
# ============================================================================

# 1a. Load COMPUSTAT annual
comp = pd.read_parquet(f'{DATA_DIR}/comp_annual.parquet',
                       columns=['gvkey', 'datadate', 'fyear', 'fyr', 'ib'])
comp = comp.dropna(subset=['ib', 'fyear'])
comp['datadate'] = pd.to_datetime(comp['datadate'])
comp['fyear'] = comp['fyear'].astype(int)

# 1b. CCM link (LC/LU, primary)
ccm = pd.read_parquet(f'{DATA_DIR}/ccm_link.parquet')
ccm = ccm[ccm['linktype'].isin(['LU', 'LC']) & ccm['linkprim'].isin(['P', 'C'])].copy()
ccm['permno'] = ccm['permno'].astype(int)
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt']).fillna(pd.Timestamp('2099-12-31'))

merged = comp.merge(ccm[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='gvkey', how='inner')
merged = merged[(merged['datadate'] >= merged['linkdt']) &
                (merged['datadate'] <= merged['linkenddt'])].copy()
merged = merged.drop_duplicates(subset=['gvkey', 'datadate'], keep='first')

# 1c. CRSP monthly for shrout, cfacshr (match at datadate month)
crsp = pd.read_parquet(f'{DATA_DIR}/crsp_monthly.parquet',
                       columns=['permno', 'date', 'shrout', 'cfacshr', 'shrcd', 'exchcd'])
crsp['date'] = pd.to_datetime(crsp['date'])
for c in ['permno', 'shrout', 'cfacshr', 'shrcd', 'exchcd']:
    crsp[c] = pd.to_numeric(crsp[c], errors='coerce')
crsp['ym'] = crsp['date'].dt.to_period('M')

merged['ym'] = merged['datadate'].dt.to_period('M')
merged = merged.merge(crsp[['permno', 'ym', 'shrout', 'cfacshr', 'shrcd', 'exchcd']],
                      on=['permno', 'ym'], how='left')
merged = merged.dropna(subset=['shrout', 'cfacshr'])
merged = merged[(merged['shrcd'].isin([10, 11])) & (merged['exchcd'].between(1, 3))].copy()

# 1d. Compute per-share ib (earnings.do: split=(shrout/1000)*cfacshr, ib=ib/split)
merged['split'] = (merged['shrout'] / 1000) * merged['cfacshr']
merged = merged[merged['split'] > 0].copy()
merged['ib_ps'] = merged['ib'] / merged['split']
merged = merged.sort_values(['permno', 'fyear', 'datadate'])
merged = merged.drop_duplicates(subset=['permno', 'fyear'], keep='first')

# 1e. Create leads/lags by fyear (earnings.do: sort permco permno fpedats, by: gen ib_F0=F0.ib)
panel = merged[['permno', 'fyear', 'fyr', 'ib_ps']].copy()
for offset, name in [(-1, 'ib_L1'), (0, 'ib_F0'), (1, 'ib_F1'), (2, 'ib_F2'), (3, 'ib_F3')]:
    tmp = panel[['permno', 'fyear', 'ib_ps']].copy()
    tmp['fyear'] = tmp['fyear'] - offset
    tmp = tmp.rename(columns={'ib_ps': name})
    panel = panel.merge(tmp[['permno', 'fyear', name]], on=['permno', 'fyear'], how='left')

# 1f. Map fyear → yr (yr = fyear for IBES December alignment)
panel['yr'] = panel['fyear']

print(f"COMPUSTAT ib panel: {len(panel):,} firm-years")

# ============================================================================
# Step 2: Merge with descriptive.parquet (for LTG decile and GROW_F0)
# ============================================================================
desc = pd.read_parquet(f'{OUT_DIR}/descriptive.parquet',
                       columns=['PERMNO', 'yr', 'LTG', 'GROW_F0'])
desc = desc[desc['yr'] <= 2015].copy()

df = desc.merge(panel[['permno', 'yr', 'ib_L1', 'ib_F0', 'ib_F1', 'ib_F2', 'ib_F3']],
                left_on=['PERMNO', 'yr'], right_on=['permno', 'yr'], how='inner')

# Ensure float64 to avoid NA boolean issues
for c in ['ib_L1', 'ib_F0', 'ib_F1', 'ib_F2', 'ib_F3', 'GROW_F0']:
    df[c] = df[c].astype(float)

print(f"Merged with descriptive: {len(df):,} obs (of {len(desc):,})")

# ============================================================================
# Step 3: Winsorize ib_* at 1/99 by year (descriptive.do line 255)
# ============================================================================
ib_cols = ['ib_L1', 'ib_F0', 'ib_F1', 'ib_F2', 'ib_F3']
for col in ib_cols:
    p01 = df.groupby('yr')[col].transform(lambda x: x.quantile(0.01))
    p99 = df.groupby('yr')[col].transform(lambda x: x.quantile(0.99))
    df[col] = df[col].clip(lower=p01, upper=p99)

# ============================================================================
# Step 4: Compute forecast errors (JFsubmission.do lines 312-319)
# geps_k = ib_F(k)/ib_F(k-1) - 1 - GROW_F0/100, with per-variable positive denom
# ============================================================================
df["geps0"] = np.where(df["ib_L1"] > 0,
                       (df["ib_F0"] / df["ib_L1"]) - 1 - df["GROW_F0"] / 100, np.nan)
df["geps1"] = np.where(df["ib_F0"] > 0,
                       (df["ib_F1"] / df["ib_F0"]) - 1 - df["GROW_F0"] / 100, np.nan)
df["geps2"] = np.where(df["ib_F1"] > 0,
                       (df["ib_F2"] / df["ib_F1"]) - 1 - df["GROW_F0"] / 100, np.nan)
df["geps3"] = np.where(df["ib_F2"] > 0,
                       (df["ib_F3"] / df["ib_F2"]) - 1 - df["GROW_F0"] / 100, np.nan)

# ============================================================================
# Step 5: Winsorize geps at 1/99 by year (JFsubmission.do lines 322-328)
# ============================================================================
for col in ["geps0", "geps1", "geps2", "geps3"]:
    p01 = df.groupby("yr")[col].transform(lambda x: x.quantile(0.01))
    p99 = df.groupby("yr")[col].transform(lambda x: x.quantile(0.99))
    df[col] = df[col].clip(lower=p01, upper=p99)

# ============================================================================
# Step 6: 3-year non-overlapping cohorts (JFsubmission.do lines 330-332)
# ============================================================================
cohort_years = list(range(1981, 2016, 3))  # 1981, 1984, ..., 2014
max_yr_geps3 = df.loc[df["geps3"].notna(), "yr"].max()
df = df[df["yr"] <= max_yr_geps3].copy()
df = df[df["yr"].isin(cohort_years)].copy()

# ============================================================================
# Step 7: Bootstrap means and confidence intervals
# ============================================================================
rng = np.random.RandomState(1234)
n_boot = 1000

results = {}
for label, ltg_val in [("HLTG", 10), ("LLTG", 1)]:
    sub = df[df["LTG"] == ltg_val]
    means, ci_lo, ci_hi = [], [], []
    for k in range(4):
        col = f"geps{k}"
        vals = sub[col].dropna().values
        boot_means = np.array([
            vals[rng.randint(0, len(vals), size=len(vals))].mean()
            for _ in range(n_boot)
        ])
        means.append(boot_means.mean())
        ci_lo.append(np.percentile(boot_means, 2.5))
        ci_hi.append(np.percentile(boot_means, 97.5))
    results[label] = {"mean": means, "ci_lo": ci_lo, "ci_hi": ci_hi}

# ============================================================================
# Step 8: Plot
# ============================================================================
x = np.arange(4)
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(x, results["HLTG"]["mean"], color="red", marker="o", label="HLTG (LTG=10)")
ax.plot(x, results["HLTG"]["ci_lo"], color="red", linestyle="dotted", linewidth=0.8)
ax.plot(x, results["HLTG"]["ci_hi"], color="red", linestyle="dotted", linewidth=0.8)

ax.plot(x, results["LLTG"]["mean"], color="blue", marker="o", label="LLTG (LTG=1)")
ax.plot(x, results["LLTG"]["ci_lo"], color="blue", linestyle="dotted", linewidth=0.8)
ax.plot(x, results["LLTG"]["ci_hi"], color="blue", linestyle="dotted", linewidth=0.8)

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Years relative to formation")
ax.set_ylabel("EPS(t)/EPS(t-1)-LTG(0)")
ax.set_xticks(x)
ax.legend()
ax.set_title("Figure 4: Forecast Errors of LTG in Predicting Future EPS Growth")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Figure4.png", dpi=150)
plt.close()

print(f"\nSaved {OUT_DIR}/Figure4.png")
for label in ["HLTG", "LLTG"]:
    print(f"\n{label}:")
    for k in range(4):
        m = results[label]["mean"][k]
        lo = results[label]["ci_lo"][k]
        hi = results[label]["ci_hi"][k]
        print(f"  k={k}: mean={m:.4f}  CI=[{lo:.4f}, {hi:.4f}]")
