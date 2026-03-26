"""
Build COMPUSTAT-based per-share EPS panel matching the paper's earnings.do.

eps = ib / ((shrout/1000) * cfacshr)

where:
  ib      = income before extraordinary items (COMPUSTAT funda)
  shrout  = shares outstanding in thousands (CRSP msf)
  cfacshr = cumulative split adjustment factor (CRSP msf)

Output: BGLS2019_rep/replication/output/compustat_eps_panel.parquet
  Columns: PERMNO, yr, eps (COMPUSTAT-based per-share EPS)
"""

import numpy as np
import pandas as pd

# ── 1. Load COMPUSTAT annual ib ───────────────────────────────────────────
comp = pd.read_parquet("data/comp_annual.parquet",
                       columns=["gvkey", "fyear", "ib"])
comp = comp.dropna(subset=["ib"])
comp = comp.rename(columns={"fyear": "yr"})

# ── 2. CCM link: gvkey → permno ──────────────────────────────────────────
ccm = pd.read_parquet("data/ccm_link.parquet")
ccm = ccm[ccm["linktype"].isin(["LC", "LU"])].copy()
ccm = ccm[ccm["linkprim"].isin(["P", "C"])].copy()
ccm["permno"] = ccm["permno"].astype(int)

# Prefer LC+P links; deduplicate gvkey
ccm["priority"] = ccm["linktype"].map({"LC": 0, "LU": 1})
ccm = ccm.sort_values("priority").drop_duplicates(subset=["gvkey"], keep="first")
ccm = ccm[["gvkey", "permno"]].copy()

comp = comp.merge(ccm, on="gvkey", how="inner")
comp = comp.rename(columns={"permno": "PERMNO"})

# ── 3. CRSP December shrout and cfacshr ──────────────────────────────────
crsp = pd.read_parquet("data/crsp_monthly.parquet",
                       columns=["permno", "date", "shrout"])
crsp["date"] = pd.to_datetime(crsp["date"])
crsp["yr"] = crsp["date"].dt.year
crsp["month"] = crsp["date"].dt.month

cfac = pd.read_parquet("data/crsp_cfacshr.parquet")
cfac["date"] = pd.to_datetime(cfac["date"])
cfac["yr"] = cfac["date"].dt.year
cfac["month"] = cfac["date"].dt.month

# Merge shrout + cfacshr
crsp = crsp.merge(cfac[["permno", "yr", "month", "cfacshr"]],
                  on=["permno", "yr", "month"], how="left")

# Take December observation (or latest available month in the year)
crsp = crsp.sort_values(["permno", "yr", "month"])
crsp_dec = crsp[crsp["month"] == 12].drop_duplicates(subset=["permno", "yr"], keep="last")

# For years without December, take the latest month
crsp_other = crsp.drop_duplicates(subset=["permno", "yr"], keep="last")
crsp_dec = pd.concat([crsp_dec,
                      crsp_other[~crsp_other.set_index(["permno", "yr"]).index
                                  .isin(crsp_dec.set_index(["permno", "yr"]).index)]])

crsp_dec = crsp_dec[["permno", "yr", "shrout", "cfacshr"]].copy()
crsp_dec = crsp_dec.rename(columns={"permno": "PERMNO"})
crsp_dec = crsp_dec.dropna(subset=["shrout", "cfacshr"])

# ── 4. Merge and compute eps ─────────────────────────────────────────────
df = comp.merge(crsp_dec, on=["PERMNO", "yr"], how="inner")

# shrout is in thousands in CRSP; ib is in millions in COMPUSTAT
# shsplit = (shrout/1000) * cfacshr  [shrout/1000 converts to millions]
# Filter extreme cfacshr values that produce nonsensical EPS
df = df[df["cfacshr"] >= 0.001].copy()
df["shsplit"] = (df["shrout"] / 1000.0) * df["cfacshr"]
# Require at least 0.1M shares (shsplit >= 0.1) to avoid extreme per-share values
df["eps"] = np.where(df["shsplit"] >= 0.1, df["ib"] / df["shsplit"], np.nan)
df = df.dropna(subset=["eps"])

# Keep one per PERMNO-yr (prefer the one with larger |ib| if duplicates)
df["abs_ib"] = df["ib"].abs()
df = df.sort_values(["PERMNO", "yr", "abs_ib"], ascending=[True, True, False])
df = df.drop_duplicates(subset=["PERMNO", "yr"], keep="first")

# Cap extreme EPS values — most real per-share EPS is well under $500
# (outliers arise from tiny share counts or extreme cfacshr)
df["eps"] = df["eps"].clip(-500, 500)

result = df[["PERMNO", "yr", "eps"]].copy()
print(f"COMPUSTAT EPS panel: {len(result):,} obs, "
      f"{result['PERMNO'].nunique():,} firms, "
      f"years {result['yr'].min()}-{result['yr'].max()}")

result.to_parquet("BGLS2019_rep/replication/output/compustat_eps_panel.parquet", index=False)
print("Saved BGLS2019_rep/replication/output/compustat_eps_panel.parquet")
