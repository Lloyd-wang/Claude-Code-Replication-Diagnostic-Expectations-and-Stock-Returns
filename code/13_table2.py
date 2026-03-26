"""
Table II from BGLS (2019): Coibion-Gorodnichenko regressions.

Regress forecast errors (y_h) on forecast revisions (delta_k) with year FE,
clustered by year. 3x3 table: rows = revision horizon k=1,2,3;
columns = forecast-error horizon 3yr, 4yr, 5yr.

This version follows the paper's methodology: LTG forecasts made 30-90 days
after each annual earnings announcement (revisions03.do lines 120-122),
rather than December IBES consensus.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── 1. Build annual actual EPS from COMPUSTAT ib / CRSP shares ───────────────
print("Building annual EPS from COMPUSTAT ib and CRSP shares...")

# 1a. Load COMPUSTAT annual ib
comp = pd.read_parquet("data/comp_annual.parquet", columns=["gvkey", "fyear", "ib"])
comp = comp.dropna(subset=["ib"])
print(f"  COMPUSTAT ib obs: {len(comp):,}")

# 1b. Link gvkey → permno via CCM
ccm = pd.read_parquet("data/ccm_link.parquet")
ccm = ccm.loc[ccm["linktype"].isin(["LC", "LU"]) & ccm["linkprim"].isin(["P", "C"])].copy()
ccm["permno"] = ccm["permno"].astype(int)
# Keep best link per gvkey: prefer LC over LU, P over C
ccm["_lt"] = ccm["linktype"].map({"LC": 0, "LU": 1})
ccm["_lp"] = ccm["linkprim"].map({"P": 0, "C": 1})
ccm = ccm.sort_values(["gvkey", "_lt", "_lp"]).drop_duplicates(subset=["gvkey"], keep="first")
ccm = ccm[["gvkey", "permno"]].copy()

comp = comp.merge(ccm, on="gvkey", how="inner")
print(f"  After CCM link: {len(comp):,}")

# 1c. Link permno → TICKER via descriptive.parquet
desc_map = pd.read_parquet("BGLS2019_rep/replication/output/descriptive.parquet", columns=["TICKER", "PERMNO"])
desc_map = desc_map.drop_duplicates(subset=["TICKER", "PERMNO"])
# If multiple TICKERs per PERMNO, keep first alphabetically
desc_map = desc_map.sort_values(["PERMNO", "TICKER"]).drop_duplicates(subset=["PERMNO"], keep="first")
desc_map = desc_map.rename(columns={"PERMNO": "permno"})

comp = comp.merge(desc_map, on="permno", how="inner")
print(f"  After TICKER link: {len(comp):,}")

# 1d. Get December shrout and cfacshr from CRSP for each permno-year
crsp_m = pd.read_parquet("data/crsp_monthly.parquet", columns=["permno", "date", "shrout"])
crsp_m["date"] = pd.to_datetime(crsp_m["date"])
crsp_m["yr"] = crsp_m["date"].dt.year
crsp_m["month"] = crsp_m["date"].dt.month
crsp_dec = crsp_m.loc[crsp_m["month"] == 12, ["permno", "yr", "shrout"]].copy()
crsp_dec = crsp_dec.dropna(subset=["shrout"])

cfa = pd.read_parquet("data/crsp_cfacshr.parquet")
cfa["date"] = pd.to_datetime(cfa["date"])
cfa["yr"] = cfa["date"].dt.year
cfa["month"] = cfa["date"].dt.month
cfa["permno"] = cfa["permno"].astype(int)
cfa_dec = cfa.loc[cfa["month"] == 12, ["permno", "yr", "cfacshr"]].copy()
cfa_dec = cfa_dec.dropna(subset=["cfacshr"])

crsp_dec = crsp_dec.merge(cfa_dec, on=["permno", "yr"], how="inner")
# Keep last December obs if duplicates
crsp_dec = crsp_dec.sort_values(["permno", "yr"]).drop_duplicates(
    subset=["permno", "yr"], keep="last"
)

# 1e. Merge CRSP shares into comp and compute EPS
#     CRSP shrout is in thousands; ib is in millions
#     eps = ib / ((shrout / 1000) * cfacshr)
comp = comp.rename(columns={"fyear": "yr"})
comp = comp.merge(crsp_dec, on=["permno", "yr"], how="inner")
comp["eps"] = comp["ib"] / ((comp["shrout"] / 1000) * comp["cfacshr"])
comp = comp[["TICKER", "yr", "eps"]].copy()
comp = comp.dropna(subset=["eps"])
comp = comp[np.isfinite(comp["eps"])]
print(f"  EPS obs (COMPUSTAT-based): {len(comp):,}")

# Build the 'act' DataFrame that the rest of the code expects
# (TICKER, yr, VALUE) — VALUE is the COMPUSTAT-based EPS
act = comp.rename(columns={"eps": "VALUE"})
# Keep one per TICKER-yr (last if duplicates)
act = act.sort_values(["TICKER", "yr"]).drop_duplicates(
    subset=["TICKER", "yr"], keep="last"
)
print(f"  Unique TICKER-yr: {len(act):,}")

# We still need ANNDATS for the LTG timing window, so load announcement dates from IBES actuals
print("Loading IBES announcement dates for LTG timing...")
act_ibes = pd.read_stata("data/EPS_unadj_act.dta")
act_ibes = act_ibes.loc[act_ibes["PDICITY"] == "ANN", ["TICKER", "PENDS", "ANNDATS"]].copy()
act_ibes = act_ibes.dropna(subset=["PENDS", "ANNDATS"])
act_ibes["yr"] = act_ibes["PENDS"].dt.year
# Keep one announcement date per TICKER-yr
act_ibes = act_ibes.sort_values(["TICKER", "yr", "ANNDATS"]).drop_duplicates(
    subset=["TICKER", "yr"], keep="last"
)
act_ibes = act_ibes[["TICKER", "yr", "ANNDATS"]].copy()
print(f"  IBES announcement dates: {len(act_ibes):,}")

# ── 2. Load LTG forecasts (FPI='0') using chunked reading ───────────────────
print("Loading LTG forecasts (chunked)...")
ltg_chunks = []
reader = pd.read_stata("data/EPS_unadj_forecast.dta", chunksize=50000)
for chunk in reader:
    sub = chunk.loc[chunk["FPI"] == "0", ["TICKER", "STATPERS", "MEANEST"]].copy()
    if len(sub) > 0:
        ltg_chunks.append(sub)
ltg = pd.concat(ltg_chunks, ignore_index=True)
ltg = ltg.dropna(subset=["STATPERS", "MEANEST"])
print(f"  LTG forecasts: {len(ltg):,} obs")

# ── 3. Match each LTG forecast to the most recent annual announcement ────────
#    Then keep only forecasts 30-90 days after that announcement.
print("Matching LTG forecasts to earnings announcements...")

# For each LTG forecast, find the most recent annual announcement before it.
# Use a cross-merge approach: merge on TICKER, then filter.
# To keep memory manageable, do this via merge_asof with proper sorting.
act_for_merge = (
    act_ibes[["TICKER", "ANNDATS", "yr"]]
    .rename(columns={"yr": "ann_yr"})
    .sort_values(["TICKER", "ANNDATS"])
    .drop_duplicates(subset=["TICKER", "ANNDATS"])
    .reset_index(drop=True)
)
ltg_sorted = ltg.sort_values(["TICKER", "STATPERS"]).reset_index(drop=True)

# merge_asof requires left keys sorted; sort by the on-key within each by-group
# pandas merge_asof needs STATPERS globally sorted when using by="TICKER"
# Actually it needs left_on sorted globally OR within groups. Let's just sort by STATPERS.
ltg_sorted = ltg_sorted.sort_values("STATPERS").reset_index(drop=True)
act_for_merge = act_for_merge.sort_values("ANNDATS").reset_index(drop=True)

matched = pd.merge_asof(
    ltg_sorted,
    act_for_merge,
    by="TICKER",
    left_on="STATPERS",
    right_on="ANNDATS",
    direction="backward",
)

# Compute days between announcement and forecast
matched["days_after"] = (matched["STATPERS"] - matched["ANNDATS"]).dt.days

# Keep only 30-90 day window
matched = matched.loc[
    (matched["days_after"] >= 30) & (matched["days_after"] <= 90)
].copy()
print(f"  After 30-90 day filter: {len(matched):,} obs")

# If multiple forecasts per firm in the window, keep the earliest STATPERS
matched = matched.sort_values(["TICKER", "ann_yr", "STATPERS"])
matched = matched.drop_duplicates(subset=["TICKER", "ann_yr"], keep="first")
print(f"  After keeping first per firm-year: {len(matched):,} obs")

# Formation year = fiscal year of the matched earnings announcement
matched["yr"] = matched["ann_yr"]

# ── 4. Build panel of LTG by TICKER-year ─────────────────────────────────────
panel = matched[["TICKER", "yr", "MEANEST"]].copy()
panel["value0"] = panel["MEANEST"] / 100  # LTG is in percent

# Sample period
panel = panel.loc[(panel["yr"] >= 1981) & (panel["yr"] <= 2015)].copy()
print(f"  Panel (1981-2015): {len(panel):,} obs")

# ── 5. Compute revisions by merging lagged LTG ──────────────────────────────
for k in [1, 2, 3]:
    lagged = panel[["TICKER", "yr", "value0"]].copy()
    lagged["yr"] = lagged["yr"] + k
    lagged = lagged.rename(columns={"value0": f"L{k}value0"})
    panel = panel.merge(lagged, on=["TICKER", "yr"], how="left")

# ── 6. Pre-winsorize actual EPS at 1/99 by year ─────────────────────────────
print("Pre-winsorizing actual EPS at 1/99 by year...")
act_win = act[["TICKER", "yr", "VALUE"]].copy()
lo = act_win.groupby("yr")["VALUE"].transform(lambda x: x.quantile(0.01))
hi = act_win.groupby("yr")["VALUE"].transform(lambda x: x.quantile(0.99))
act_win["VALUE"] = act_win["VALUE"].clip(lower=lo, upper=hi)

# Keep one actual per TICKER-yr (take last if duplicates)
act_win = act_win.sort_values(["TICKER", "yr"]).drop_duplicates(
    subset=["TICKER", "yr"], keep="last"
)

# ── 7. Merge future actual EPS for forecast errors ──────────────────────────
# eps5 = actual EPS at formation year t
# eps8 = actual EPS at t+3, eps9 = t+4, eps10 = t+5
panel = panel.merge(
    act_win.rename(columns={"VALUE": "eps5"}),
    on=["TICKER", "yr"],
    how="left",
)

for offset, label in [(3, "eps8"), (4, "eps9"), (5, "eps10")]:
    future = act_win.copy()
    future["yr"] = future["yr"] - offset
    future = future.rename(columns={"VALUE": label})
    panel = panel.merge(future, on=["TICKER", "yr"], how="left")

# ── 8. Compute forecast errors ──────────────────────────────────────────────
mask_pos = (panel["eps5"] > 0).fillna(False)
panel["y8"] = np.where(
    mask_pos & (panel["eps8"] > 0).fillna(False),
    (panel["eps8"] / panel["eps5"]) ** (1 / 3) - (1 + panel["value0"]),
    np.nan,
)
panel["y9"] = np.where(
    mask_pos & (panel["eps9"] > 0).fillna(False),
    (panel["eps9"] / panel["eps5"]) ** (1 / 4) - (1 + panel["value0"]),
    np.nan,
)
panel["y10"] = np.where(
    mask_pos & (panel["eps10"] > 0).fillna(False),
    (panel["eps10"] / panel["eps5"]) ** (1 / 5) - (1 + panel["value0"]),
    np.nan,
)

# ── 9. Winsorize at 5/95 by year ────────────────────────────────────────────
winsor_vars = ["value0", "L1value0", "L2value0", "L3value0", "y8", "y9", "y10"]
for v in winsor_vars:
    lo = panel.groupby("yr")[v].transform(lambda x: x.quantile(0.05))
    hi = panel.groupby("yr")[v].transform(lambda x: x.quantile(0.95))
    panel[v] = panel[v].clip(lower=lo, upper=hi)

# ── 10. Constant sample ─────────────────────────────────────────────────────
mask_y = panel[["y8", "y9", "y10"]].notna().all(axis=1)
mask_x = panel[["value0", "L1value0", "L2value0", "L3value0"]].notna().all(axis=1)
df = panel.loc[mask_y & mask_x].copy()
print(f"Constant sample: {len(df):,} obs")

# ── 11. Forecast revisions ──────────────────────────────────────────────────
for k in [1, 2, 3]:
    df[f"delta{k}"] = df["value0"] - df[f"L{k}value0"]

# ── 12. Year dummies ────────────────────────────────────────────────────────
year_dummies = pd.get_dummies(df["yr"], prefix="yr", drop_first=True, dtype=float)


# ── 13. Regression helper: OLS with year FE, clustered SE by year ────────────
def run_reg(y, x_name):
    """Return (beta, se) for coefficient on x_name."""
    endog = df[y].values
    exog = sm.add_constant(
        pd.concat(
            [df[[x_name]].reset_index(drop=True), year_dummies.reset_index(drop=True)],
            axis=1,
        )
    )
    groups = df["yr"].values
    model = sm.OLS(endog, exog, missing="drop")
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    beta = res.params[x_name]
    se = res.bse[x_name]
    return beta, se


# ── 14. Run 3x3 regressions ─────────────────────────────────────────────────
results = np.zeros((6, 3))  # 6 rows (coef+se for k=1,2,3), 3 cols (y8,y9,y10)

dep_vars = ["y8", "y9", "y10"]
for col_idx, yvar in enumerate(dep_vars):
    for k in [1, 2, 3]:
        beta, se = run_reg(yvar, f"delta{k}")
        row_b = 2 * (k - 1)
        row_s = row_b + 1
        results[row_b, col_idx] = beta
        results[row_s, col_idx] = se

# ── 15. Format and save ─────────────────────────────────────────────────────
header = f"{'':>20s} {'3-Year':>12s} {'4-Year':>12s} {'5-Year':>12s}"
lines = [
    "Table II: Coibion-Gorodnichenko Regressions for EPS",
    "=" * 60,
    "Dep var: forecast error (EPS_{t+n}/EPS_t)^{1/n} - LTG_t",
    "Indep var: LTG_t - LTG_{t-k}",
    "Year FE, SEs clustered by year",
    "",
    header,
    "-" * 60,
]

for k in [1, 2, 3]:
    row_b = 2 * (k - 1)
    row_s = row_b + 1
    coef_line = f"  LTG_t - LTG_{{t-{k}}}     "
    se_line = f"                      "
    for c in range(3):
        coef_line += f"{results[row_b, c]:12.4f}"
        se_line += f"  ({results[row_s, c]:.4f})"
    lines.append(coef_line)
    lines.append(se_line)
    lines.append("")

lines.append("-" * 60)
lines.append(f"N = {len(df):,}")
lines.append("")
lines.append("Note: LTG forecasts are selected 30-90 days after each annual")
lines.append("earnings announcement, following the paper's revisions03.do.")

table_text = "\n".join(lines)
print(table_text)

out_path = "BGLS2019_rep/replication/output/Table2.txt"
with open(out_path, "w") as f:
    f.write(table_text + "\n")

print(f"\nSaved to {out_path}")
