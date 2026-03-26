"""
Produce Figure 1 and Table I from BGLS (2019).

Figure 1: Bar chart of geometric mean annual EW returns by LTG decile (1981-2015).
Table I:  Summary statistics by LTG decile (N, mean LTG, EW/VW returns & SDs).

Inputs
------
replication/output/portfolio_returns.parquet
replication/output/descriptive.parquet

Outputs
-------
replication/output/Figure1.png
replication/output/Table1.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "output"

# ── Load data ────────────────────────────────────────────────────────────────
port = pd.read_parquet(OUT / "portfolio_returns.parquet")
desc = pd.read_parquet(OUT / "descriptive.parquet")

# Filter to 1981-2015
port = port[(port["yr"] >= 1981) & (port["yr"] <= 2015)].copy()
desc = desc[(desc["yr"] >= 1981) & (desc["yr"] <= 2015)].copy()
desc = desc[desc["LTG"].between(1, 10)].copy()

# ── Helper: compound monthly returns to annual, then geometric mean ──────────
def geo_mean_annual(df, ret_col):
    """
    For each (yr, LTG), compound monthly returns to an annual return.
    Then compute geometric mean across formation years.
    Returns a Series indexed by LTG.
    """
    annual = (
        df.groupby(["yr", "LTG"])[ret_col]
        .apply(lambda x: np.prod(1 + x) - 1)
        .reset_index(name="annual_ret")
    )
    geo = (
        annual.groupby("LTG")["annual_ret"]
        .apply(lambda x: np.exp(np.mean(np.log(1 + x))) - 1)
    )
    return geo

def std_annual(df, ret_col):
    """Standard deviation of annual returns across formation years."""
    annual = (
        df.groupby(["yr", "LTG"])[ret_col]
        .apply(lambda x: np.prod(1 + x) - 1)
        .reset_index(name="annual_ret")
    )
    return annual.groupby("LTG")["annual_ret"].std()


ew_geo = geo_mean_annual(port, "ewret")
vw_geo = geo_mean_annual(port, "vwret")
ew_sd = std_annual(port, "ewret")
vw_sd = std_annual(port, "vwret")

# ── Figure 1 ─────────────────────────────────────────────────────────────────
deciles = np.arange(1, 11)
labels = [f"D{d}" for d in deciles]
labels[0] = "LLTG\n(D1)"
labels[-1] = "HLTG\n(D10)"

spread = ew_geo.loc[1] - ew_geo.loc[10]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(deciles, ew_geo.loc[deciles].values * 100, color="steelblue",
              edgecolor="black", linewidth=0.5)
ax.set_xticks(deciles)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Geometric Mean Annual Return (%)", fontsize=10)
ax.set_title("Figure 1: Equal-Weighted Returns by LTG Decile (1981-2015)",
             fontsize=11)
ax.axhline(0, color="black", linewidth=0.5)

# Annotate spread
ax.text(
    0.98, 0.95,
    f"LLTG \u2013 HLTG spread: {spread * 100:.2f}%",
    transform=ax.transAxes, ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
)

plt.tight_layout()
fig.savefig(OUT / "Figure1.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT / 'Figure1.png'}")

# ── Table I (matching the paper) ─────────────────────────────────────────────

# --- Merge Compustat annual data via CCM link ---
DATA = Path(__file__).resolve().parent.parent / "data"
comp = pd.read_parquet(DATA / "comp_annual.parquet")
ccm = pd.read_parquet(DATA / "ccm_link.parquet")

# Keep primary links only
ccm = ccm[ccm["linktype"].isin(["LC", "LU"])].copy()
ccm = ccm[ccm["linkprim"].isin(["P", "C"])].copy()
ccm["permno"] = ccm["permno"].astype(int)
ccm = ccm[["gvkey", "permno"]].drop_duplicates()

# Merge comp with ccm to get permno
comp = comp.merge(ccm, on="gvkey", how="inner")
comp = comp[["permno", "fyear", "fyr", "at", "sale", "cogs", "ni", "csho", "seq", "pstk"]].copy()
comp = comp.rename(columns={"permno": "PERMNO"})

# Try merging on yr == fyear first, then fill with yr-1 == fyear for fyr > 6
desc_t = desc.copy()

# Merge attempt 1: fyear == yr
merge1 = desc_t.merge(
    comp, left_on=["PERMNO", "yr"], right_on=["PERMNO", "fyear"], how="left",
    suffixes=("", "_comp"),
)

# Merge attempt 2: fyear == yr - 1 (for firms with fiscal year ending after June)
comp2 = comp.copy()
comp2["yr_match"] = comp2["fyear"] + 1
merge2 = desc_t.merge(
    comp2[["PERMNO", "yr_match", "at", "sale", "cogs", "ni", "csho", "fyr"]],
    left_on=["PERMNO", "yr"], right_on=["PERMNO", "yr_match"], how="left",
    suffixes=("", "_alt"),
)

# Use merge1 where available, fill from merge2
for col in ["at", "sale", "cogs", "ni", "csho"]:
    if col in merge1.columns and col + "_alt" in merge2.columns:
        merge1[col] = merge1[col].fillna(merge2[col + "_alt"])
    elif col in merge1.columns:
        merge1[col] = merge1[col].fillna(merge2[col])

df = merge1.copy()

# Drop duplicate comp columns
drop_cols = [c for c in df.columns if c.endswith("_comp")]
df = df.drop(columns=drop_cols, errors="ignore")

# Cast nullable COMPUSTAT columns to float64 to avoid NA boolean ambiguity
for col in ["at", "sale", "cogs", "ni", "csho", "seq", "pstk"]:
    if col in df.columns:
        df[col] = df[col].astype("float64")

# --- Compute variables ---

# 1. Expected growth EPS (LTG) -- GROW_F0, in percent
df["ltg_pct"] = df["GROW_F0"]

# 2. Assets ($M) -- total assets from Compustat
df["assets"] = df["at"]

# 3. Market capitalisation ($M) -- mcap is in thousands
df["mcap_m"] = df["mcap"] / 1000.0

# 4. Size decile -- NYSE breakpoints on mcap
def assign_size_decile(row_mcap, nyse_breakpoints):
    """Assign size decile (1-10) based on NYSE breakpoints."""
    if pd.isna(row_mcap):
        return np.nan
    for i in range(9):
        if row_mcap <= nyse_breakpoints[i]:
            return i + 1
    return 10

size_deciles = []
for yr_val, grp in df.groupby("yr"):
    nyse = grp[grp["exchcd"] == 1.0]["mcap"].dropna()
    if len(nyse) == 0:
        grp["size_cat"] = np.nan
    else:
        bp = np.percentile(nyse, np.arange(10, 100, 10))
        grp["size_cat"] = grp["mcap"].apply(lambda x: assign_size_decile(x, bp))
    size_deciles.append(grp)
df = pd.concat(size_deciles, ignore_index=True)

# 5. Years publicly traded: (STATPERS - begdat) in years, where begdat is
#    the first CRSP listing date (min st_date per permno from crsp_stocknames).
_crsp_names = pd.read_parquet(
    Path(__file__).resolve().parent.parent / "data" / "crsp_stocknames.parquet",
    columns=["permno", "st_date"],
)
_crsp_names["st_date"] = pd.to_datetime(_crsp_names["st_date"])
_begdat = _crsp_names.groupby("permno")["st_date"].min().reset_index()
_begdat = _begdat.rename(columns={"permno": "PERMNO", "st_date": "begdat"})
df = df.merge(_begdat, on="PERMNO", how="left")
df["yrs_traded"] = (df["STATPERS"] - df["begdat"]).dt.days // 365
df["yrs_traded"] = df["yrs_traded"].astype(float)

# 6. %Listed after 5 years: (edate - STATPERS) > 5 years; only for yr <= 2011
df["listed_5yr"] = np.nan
mask_surv = df["yr"] <= 2011
df.loc[mask_surv, "listed_5yr"] = (
    ((df.loc[mask_surv, "edate"] - df.loc[mask_surv, "STATPERS"]).dt.days / 365.25) > 5
).astype(float)

# 7. Operating margin to assets: (sale - cogs) / at
df["mgn"] = np.where(df["at"] > 0, (df["sale"] - df["cogs"]) / df["at"], np.nan)

# 8. Return on equity: ni / seq (total NI / stockholders' equity, both from Compustat)
pfd = df["pstk"].fillna(0)
be_total = df["seq"] - pfd
df["roe"] = np.where(be_total > 0, df["ni"] / be_total, np.nan)

# 9. Percent EPS positive: eps_L1 > 0, use eps_F0 if eps_L1 missing
eps_val = df["eps_L1"].copy()
eps_val = eps_val.fillna(df["eps_F0"])
df["eps_pos"] = (eps_val > 0).astype(float)
df.loc[eps_val.isna(), "eps_pos"] = np.nan

# --- Winsorize at 1/99 by year ---
def winsorize_by_year(series, group_col, lo=0.01, hi=0.99):
    """Winsorize series within each year group."""
    result = series.copy()
    for yr_val, idx in series.groupby(group_col).groups.items():
        vals = series.loc[idx].dropna()
        if len(vals) < 10:
            continue
        lb = vals.quantile(lo)
        ub = vals.quantile(hi)
        result.loc[idx] = result.loc[idx].clip(lb, ub)
    return result

for var in ["ltg_pct", "assets", "mcap_m", "size_cat", "mgn", "roe"]:
    df[var] = winsorize_by_year(df[var], df["yr"])

# --- Collapse: mean within LTG-yr, then mean across years within LTG ---
agg_vars = {
    "ltg_pct": "mean",
    "assets": "mean",
    "mcap_m": "mean",
    "size_cat": "mean",
    "yrs_traded": "mean",
    "listed_5yr": "mean",
    "mgn": "mean",
    "roe": "mean",
    "eps_pos": "mean",
    "PERMNO": "count",  # number of observations
}

# First collapse: mean within (yr, LTG)
yr_ltg = df.groupby(["yr", "LTG"]).agg(agg_vars).reset_index()
yr_ltg = yr_ltg.rename(columns={"PERMNO": "n_obs"})

# Second collapse: mean across years within LTG
ltg_means = yr_ltg.groupby("LTG").mean(numeric_only=True)

# --- Format and print Table I ---
rows = {}
rows["Expected growth EPS (LTG)"] = ltg_means["ltg_pct"].apply(lambda x: f"{x:.0f}%")
rows["Assets ($M)"] = ltg_means["assets"].apply(lambda x: f"{x:,.0f}")
rows["Market capitalization ($M)"] = ltg_means["mcap_m"].apply(lambda x: f"{x:,.0f}")
rows["Size decile"] = ltg_means["size_cat"].apply(lambda x: f"{x:.1f}")
rows["Years publicly traded"] = ltg_means["yrs_traded"].apply(lambda x: f"{x:.1f}")
rows["%Listed after 5 years"] = ltg_means["listed_5yr"].apply(lambda x: f"{x * 100:.0f}%")
rows["Operating margin to assets"] = ltg_means["mgn"].apply(lambda x: f"{x * 100:.0f}%")
rows["Return on equity"] = ltg_means["roe"].apply(lambda x: f"{x * 100:.0f}%")
rows["Percent EPS positive"] = ltg_means["eps_pos"].apply(lambda x: f"{x * 100:.0f}%")
rows["Observations"] = ltg_means["n_obs"].apply(lambda x: f"{x:,.0f}")

# Build output table
header = f"{'Variable':<30s}" + "".join(f"{'D' + str(d):>10s}" for d in range(1, 11))
sep = "-" * (30 + 10 * 10)

lines = []
lines.append("Table I: Characteristics of LTG Portfolios (1981-2015)")
lines.append(sep)
lines.append(header)
lines.append(sep)
for label, vals in rows.items():
    line = f"{label:<30s}"
    for d in range(1, 11):
        line += f"{vals.loc[d]:>10s}"
    lines.append(line)
lines.append(sep)

table_txt = "\n".join(lines) + "\n"

with open(OUT / "Table1.txt", "w") as f:
    f.write(table_txt)

print(f"Saved {OUT / 'Table1.txt'}")
print()
print(table_txt)
