"""
00_download_data.py — Download WRDS data and build COMPUSTAT EPS panel
=====================================================================
Run from the repo root (BGLS2019_rep/).

Part 1 — WRDS downloads (into data/):
  comp_annual.parquet      — COMPUSTAT annual (funda): ib, ni, sale, ceq, etc.
  comp_quarterly.parquet   — COMPUSTAT quarterly (fundq): niq, ibq, cshprq, rdq
  crsp_monthly.parquet     — CRSP monthly (msf + msenames + msedelist)
  crsp_cfacshr.parquet     — CRSP monthly cfacshr lookup (permno × date)
  crsp_stocknames.parquet  — CRSP stocknames (permno, st_date, end_date)
  ccm_link.parquet         — CRSP-COMPUSTAT Merged link history

Part 2 — Build COMPUSTAT EPS panel (into replication/output/):
  compustat_eps_panel.parquet — ib per share, split-adjusted (PERMNO × year)

Data NOT downloaded here (must be obtained separately):
  daily_ret.parquet                        — CRSP daily (83M rows)
  EPS_unadj_act.dta, EPS_unadj_forecast.dta  — IBES
  iclink_updated.dta                          — IBES-CRSP link
  FF_5_month.csv, daily_FF3.csv               — Fama-French factors

WRDS credentials: set WRDS_USER and PGPASSWORD env vars or enter interactively.
Sample period: 1970-2018 (covers original paper's 1981-2015 with lookback).
"""

import wrds
import numpy as np
import pandas as pd
import os
import gc
import time

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

WRDS_USER = os.environ.get('WRDS_USER', 'your_wrds_username')

t0 = time.time()
print("Connecting to WRDS...")
db = wrds.Connection(wrds_username=WRDS_USER)
print(f"Connected. ({time.time()-t0:.0f}s)\n")


# ============================================================================
# 1. COMPUSTAT ANNUAL (comp.funda)
#    Key vars: ib (EPS), ni, sale, ceq, seq, at, lt, sich (SIC), fyr, etc.
# ============================================================================
print("=" * 70)
print("1. Downloading COMPUSTAT Annual (comp.funda)...")
chunks = []
for ys, ye in [(1970,1979),(1980,1989),(1990,1999),(2000,2009),(2010,2018)]:
    print(f"   {ys}-{ye}...", end=" ", flush=True)
    df = db.raw_sql(f"""
    SELECT gvkey, datadate, fyear, fyr,
           ib, ni, sale, cogs, ebit, epspx,
           prcc_f, csho, cshpri, adjex_f,
           ceq, seq, pstk, pstkl, pstkrv, txditc, txdb, itcb,
           at, lt, sich,
           act, che, lct, dlc, dp,
           ibc, oancf, xidoc, dvc
    FROM comp.funda
    WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
      AND fyear BETWEEN {ys} AND {ye}
    """)
    print(f"{len(df):,} rows ({time.time()-t0:.0f}s)")
    chunks.append(df)
comp_annual = pd.concat(chunks, ignore_index=True)
print(f"   Total: {len(comp_annual):,} rows")
comp_annual.to_parquet(f'{DATA_DIR}/comp_annual.parquet', index=False)
print(f"   Saved: comp_annual.parquet\n")
del chunks; gc.collect()


# ============================================================================
# 2. COMPUSTAT QUARTERLY (comp.fundq)
#    Key vars: niq (net income), ibq, cshprq, rdq (announcement date)
# ============================================================================
print("=" * 70)
print("2. Downloading COMPUSTAT Quarterly (comp.fundq)...")
chunks = []
for ys, ye in [(1970,1979),(1980,1989),(1990,1999),(2000,2009),(2010,2018)]:
    print(f"   {ys}-{ye}...", end=" ", flush=True)
    df = db.raw_sql(f"""
    SELECT gvkey, datadate, fyearq, fqtr, fyr,
           niq, ibq, cshprq,
           saleq, cogsq, atq, rdq
    FROM comp.fundq
    WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
      AND fyearq BETWEEN {ys} AND {ye}
    """)
    print(f"{len(df):,} rows ({time.time()-t0:.0f}s)")
    chunks.append(df)
comp_qtr = pd.concat(chunks, ignore_index=True)
print(f"   Total: {len(comp_qtr):,} rows")
comp_qtr.to_parquet(f'{DATA_DIR}/comp_quarterly.parquet', index=False)
print(f"   Saved: comp_quarterly.parquet\n")
del chunks; gc.collect()


# ============================================================================
# 3. CRSP MONTHLY (crsp.msf + msenames + msedelist)
#    Key vars: ret, prc, shrout, cfacshr, cfacpr, shrcd, exchcd, siccd,
#              dlstcd, dlret
#    Server-side join with msenames for shrcd/exchcd/siccd
# ============================================================================
print("=" * 70)
print("3. Downloading CRSP Monthly (crsp.msf + msenames)...")
chunks = []
for ys, ye in [(1970,1979),(1980,1989),(1990,1999),(2000,2009),(2010,2018)]:
    print(f"   {ys}-{ye}...", end=" ", flush=True)
    df = db.raw_sql(f"""
    SELECT DISTINCT ON (a.permno, a.date)
           a.permno, a.permco, a.date,
           a.ret, a.retx, a.prc, a.shrout,
           a.cfacshr, a.cfacpr,
           b.shrcd, b.exchcd, b.siccd
    FROM crsp.msf AS a
    LEFT JOIN crsp.msenames AS b
      ON a.permno = b.permno
      AND a.date >= b.namedt AND a.date <= b.nameendt
    WHERE a.date BETWEEN '{ys}-01-01' AND '{ye}-12-31'
    ORDER BY a.permno, a.date, b.namedt DESC
    """)
    print(f"{len(df):,} rows ({time.time()-t0:.0f}s)")
    chunks.append(df)
crsp_msf = pd.concat(chunks, ignore_index=True)
crsp_msf['date'] = pd.to_datetime(crsp_msf['date'])
print(f"   MSF total: {len(crsp_msf):,} rows")

# Merge delisting returns
print("   Downloading delisting returns (crsp.msedelist)...", end=" ", flush=True)
delist = db.raw_sql("SELECT permno, dlstdt, dlstcd, dlret FROM crsp.msedelist")
print(f"{len(delist):,} rows ({time.time()-t0:.0f}s)")
delist['dlstdt'] = pd.to_datetime(delist['dlstdt'])
delist['ym'] = delist['dlstdt'].dt.to_period('M')
crsp_msf['ym'] = crsp_msf['date'].dt.to_period('M')
crsp_monthly = crsp_msf.merge(
    delist[['permno', 'ym', 'dlstcd', 'dlret']], on=['permno', 'ym'], how='left'
)
crsp_monthly = crsp_monthly.drop(columns=['ym'])
crsp_monthly = crsp_monthly.drop_duplicates(subset=['permno', 'date'], keep='first')
print(f"   Final: {len(crsp_monthly):,} rows")
crsp_monthly.to_parquet(f'{DATA_DIR}/crsp_monthly.parquet', index=False)
print(f"   Saved: crsp_monthly.parquet\n")


# ============================================================================
# 4. CRSP CFACSHR (derived from monthly, for split-adjusting quarterly EPS)
# ============================================================================
print("=" * 70)
print("4. Building CRSP cfacshr lookup...")
cfac = crsp_monthly[['permno', 'date', 'cfacshr', 'cfacpr']].dropna(subset=['cfacshr']).copy()
cfac = cfac.drop_duplicates(subset=['permno', 'date'], keep='first')
cfac.to_parquet(f'{DATA_DIR}/crsp_cfacshr.parquet', index=False)
print(f"   Saved: crsp_cfacshr.parquet ({len(cfac):,} rows)\n")
del crsp_msf, crsp_monthly, cfac; gc.collect()


# ============================================================================
# 5. CRSP DAILY — SKIPPED
#    daily_ret.parquet (83M rows, 1.6GB) is already available locally.
#    Contains: PERMNO, date, RET, PRC, SHROUT, VOL, vwretd.
#    Used only by Figure 5 (event study around earnings announcements).
# ============================================================================
print("=" * 70)
print("5. CRSP Daily: SKIPPED (daily_ret.parquet already exists, 83M rows)\n")


# ============================================================================
# 6. CRSP STOCKNAMES
#    Key vars: st_date, end_date (for firm age), ticker, shrcd, exchcd
# ============================================================================
print("=" * 70)
print("6. Downloading CRSP Stocknames...")
stocknames = db.raw_sql("""
SELECT permno, permco, namedt, nameenddt,
       ticker, cusip, ncusip,
       shrcd, exchcd, siccd,
       st_date, end_date
FROM crsp.stocknames
""")
print(f"   Rows: {len(stocknames):,} ({time.time()-t0:.0f}s)")
stocknames.to_parquet(f'{DATA_DIR}/crsp_stocknames.parquet', index=False)
print(f"   Saved: crsp_stocknames.parquet\n")


# ============================================================================
# 7. CCM LINK (CRSP-COMPUSTAT Merged)
#    For matching COMPUSTAT gvkey to CRSP permno
# ============================================================================
print("=" * 70)
print("7. Downloading CCM Link History...")
ccm = db.raw_sql("""
SELECT gvkey, lpermno AS permno, lpermco AS permco,
       linktype, linkprim, linkdt, linkenddt
FROM crsp.ccmxpf_lnkhist
""")
print(f"   Rows: {len(ccm):,} ({time.time()-t0:.0f}s)")
ccm.to_parquet(f'{DATA_DIR}/ccm_link.parquet', index=False)
print(f"   Saved: ccm_link.parquet\n")


# ============================================================================
# Summary
# ============================================================================
db.close()
print("=" * 70)
print(f"DOWNLOAD COMPLETE ({time.time()-t0:.0f}s). Files in {DATA_DIR}:")
print("=" * 70)
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith('.parquet'):
        sz = os.path.getsize(f'{DATA_DIR}/{f}') / 1e6
        print(f"  {f:40s}  {sz:8.1f} MB")


# ============================================================================
# 8. BUILD COMPUSTAT EPS PANEL (matching earnings.do)
#    eps = ib / ((shrout/1000) * cfacshr)
#    Output: replication/output/compustat_eps_panel.parquet
# ============================================================================
OUT_DIR = 'replication/output'
os.makedirs(OUT_DIR, exist_ok=True)

print("\n" + "=" * 70)
print("8. Building COMPUSTAT EPS panel...")

# 8a. Load COMPUSTAT annual ib
comp_eps = pd.read_parquet(f'{DATA_DIR}/comp_annual.parquet',
                           columns=['gvkey', 'fyear', 'ib'])
comp_eps = comp_eps.dropna(subset=['ib'])
comp_eps = comp_eps.rename(columns={'fyear': 'yr'})

# 8b. CCM link: gvkey → permno
ccm_eps = pd.read_parquet(f'{DATA_DIR}/ccm_link.parquet')
ccm_eps = ccm_eps[ccm_eps['linktype'].isin(['LC', 'LU'])].copy()
ccm_eps = ccm_eps[ccm_eps['linkprim'].isin(['P', 'C'])].copy()
ccm_eps['permno'] = ccm_eps['permno'].astype(int)
ccm_eps['priority'] = ccm_eps['linktype'].map({'LC': 0, 'LU': 1})
ccm_eps = ccm_eps.sort_values('priority').drop_duplicates(subset=['gvkey'], keep='first')
ccm_eps = ccm_eps[['gvkey', 'permno']].copy()

comp_eps = comp_eps.merge(ccm_eps, on='gvkey', how='inner')
comp_eps = comp_eps.rename(columns={'permno': 'PERMNO'})

# 8c. CRSP December shrout and cfacshr
crsp_eps = pd.read_parquet(f'{DATA_DIR}/crsp_monthly.parquet',
                           columns=['permno', 'date', 'shrout'])
crsp_eps['date'] = pd.to_datetime(crsp_eps['date'])
crsp_eps['yr'] = crsp_eps['date'].dt.year
crsp_eps['month'] = crsp_eps['date'].dt.month

cfac_eps = pd.read_parquet(f'{DATA_DIR}/crsp_cfacshr.parquet')
cfac_eps['date'] = pd.to_datetime(cfac_eps['date'])
cfac_eps['yr'] = cfac_eps['date'].dt.year
cfac_eps['month'] = cfac_eps['date'].dt.month

crsp_eps = crsp_eps.merge(cfac_eps[['permno', 'yr', 'month', 'cfacshr']],
                          on=['permno', 'yr', 'month'], how='left')

crsp_eps = crsp_eps.sort_values(['permno', 'yr', 'month'])
crsp_dec = crsp_eps[crsp_eps['month'] == 12].drop_duplicates(
    subset=['permno', 'yr'], keep='last')
crsp_other = crsp_eps.drop_duplicates(subset=['permno', 'yr'], keep='last')
crsp_dec = pd.concat([crsp_dec,
                      crsp_other[~crsp_other.set_index(['permno', 'yr']).index
                                  .isin(crsp_dec.set_index(['permno', 'yr']).index)]])
crsp_dec = crsp_dec[['permno', 'yr', 'shrout', 'cfacshr']].copy()
crsp_dec = crsp_dec.rename(columns={'permno': 'PERMNO'})
crsp_dec = crsp_dec.dropna(subset=['shrout', 'cfacshr'])

# 8d. Merge and compute eps
df_eps = comp_eps.merge(crsp_dec, on=['PERMNO', 'yr'], how='inner')
df_eps = df_eps[df_eps['cfacshr'] >= 0.001].copy()
df_eps['shsplit'] = (df_eps['shrout'] / 1000.0) * df_eps['cfacshr']
df_eps['eps'] = np.where(df_eps['shsplit'] >= 0.1,
                         df_eps['ib'] / df_eps['shsplit'], np.nan)
df_eps = df_eps.dropna(subset=['eps'])
df_eps['abs_ib'] = df_eps['ib'].abs()
df_eps = df_eps.sort_values(['PERMNO', 'yr', 'abs_ib'], ascending=[True, True, False])
df_eps = df_eps.drop_duplicates(subset=['PERMNO', 'yr'], keep='first')
df_eps['eps'] = df_eps['eps'].clip(-500, 500)

result = df_eps[['PERMNO', 'yr', 'eps']].copy()
result.to_parquet(f'{OUT_DIR}/compustat_eps_panel.parquet', index=False)
print(f"   EPS panel: {len(result):,} obs, {result['PERMNO'].nunique():,} firms, "
      f"years {result['yr'].min()}-{result['yr'].max()}")
print(f"   Saved: {OUT_DIR}/compustat_eps_panel.parquet")
print(f"\nALL DONE ({time.time()-t0:.0f}s)")
