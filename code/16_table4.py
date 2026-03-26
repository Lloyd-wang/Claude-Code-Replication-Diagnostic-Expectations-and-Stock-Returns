"""
Table IV: Double Sorts on LTG and Earnings Persistence / Volatility
Replication of BGLS (2019) Table IV — Table04_A.do + Table04_B.do

Methodology (matching original Stata code):
  1. COMPUSTAT quarterly: NPS = niq/cshprq, split-adjusted
  2. LTM sum, log growth gnps, winsorize 1/99 by quarter
  3. Rolling 20-quarter AR(1) → slope_gnps (persistence)
  4. Prediction errors → rolling 20-quarter std → sderror_gnps (volatility)
  5. Keep last quarter per firm-year
  6. Add age from CRSP, filter >=5yr listing, match=min(age,10)
  7. Peer-group MEDIANS by (yr, match, industry) with hierarchical cascade
  8. Assign to ALL firms in descriptive via (yr, match, industry) merge
  9. Sort ALL firms into deciles by yr, then keep LLTG+HLTG
  10. EW monthly portfolio returns → log → annual → t-stats

Persistence: 1981-2015. Volatility: 1981-2012.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
OUT_DIR = 'replication/output'

# ============================================================================
# STEP 1: Compute quarterly EPS persistence and volatility from COMPUSTAT
# ============================================================================

# --- 1a. Load COMPUSTAT quarterly (niq) and merge with CCM link ---
comp = pd.read_parquet(f'{DATA_DIR}/comp_quarterly.parquet',
                       columns=['gvkey', 'datadate', 'fyearq', 'fqtr', 'niq', 'cshprq'])
comp = comp.dropna(subset=['niq'])  # pre-filter to save memory
ccm = pd.read_parquet(f'{DATA_DIR}/ccm_link.parquet')

ccm = ccm[ccm['linktype'].isin(['LU', 'LC'])].copy()
ccm = ccm[ccm['linkprim'].isin(['P', 'C'])].copy()
ccm['permno'] = ccm['permno'].astype(int)

comp['datadate'] = pd.to_datetime(comp['datadate'])
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))

merged = comp.merge(ccm[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='gvkey', how='inner')
merged = merged[(merged['datadate'] >= merged['linkdt']) &
                (merged['datadate'] <= merged['linkenddt'])].copy()
merged = merged.sort_values(['gvkey', 'datadate', 'permno'])
merged = merged.drop_duplicates(subset=['gvkey', 'datadate'], keep='first')

print(f"COMPUSTAT-CCM merged observations: {len(merged):,}")

# --- 1b. Compute NPS = niq / cshprq, split-adjust ---
cq = merged[['permno', 'datadate', 'niq', 'cshprq']].copy()
cq = cq.dropna(subset=['niq', 'cshprq'])
cq = cq[cq['cshprq'] > 0].copy()
cq['nps'] = cq['niq'] / cq['cshprq']

cfac = pd.read_parquet(f'{DATA_DIR}/crsp_cfacshr.parquet')
cfac['date'] = pd.to_datetime(cfac['date'])
cfac['permno'] = cfac['permno'].astype(int)
cfac = cfac.dropna(subset=['cfacshr'])

cq['ym'] = cq['datadate'].dt.to_period('M')
cfac['ym'] = cfac['date'].dt.to_period('M')
cq = cq.merge(cfac[['permno', 'ym', 'cfacshr']], on=['permno', 'ym'], how='left')
print(f"  cfacshr matched: {cq['cfacshr'].notna().sum():,} / {len(cq):,}")
cq['nps'] = cq['nps'] / cq['cfacshr']
cq = cq.dropna(subset=['nps'])

# Calendar quarter from datadate (matching Stata: qr=qofd(date))
cq['cal_yr'] = cq['datadate'].dt.year
cq['cal_q'] = (cq['datadate'].dt.month - 1) // 3  # 0-based quarter
cq['qr'] = cq['cal_yr'] * 4 + cq['cal_q']

# Average duplicates within (permno, qr) — Stata: collapse (mean) nps, by(permno permco qr)
cq = cq.groupby(['permno', 'qr']).agg(
    nps=('nps', 'mean'),
    datadate=('datadate', 'last'),
    cal_yr=('cal_yr', 'first'),
).reset_index()

# --- 1c. Consecutive calendar quarters ---
cq = cq.sort_values(['permno', 'qr']).reset_index(drop=True)
cq['prev_qr'] = cq.groupby('permno')['qr'].shift(1)
cq['is_consecutive'] = (cq['qr'] - cq['prev_qr']) == 1
cq['seq_break'] = (~cq['is_consecutive'].fillna(False)).astype(int)
cq['seq_id'] = cq.groupby('permno')['seq_break'].cumsum()

# --- 1d. Trailing 4-quarter sum (LTM EPS) ---
cq['Y'] = cq.groupby(['permno', 'seq_id'])['nps'].transform(
    lambda x: x.rolling(4, min_periods=4).sum()
)

# --- 1e. Log growth ---
cq['Y_pos'] = cq['Y'].where(cq['Y'] > 0)
cq['L1_Y_pos'] = cq.groupby(['permno', 'seq_id'])['Y_pos'].shift(1)
cq['gnps'] = np.log(cq['Y_pos']) - np.log(cq['L1_Y_pos'])

# --- 1f. Winsorize gnps at 1/99 by calendar quarter ---
lower = cq.groupby('qr')['gnps'].transform(lambda x: x.quantile(0.01))
upper = cq.groupby('qr')['gnps'].transform(lambda x: x.quantile(0.99))
cq['gnps'] = cq['gnps'].clip(lower=lower, upper=upper)

# --- 1g. Rolling 20-quarter AR(1) ---
WINDOW = 20
MIN_OBS = 20

cq['lag_gnps'] = cq.groupby(['permno', 'seq_id'])['gnps'].shift(1)
valid = cq.dropna(subset=['gnps', 'lag_gnps']).copy()
valid = valid.sort_values(['permno', 'qr']).reset_index(drop=True)
print(f"Valid quarterly observations for AR(1): {len(valid):,}")

print("Computing rolling AR(1) regressions...")
valid['gnps_f'] = valid['gnps'].astype('float64')
valid['lag_gnps_f'] = valid['lag_gnps'].astype('float64')
valid['xy'] = valid['gnps_f'] * valid['lag_gnps_f']
valid['x2'] = valid['lag_gnps_f'] ** 2

grp_key = ['permno', 'seq_id']
g = valid.groupby(grp_key)

roll_sum_y  = g['gnps_f'].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_OBS).sum())
roll_sum_x  = g['lag_gnps_f'].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_OBS).sum())
roll_sum_xy = g['xy'].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_OBS).sum())
roll_sum_x2 = g['x2'].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_OBS).sum())
roll_cnt    = g['gnps_f'].transform(lambda s: s.rolling(WINDOW, min_periods=MIN_OBS).count())

mean_y = roll_sum_y / roll_cnt
mean_x = roll_sum_x / roll_cnt
cov_xy = (roll_sum_xy - roll_cnt * mean_x * mean_y) / (roll_cnt - 1)
var_x  = (roll_sum_x2 - roll_cnt * mean_x ** 2) / (roll_cnt - 1)

valid['slope_gnps'] = cov_xy / var_x
valid['intercept'] = mean_y - valid['slope_gnps'] * mean_x

# Prediction error at each quarter and rolling std (volatility)
valid['error'] = valid['gnps_f'] - (valid['intercept'] + valid['slope_gnps'] * valid['lag_gnps_f'])
valid['sderror_gnps'] = valid.groupby(grp_key)['error'].transform(
    lambda s: s.rolling(WINDOW, min_periods=MIN_OBS).std()
)

ar1_results = valid[['permno', 'seq_id', 'qr', 'slope_gnps', 'sderror_gnps']].copy()
del valid, cq, roll_sum_y, roll_sum_x, roll_sum_xy, roll_sum_x2, roll_cnt
del mean_y, mean_x, cov_xy, var_x, merged, comp, cfac
import gc; gc.collect()
print(f"AR(1) results computed: {len(ar1_results):,}")

# --- 1h. Keep last quarter per firm-year ---
# Use calendar year of the END quarter for matching with descriptive yr.
# (Table04_A.do uses yr=year(window start), but that creates a ~5yr look-ahead.)
ar1_results = ar1_results.dropna(subset=['slope_gnps'])
ar1_results['year'] = ar1_results['qr'].apply(lambda q: q // 4)

# Keep last report per (permno, year) — matching Table04_A.do line 246-250
ar1_results = ar1_results.sort_values(['permno', 'year', 'qr'])
firm_year = ar1_results.groupby(['permno', 'year']).last().reset_index()
firm_year = firm_year[['permno', 'year', 'slope_gnps', 'sderror_gnps']].copy()

print(f"Firm-year obs with persistence: {firm_year['slope_gnps'].notna().sum():,}")
print(f"Firm-year obs with volatility:  {firm_year['sderror_gnps'].notna().sum():,}")

# ============================================================================
# STEP 2: Get FF48 industry mapping + age from CRSP
# ============================================================================

ff48 = pd.read_stata(
    'replication/original/'
    'Stata code for data analysis and pseudo-data/Data/industries.dta'
)
ff48 = ff48[['sic', 'industry']].copy()
ff48['sic'] = ff48['sic'].astype(int)

# Get SIC from CRSP monthly (broader coverage)
crsp_sic = pd.read_parquet(f'{OUT_DIR}/crsp_monthly_filtered.parquet',
                            columns=['permno', 'siccd'])
crsp_sic = crsp_sic.dropna(subset=['siccd']).copy()
crsp_sic['siccd'] = crsp_sic['siccd'].astype(int)
sic_map = crsp_sic.groupby('permno')['siccd'].agg(
    lambda x: x.value_counts().index[0]).reset_index()
sic_map = sic_map.merge(ff48, left_on='siccd', right_on='sic', how='left')
sic_map = sic_map[['permno', 'industry']].dropna(subset=['industry']).copy()

print(f"\nFF48 industry mapping: {len(sic_map):,} permnos")

# Age from CRSP stocknames
sn = pd.read_parquet(f'{DATA_DIR}/crsp_stocknames.parquet')
sn['st_date'] = pd.to_datetime(sn['st_date'])
sn['end_date'] = pd.to_datetime(sn['end_date'])
sn_begend = sn.groupby('permno').agg(
    ST_DATE=('st_date', 'min'),
    END_DATE=('end_date', 'max')
).reset_index()

# ============================================================================
# STEP 3: Compute peer-group MEDIANS of persistence and volatility
# Matching Table04_A.do / Table04_B.do: collapse (median) by (yr, match, industry)
# ============================================================================

# Add industry and age to firm-year AR(1) data
firm_year = firm_year.merge(sic_map, on='permno', how='left')
firm_year = firm_year.merge(sn_begend, on='permno', how='left')

# Add date for age calculation (use end-of-year as proxy)
firm_year['date'] = pd.to_datetime(firm_year['year'].astype(str) + '-12-31')
firm_year['age'] = ((firm_year['date'] - firm_year['ST_DATE']).dt.days / 365).apply(
    lambda x: int(x) if pd.notna(x) else np.nan)
firm_year['listing_years'] = ((firm_year['END_DATE'] - firm_year['ST_DATE']).dt.days / 365).apply(
    lambda x: int(x) if pd.notna(x) else np.nan)

# Apply 5-year listing filter (Table04_A.do line 262)
firm_year = firm_year[firm_year['yr_filter'] != True].copy() if False else firm_year.copy()
n_before_5yr = len(firm_year)
firm_year = firm_year[(firm_year['year'] >= 1981) &
                      (firm_year['listing_years'] >= 5)].copy()
print(f"\nAfter 5yr listing filter: {len(firm_year):,} (from {n_before_5yr:,})")

# match = min(age, 10)
firm_year['match'] = firm_year['age'].clip(upper=10)

# --- Persistence peer medians (Table04_A.do lines 268-293) ---
fy_pers = firm_year.dropna(subset=['slope_gnps', 'industry']).copy()

# Level 1: (yr, match, industry) — minimum 5 obs
peer_pers_1 = fy_pers.groupby(['year', 'match', 'industry']).agg(
    slope_gnps_peer=('slope_gnps', 'median'),
    obs=('slope_gnps', 'count')
).reset_index()
peer_pers_1 = peer_pers_1[peer_pers_1['obs'] >= 5].drop(columns='obs')

# Level 2: (match, industry)
peer_pers_2 = fy_pers.groupby(['match', 'industry']).agg(
    slope_gnps_peer=('slope_gnps', 'median'),
    obs=('slope_gnps', 'count')
).reset_index()
peer_pers_2 = peer_pers_2[peer_pers_2['obs'] >= 5].drop(columns='obs')

# Level 3: (yr, match)
peer_pers_3 = firm_year.dropna(subset=['slope_gnps']).groupby(['year', 'match']).agg(
    slope_gnps_peer=('slope_gnps', 'median'),
    obs=('slope_gnps', 'count')
).reset_index()
peer_pers_3 = peer_pers_3[peer_pers_3['obs'] >= 5].drop(columns='obs')

# Level 4: (match)
peer_pers_4 = firm_year.dropna(subset=['slope_gnps']).groupby(['match']).agg(
    slope_gnps_peer=('slope_gnps', 'median'),
    obs=('slope_gnps', 'count')
).reset_index()
peer_pers_4 = peer_pers_4[peer_pers_4['obs'] >= 5].drop(columns='obs')

print(f"Persistence peer groups: L1={len(peer_pers_1)}, L2={len(peer_pers_2)}, "
      f"L3={len(peer_pers_3)}, L4={len(peer_pers_4)}")

# --- Volatility peer medians (Table04_B.do lines 92-133) ---
fy_vol = firm_year.dropna(subset=['sderror_gnps']).copy()
fy_vol_ind = fy_vol[fy_vol['industry'].notna()].copy()

# Level 1: (yr, match, industry) — minimum 1 obs (Table04_B uses obs>=1)
peer_vol_1 = fy_vol_ind.groupby(['year', 'match', 'industry']).agg(
    sderror_peer=('sderror_gnps', 'median'),
    obs=('sderror_gnps', 'count')
).reset_index()
peer_vol_1 = peer_vol_1[peer_vol_1['obs'] >= 1].drop(columns='obs')

# Level 2: (match, industry)
peer_vol_2 = fy_vol_ind.groupby(['match', 'industry']).agg(
    sderror_peer=('sderror_gnps', 'median'),
    obs=('sderror_gnps', 'count')
).reset_index()
peer_vol_2 = peer_vol_2[peer_vol_2['obs'] >= 1].drop(columns='obs')

# Level 3: (yr, match)
peer_vol_3 = fy_vol.groupby(['year', 'match']).agg(
    sderror_peer=('sderror_gnps', 'median'),
    obs=('sderror_gnps', 'count')
).reset_index()
peer_vol_3 = peer_vol_3[peer_vol_3['obs'] >= 1].drop(columns='obs')

# Level 4: (match)
peer_vol_4 = fy_vol.groupby(['match']).agg(
    sderror_peer=('sderror_gnps', 'median'),
    obs=('sderror_gnps', 'count')
).reset_index()
peer_vol_4 = peer_vol_4[peer_vol_4['obs'] >= 1].drop(columns='obs')

# Level 5: (industry) — extra cascade in Table04_B
peer_vol_5 = fy_vol_ind.groupby(['industry']).agg(
    sderror_peer=('sderror_gnps', 'median'),
    obs=('sderror_gnps', 'count')
).reset_index()
peer_vol_5 = peer_vol_5[peer_vol_5['obs'] >= 1].drop(columns='obs')

# Level 6: (yr) — extra cascade in Table04_B
peer_vol_6 = fy_vol.groupby(['year']).agg(
    sderror_peer=('sderror_gnps', 'median'),
    obs=('sderror_gnps', 'count')
).reset_index()
peer_vol_6 = peer_vol_6[peer_vol_6['obs'] >= 1].drop(columns='obs')

print(f"Volatility peer groups: L1={len(peer_vol_1)}, L2={len(peer_vol_2)}, "
      f"L3={len(peer_vol_3)}, L4={len(peer_vol_4)}, L5={len(peer_vol_5)}, L6={len(peer_vol_6)}")

# ============================================================================
# STEP 4: Load descriptive, add age, merge peer medians with cascade
# ============================================================================

desc = pd.read_parquet(f'{OUT_DIR}/descriptive.parquet')
desc = desc[desc['LTG'].notna()].copy()
desc = desc.rename(columns={'PERMNO': 'permno'})
desc = desc[(desc['yr'] >= 1981) & (desc['yr'] <= 2015)].copy()

# Add age and industry
desc = desc.merge(sn_begend, on='permno', how='left')
desc['STATPERS'] = pd.to_datetime(desc['STATPERS'])
desc['age'] = ((desc['STATPERS'] - desc['ST_DATE']).dt.days / 365).apply(
    lambda x: int(x) if pd.notna(x) else np.nan)
desc['match'] = desc['age'].clip(upper=10)
desc = desc.merge(sic_map, on='permno', how='left')

# Drop firms without SIC or age (Table04_A.do line 299: drop if sic+age==.)
desc = desc.dropna(subset=['match']).copy()

print(f"\nDescriptive: {len(desc):,} obs after adding age/industry")

# --- Merge persistence peer medians with cascade ---
# Level 1: (yr, match, industry)
desc = desc.merge(peer_pers_1, left_on=['yr', 'match', 'industry'],
                  right_on=['year', 'match', 'industry'], how='left')
desc = desc.drop(columns=['year'], errors='ignore')
desc = desc.rename(columns={'slope_gnps_peer': 'avg_slope'})

# Level 2: (match, industry) — update missing
tmp = desc[['match', 'industry']].merge(peer_pers_2, on=['match', 'industry'], how='left')
desc['avg_slope'] = desc['avg_slope'].fillna(tmp['slope_gnps_peer'])

# Level 3: (yr, match)
tmp = desc[['yr', 'match']].merge(peer_pers_3, left_on=['yr', 'match'],
                                   right_on=['year', 'match'], how='left')
desc['avg_slope'] = desc['avg_slope'].fillna(tmp['slope_gnps_peer'])

# Level 4: (match)
tmp = desc[['match']].merge(peer_pers_4, on='match', how='left')
desc['avg_slope'] = desc['avg_slope'].fillna(tmp['slope_gnps_peer'])

# --- Merge volatility peer medians with cascade ---
# Level 1: (yr, match, industry)
tmp = desc[['yr', 'match', 'industry']].merge(
    peer_vol_1, left_on=['yr', 'match', 'industry'],
    right_on=['year', 'match', 'industry'], how='left')
desc['avg_sderror'] = tmp['sderror_peer']

# Level 2: (match, industry) — update missing
tmp2 = desc[['match', 'industry']].merge(peer_vol_2, on=['match', 'industry'], how='left')
desc['avg_sderror'] = desc['avg_sderror'].fillna(tmp2['sderror_peer'])

# Level 3: (yr, match)
tmp3 = desc[['yr', 'match']].merge(peer_vol_3, left_on=['yr', 'match'],
                                    right_on=['year', 'match'], how='left')
desc['avg_sderror'] = desc['avg_sderror'].fillna(tmp3['sderror_peer'])

# Level 4: (match)
tmp4 = desc[['match']].merge(peer_vol_4, on='match', how='left')
desc['avg_sderror'] = desc['avg_sderror'].fillna(tmp4['sderror_peer'])

n_pers = desc['avg_slope'].notna().sum()
n_vol = desc['avg_sderror'].notna().sum()
print(f"  Persistence matched: {n_pers:,} ({n_pers/len(desc)*100:.1f}%)")
print(f"  Volatility matched:  {n_vol:,} ({n_vol/len(desc)*100:.1f}%)")

# ============================================================================
# STEP 5: Portfolio formation and return computation
# Sort ALL firms into deciles, then keep LLTG+HLTG (matching original)
# ============================================================================

crsp = pd.read_parquet(f'{OUT_DIR}/crsp_monthly_filtered.parquet')
crsp = crsp[['permno', 'date', 'year', 'month', 'ret']].dropna(subset=['ret']).copy()
crsp['gross'] = 1 + crsp['ret']


def compute_panel(desc_panel, sort_col, panel_label):
    """
    Compute double-sort portfolio returns for one panel.
    Original: sort ALL firms into deciles by yr, then keep LLTG+HLTG for portfolios.
    """
    # Sort ALL firms into deciles by year (Table04_A.do line 332)
    panel = desc_panel.copy()
    panel_valid = panel.dropna(subset=[sort_col]).copy()
    panel_valid['decile'] = panel_valid.groupby('yr')[sort_col].transform(
        lambda x: pd.qcut(x.rank(method='first'), 10, labels=range(1, 11))
    ).astype(int)

    # Group: 0=missing, 1=Bottom30%, 2=Middle, 3=Top30%
    panel_valid['Xgroup'] = 2
    panel_valid.loc[panel_valid['decile'] <= 3, 'Xgroup'] = 1
    panel_valid.loc[panel_valid['decile'] >= 8, 'Xgroup'] = 3

    # Missing → group 0
    panel_miss = panel[panel[sort_col].isna()].copy()
    panel_miss['Xgroup'] = 0
    panel_all = pd.concat([panel_valid, panel_miss], ignore_index=True)

    # NOW keep only LLTG and HLTG for portfolio computation
    panel_all = panel_all[panel_all['LTG'].isin([1, 10])].copy()

    print(f"\n  {panel_label}:")
    print(f"    Total LLTG+HLTG obs: {len(panel_all):,}")
    print(f"    Bottom 30%: {(panel_all['Xgroup']==1).sum():,}")
    print(f"    Top 30%:    {(panel_all['Xgroup']==3).sum():,}")

    # Merge with forward-year monthly returns
    portfolio = panel_all[['permno', 'yr', 'LTG', 'Xgroup']].copy()
    portfolio['fwd_yr'] = portfolio['yr'] + 1

    port_ret = portfolio.merge(
        crsp[['permno', 'year', 'month', 'gross']],
        left_on=['permno', 'fwd_yr'], right_on=['permno', 'year'],
        how='inner'
    )
    print(f"    Monthly obs: {len(port_ret):,}")

    # EW average gross returns by (LTG, Xgroup, yr, month)
    ew = port_ret.groupby(['LTG', 'Xgroup', 'yr', 'month'])['gross'].mean().reset_index(
        name='fgross')

    # Pivot to get LLTG and HLTG for spread
    piv = ew.pivot_table(index=['Xgroup', 'yr', 'month'], columns='LTG',
                         values='fgross').reset_index()
    piv.columns.name = None

    if 1 not in piv.columns or 10 not in piv.columns:
        print("    WARNING: Missing LTG groups")
        return None

    # Create spread: HLTG - LLTG
    piv['spread_fgross'] = 1 + piv[10] - piv[1]

    # Build rows for LLTG, HLTG, and spread
    rows = []
    for ltg_code, col_name in [(1, 1), (10, 10), (11, 'spread_fgross')]:
        sub = piv[['Xgroup', 'yr', 'month']].copy()
        sub['fgross'] = piv[col_name] if col_name != 'spread_fgross' else piv['spread_fgross']
        sub['LTG'] = ltg_code
        rows.append(sub)
    all_rows = pd.concat(rows, ignore_index=True)

    # Create Top-Bottom spread
    piv_grp = all_rows.pivot_table(index=['LTG', 'yr', 'month'], columns='Xgroup',
                                   values='fgross').reset_index()
    piv_grp.columns.name = None

    tb_rows = []
    if 1 in piv_grp.columns and 3 in piv_grp.columns:
        tb = piv_grp[['LTG', 'yr', 'month']].copy()
        tb['fgross'] = 1 + piv_grp[3] - piv_grp[1]
        tb['Xgroup'] = 4
        tb_rows.append(tb)

    grp1 = all_rows[all_rows['Xgroup'] == 1].copy()
    grp3 = all_rows[all_rows['Xgroup'] == 3].copy()
    combined = pd.concat([grp1, grp3] + tb_rows, ignore_index=True)

    # Log returns → annual
    combined['lnret'] = np.log(combined['fgross'])
    annual = combined.groupby(['LTG', 'Xgroup', 'yr'])['lnret'].sum().reset_index()

    # Time-series mean and semean
    summary = annual.groupby(['LTG', 'Xgroup']).agg(
        mean_lnret=('lnret', 'mean'),
        se_lnret=('lnret', lambda x: x.std() / np.sqrt(len(x))),
        n_years=('lnret', 'count')
    ).reset_index()

    return summary


print("\nComputing persistence portfolio returns...")
desc_pers = desc[(desc['yr'] >= 1981) & (desc['yr'] <= 2015)].copy()
pers_summary = compute_panel(desc_pers, 'avg_slope', 'Persistence (1981-2015)')

print("\nComputing volatility portfolio returns...")
desc_vol = desc[(desc['yr'] >= 1981) & (desc['yr'] <= 2012)].copy()
vol_summary = compute_panel(desc_vol, 'avg_sderror', 'Volatility (1981-2012)')


# ============================================================================
# FORMAT OUTPUT
# ============================================================================

def get_val(summary, ltg, xgroup, col='mean_lnret'):
    row = summary[(summary['LTG'] == ltg) & (summary['Xgroup'] == xgroup)]
    if len(row) > 0:
        return row.iloc[0][col]
    return np.nan


def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '     ---'
    return f'{val * 100:8.1f}%'


def fmt_t(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '     ---'
    return f'{val:8.2f}'


lines = []
lines.append("=" * 85)
lines.append("Table IV: Double Sorts on LTG and Earnings Persistence / Volatility")
lines.append("BGLS (2019) Replication")
lines.append("=" * 85)
lines.append("")
lines.append("Annual log returns (EW portfolios).")
lines.append("Persistence: 1981-2015. Volatility: 1981-2012.")
lines.append("")

h1 = (f"{'':>14}{'Persistence of':^28}"
      f"{'':>4}{'Volatility of':^28}")
h2 = (f"{'':>14}{'Growth in EPS':^28}"
      f"{'':>4}{'Prediction Error of Growth in EPS':^35}")
sep = (f"{'':>14}{'-'*10}{' '}{'-'*10}{' '}{'-'*10}"
       f"{'':>4}{'-'*10}{' '}{'-'*10}{' '}{'-'*10}")
h3 = (f"{'':>14}{'Bottom':>10}{'Top':>10}{'t-Stats':>10}"
      f"{'':>4}{'Bottom':>10}{'Top':>10}{'t-Stats':>10}")
h4 = (f"{'':>14}{'30%':>10}{'30%':>10}{'Top-Bot':>10}"
      f"{'':>4}{'30%':>10}{'30%':>10}{'Top-Bot':>10}")

lines.append(h1)
lines.append(h2)
lines.append(sep)
lines.append(h3)
lines.append(h4)
lines.append(sep)

for row_label, ltg_code in [('LLTG', 1), ('HLTG', 10), ('HLTG-LLTG', 11)]:
    p_bot = get_val(pers_summary, ltg_code, 1)
    p_top = get_val(pers_summary, ltg_code, 3)
    p_tb_mean = get_val(pers_summary, ltg_code, 4, 'mean_lnret')
    p_tb_se = get_val(pers_summary, ltg_code, 4, 'se_lnret')
    p_tstat = p_tb_mean / p_tb_se if pd.notna(p_tb_se) and p_tb_se > 0 else np.nan

    v_bot = get_val(vol_summary, ltg_code, 1)
    v_top = get_val(vol_summary, ltg_code, 3)
    v_tb_mean = get_val(vol_summary, ltg_code, 4, 'mean_lnret')
    v_tb_se = get_val(vol_summary, ltg_code, 4, 'se_lnret')
    v_tstat = v_tb_mean / v_tb_se if pd.notna(v_tb_se) and v_tb_se > 0 else np.nan

    lines.append(
        f"{row_label:>14}"
        f"{fmt_pct(p_bot):>10}"
        f"{fmt_pct(p_top):>10}"
        f"{fmt_t(p_tstat):>10}"
        f"{'':>4}"
        f"{fmt_pct(v_bot):>10}"
        f"{fmt_pct(v_top):>10}"
        f"{fmt_t(v_tstat):>10}"
    )

lines.append(sep)
lines.append("")
lines.append("Notes:")
lines.append("  Sorting: industry-year average persistence/volatility (FF48 x year).")
lines.append("  Persistence = AR(1) slope from rolling 20-quarter regression of LTM log EPS growth.")
lines.append("  Volatility = std dev of AR(1) prediction errors over rolling 20 quarters.")
lines.append("  EPS = niq/cshprq from COMPUSTAT quarterly, split-adjusted by cfacshr.")
lines.append("  Returns: EW monthly gross returns -> log -> sum to annual.")
lines.append("  t-stats = mean(Top-Bottom spread) / semean.")
lines.append("=" * 85)
lines.append("")
lines.append("Paper values for comparison:")
lines.append("  Persistence: LLTG(15.0%, 12.8%, -0.73) HLTG(7.8%, 0.9%, -2.46) "
             "HLTG-LLTG(-7.2%, -11.5%, -2.05)")
lines.append("  Volatility:  LLTG(14.9%, 17.3%, 0.80) HLTG(4.9%, 0.6%, -1.82) "
             "HLTG-LLTG(-9.8%, -16.6%, -2.42)")

table_text = "\n".join(lines)
print("\n" + table_text)

with open(f'{OUT_DIR}/Table4.txt', 'w') as fout:
    fout.write(table_text + "\n")
print(f"\nSaved Table4.txt")
