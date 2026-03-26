"""
Figure 5: Earnings Announcement Returns
Following BGLS 2019 / JFsubmission.do lines 393-466 and Fig05DataSimple.do

Memory-efficient: process daily returns in chunks, compute event returns
per-chunk, never hold full daily dataset in memory.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import gc

OUT_DIR = '/workspaces/Dividend-growth/BGLS2019_rep/replication/output'
DATA_DIR = '/workspaces/Dividend-growth/data'

# ══════════════════════════════════════════════════
# Step 1: Load rdq dates and link to PERMNO
# ══════════════════════════════════════════════════
print("Loading rdq and CCM link...")
rdq = pd.read_parquet(f'{DATA_DIR}/comp_rdq.parquet')
rdq['rdq'] = pd.to_datetime(rdq['rdq'])

ccm = pd.read_parquet(f'{DATA_DIR}/ccm_link.parquet')
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])
ccm = ccm[ccm['linktype'].isin(['LC', 'LU']) & ccm['linkprim'].isin(['P', 'C'])]
ccm['permno'] = ccm['permno'].astype(int)

rdq_m = rdq.merge(ccm[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='gvkey', how='inner')
mask = (rdq_m['rdq'] >= rdq_m['linkdt'])
mask &= (rdq_m['rdq'] <= rdq_m['linkenddt']) | rdq_m['linkenddt'].isna()
rdq_m = rdq_m[mask].drop_duplicates(subset=['permno', 'rdq'])
rdq_m = rdq_m[['permno', 'rdq']].copy()
rdq_m['rdq_year'] = rdq_m['rdq'].dt.year
# Convert rdq to days since epoch for fast comparison (same as daily)
rdq_m['rdq_ord'] = (rdq_m['rdq'] - pd.Timestamp('1970-01-01')).dt.days
print(f"  rdq records: {len(rdq_m)}")

needed_permnos = set(rdq_m['permno'].unique())

# Build per-permno sorted arrays of rdq ordinals for fast lookup
from collections import defaultdict
permno_rdq = defaultdict(list)
for p, d_ord in zip(rdq_m['permno'], rdq_m['rdq_ord']):
    permno_rdq[p].append(d_ord)
# Convert to sorted numpy arrays
for k in permno_rdq:
    permno_rdq[k] = np.array(sorted(set(permno_rdq[k])))

# ══════════════════════════════════════════════════
# Step 2: Process daily returns in chunks
# For each chunk: filter to needed permnos, find returns within ±2 days
# of any rdq, assign to the closest rdq, accumulate results
# ══════════════════════════════════════════════════
print("Processing daily returns in chunks...")

# We'll accumulate (permno, rdq_ord, date, ret) tuples
event_rows = []
pf = pq.ParquetFile(f'{DATA_DIR}/daily_ret.parquet')

for i, batch in enumerate(pf.iter_batches(batch_size=3_000_000, columns=['PERMNO', 'date', 'RET'])):
    chunk = batch.to_pandas()
    chunk = chunk[chunk['PERMNO'].isin(needed_permnos)].copy()
    if len(chunk) == 0:
        continue

    chunk['date'] = pd.to_datetime(chunk['date'])
    chunk['RET'] = pd.to_numeric(chunk['RET'], errors='coerce')
    chunk['date_ord'] = (chunk['date'] - pd.Timestamp('1970-01-01')).dt.days

    # For each permno in this chunk, find daily returns near rdq dates
    for permno, pgrp in chunk.groupby('PERMNO'):
        rdq_ords = permno_rdq.get(permno)
        if rdq_ords is None:
            continue

        dates_ord = pgrp['date_ord'].values
        rets = pgrp['RET'].values

        # For each rdq, find returns within ±2 calendar days
        for rdq_ord in rdq_ords:
            within = np.abs(dates_ord - rdq_ord) <= 2
            if not within.any():
                continue
            window_rets = rets[within]
            valid = window_rets[~np.isnan(window_rets)]
            if len(valid) == 0:
                continue
            cum_r = np.prod(1 + valid)
            event_rows.append((permno, rdq_ord, cum_r))

    if (i + 1) % 5 == 0:
        print(f"  Processed {(i+1)*3:.0f}M rows, {len(event_rows)} event returns so far...")

    del chunk
    gc.collect()

print(f"  Total event returns: {len(event_rows)}")

# ══════════════════════════════════════════════════
# Step 3: Build event return DataFrame
# ══════════════════════════════════════════════════
evt_df = pd.DataFrame(event_rows, columns=['permno', 'rdq_ord', 'cum_r'])
del event_rows
gc.collect()

# Some rdq may have been split across chunks (unlikely but deduplicate)
# Actually each rdq's daily returns could span 2 chunks. We need to handle this.
# For simplicity, take the product of cum_r for duplicate (permno, rdq_ord)
evt_df = evt_df.groupby(['permno', 'rdq_ord']).agg(cum_r=('cum_r', 'prod')).reset_index()

# Convert rdq_ord (days since epoch) back to year
evt_df['rdq_year'] = evt_df['rdq_ord'].apply(
    lambda x: (pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(x))).year if x > 0 else np.nan)

# ══════════════════════════════════════════════════
# Step 4: Cumulate event returns within each calendar year
# ══════════════════════════════════════════════════
print("Cumulating within year...")
yr_evt = evt_df.groupby(['permno', 'rdq_year']).agg(
    cum_r=('cum_r', 'prod'),
    n_events=('cum_r', 'count')
).reset_index()
yr_evt.rename(columns={'rdq_year': 'yr'}, inplace=True)
yr_evt['yr'] = yr_evt['yr'].astype(int)

# ══════════════════════════════════════════════════
# Step 5: Create leads and lags
# ══════════════════════════════════════════════════
print("Creating leads/lags...")
yr_evt = yr_evt.sort_values(['permno', 'yr'])

# Only create panel for years we need (avoid huge cross product)
min_yr, max_yr = int(yr_evt['yr'].min()), int(yr_evt['yr'].max())
panel = yr_evt[['permno', 'yr', 'cum_r']].copy()

# Use shift within each permno group
for lag in range(1, 6):
    panel[f'evt_L{lag}'] = panel.groupby('permno')['cum_r'].shift(lag)
for lead in range(0, 6):
    panel[f'evt_F{lead}'] = panel.groupby('permno')['cum_r'].shift(-lead)

# ══════════════════════════════════════════════════
# Step 6: Merge with LTG deciles
# Stata: "replace yr=yr-1" then merge → event year t+1 maps to formation year t
# ══════════════════════════════════════════════════
print("Merging with LTG deciles...")
desc = pd.read_parquet(f'{OUT_DIR}/descriptive.parquet', columns=['PERMNO', 'yr', 'LTG'])
desc.rename(columns={'PERMNO': 'permno'}, inplace=True)

panel['form_yr'] = panel['yr'] - 1
merged = panel.merge(desc, left_on=['permno', 'form_yr'], right_on=['permno', 'yr'],
                     how='inner', suffixes=('', '_desc'))
merged = merged[merged['LTG'].notna()]

# Keep only where evt_F3 is available
max_yr_f3 = merged.loc[merged['evt_F3'].notna(), 'form_yr'].max()
merged = merged[merged['form_yr'] <= max_yr_f3]
print(f"  Merged records: {len(merged)}")

# ══════════════════════════════════════════════════
# Step 7: Bootstrap means for HLTG and LLTG
# evt_L4→t=-3, evt_L3→t=-2, ..., evt_L1→t=0, evt_F0→t=+1, ..., evt_F2→t=+3
# Plot t=-3 to +3
# ══════════════════════════════════════════════════
print("Bootstrapping...")
np.random.seed(1234)

def bootstrap_mean(vals, n_boot=1000):
    vals = vals[~np.isnan(vals)]
    if len(vals) < 10:
        return np.nan, np.nan, np.nan
    boot = np.array([np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n_boot)])
    return np.mean(vals), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

# After form_yr = yr - 1 (matching Stata's replace yr=yr-1):
# evt_L4 → form_yr-3, evt_L3 → form_yr-2, ..., evt_L1 → form_yr, evt_F0 → form_yr+1, ...
t_values = list(range(-3, 4))
evt_cols = ['evt_L4', 'evt_L3', 'evt_L2', 'evt_L1', 'evt_F0', 'evt_F1', 'evt_F2']

results = {}
for dec_name, dec_val in [('HLTG', 10), ('LLTG', 1)]:
    dec_data = merged[merged['LTG'] == dec_val]
    for t, col in zip(t_values, evt_cols):
        if col not in dec_data.columns:
            results[(dec_name, t)] = (np.nan, np.nan, np.nan)
            continue
        vals = (dec_data[col].dropna().values - 1) * 100  # percent
        results[(dec_name, t)] = bootstrap_mean(vals)

print(f"\n{'t':>4} {'HLTG mean':>10} {'LLTG mean':>10}")
for t in t_values:
    h = results[('HLTG', t)][0]
    l = results[('LLTG', t)][0]
    if not np.isnan(h):
        print(f"{t:>4} {h:>10.2f} {l:>10.2f}")

# ══════════════════════════════════════════════════
# Step 8: Plot
# ══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for dec_name, color in [('HLTG', 'red'), ('LLTG', 'blue')]:
    means = [results[(dec_name, t)][0] for t in t_values]
    ci_lo = [results[(dec_name, t)][1] for t in t_values]
    ci_hi = [results[(dec_name, t)][2] for t in t_values]

    ax.plot(t_values, means, '-', color=color, linewidth=2, label=dec_name)
    ax.plot(t_values, ci_lo, ':', color=color, linewidth=1)
    ax.plot(t_values, ci_hi, ':', color=color, linewidth=1)

ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('Years relative to formation', fontsize=11)
ax.set_ylabel('Returns (%)', fontsize=11)
ax.set_title('Figure 5: Earnings Announcement Returns', fontsize=13)
ax.set_xticks(range(-3, 4))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Figure5.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved Figure5.png")
