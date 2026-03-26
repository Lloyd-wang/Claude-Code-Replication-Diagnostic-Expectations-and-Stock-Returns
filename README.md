# Replication

Replication of Bordalo, Gennaioli, La Porta, and Shleifer (2019, Journal of Finance) using Claude Code. An experiment in AI-assisted replication of empirical finance research. All Python code was developed collaboratively with Claude Code (Anthropic). This repository is for academic and educational purposes only.

## Directory Structure

```
replication/
├── code/           Python scripts that produce each figure and table
├── output/         Generated outputs (figures, tables, intermediate datasets)
└── original/       Authors' original Stata and Python code (from JF replication package)
```

## Execution Order

1. **Data download**: Run `code/00_download_data.py` to download CRSP and COMPUSTAT data from WRDS into `../../data/`. IBES data and Fama-French factors must be uploaded manually (see `../../data/README.md`).
2. **Prerequisite**: Run `code/build_compustat_eps.py` to construct the COMPUSTAT-based per-share EPS panel (`output/compustat_eps_panel.parquet`).
3. **Independent scripts**: All numbered scripts (`11_*` through `16_*`) can be run independently after the prerequisite.

### Pre-built intermediate datasets (no build scripts)

The following datasets in `output/` were built interactively and do not yet have standalone build scripts. They are required by most numbered scripts:

- **`descriptive.parquet`** — Core firm-year panel (87,758 obs): IBES LTG deciles, COMPUSTAT EPS leads/lags, CRSP returns, merged via CCM+iclink
- **`crsp_monthly_filtered.parquet`** — Monthly returns filtered to sample firms (~3M obs)
- **`portfolio_returns.parquet`** — Equal-weighted and value-weighted monthly returns by LTG decile
- **`portfolio_assignments.parquet`** — Annual LTG decile assignments
- **`ltg_all_months.parquet`** — Monthly LTG forecasts at all available dates
- **`daily_ret_filtered.parquet`** — Daily returns filtered to sample firms (for Figure 5)

## Methodology Notes

- **Table II** uses analyst LTG forecasts made 30–90 days after annual earnings announcements (following `revisions03.do`), with forecast errors computed from COMPUSTAT `ib` per share.
- **Table III** estimates the diagnostic expectations model via Simulated Method of Moments (SMM), searching over parameters $(a, b, \sigma_f, \sigma_e, \theta, s)$.
- **Table IV** computes rolling 20-quarter AR(1) regressions for earnings persistence and volatility, with missing values filled by Fama-French 48 industry-year medians.
- **Figure 4** uses 3-year non-overlapping cohorts for forecast error computation, matching `JFsubmission.do`.
