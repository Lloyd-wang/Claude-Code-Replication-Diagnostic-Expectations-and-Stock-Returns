# Replication Code

Each script produces one or more figures/tables from BGLS (2019). Scripts are prefixed by paper section number for ordering.

## Prerequisites

| Script | Description |
|--------|-------------|
| `00_download_data.py` | Downloads CRSP and COMPUSTAT data from WRDS into `../../data/`. Requires WRDS account credentials. |
| `build_compustat_eps.py` | Constructs per-share EPS from COMPUSTAT `ib`, CRSP `shrout`, and `cfacshr`. Outputs `compustat_eps_panel.parquet`. Required by `13_table2.py` and `14_figures7_8.py`. |

## Main Scripts

| Script | Output | Paper Reference |
|--------|--------|-----------------|
| `11_figure1_table1.py` | Figure 1, Table I | Section II. Portfolio returns by LTG decile; descriptive statistics |
| `12_figure2.py` | Figure 2 | Section II. Evolution of EPS around portfolio formation |
| `12_figure3.py` | Figure 3 | Section II. Evolution of LTG around portfolio formation |
| `12_figure4.py` | Figure 4 | Section II. Forecast errors of LTG in predicting future EPS growth |
| `13_figure5.py` | Figure 5 | Section III. Earnings announcement returns by LTG decile |
| `13_table2.py` | Table II | Section III. Coibion-Gorodnichenko regressions for overreaction |
| `14_figures7_8.py` | Figures 7–8 | Section IV. Kernel densities of realized vs. expected EPS growth |
| `15_figure6.py` | Figure 6 | Section V. Simulated model moments (6-panel figure) |
| `15_table3.py` | Table III | Section V. SMM parameter estimates for the diagnostic Kalman filter |
| `16_table4.py` | Table IV | Section VI. Double sorts on LTG and earnings persistence/volatility |

## Data Dependencies

All scripts read from:
- `../../data/` — Raw WRDS data (CRSP, COMPUSTAT, IBES, Fama-French factors). Accessed via symlink at `../data/`.
- `../output/descriptive.parquet` — Core firm-year panel with LTG deciles, EPS leads/lags, returns (pre-built, no build script)
- `../output/crsp_monthly_filtered.parquet` — Monthly returns filtered to sample firms (pre-built, no build script)
- `../output/portfolio_returns.parquet`, `portfolio_assignments.parquet`, `ltg_all_months.parquet`, `daily_ret_filtered.parquet` — Additional pre-built datasets (see `../README.md` for details)

## Key Implementation Details

- **EPS measure**: COMPUSTAT `ib / ((shrout/1000) * cfacshr)`, matching `earnings.do` from the original code.
- **LTG timing (Table II)**: Analyst forecasts selected 30–90 days after annual earnings announcement (`ANNDATS` from IBES actuals).
- **SMM (Table III)**: Diagnostic expectation formula $DE_t = (1+\theta) \cdot RE_t - \theta \cdot a^s \cdot RE_{t-s}$, with grid search over $(a, b, \sigma_f, \sigma_e, \theta, s)$.
- **Industry fill (Table IV)**: Missing persistence/volatility filled with Fama-French 48 industry-year medians.
- **Winsorization**: EPS at 1/99 by year; regression variables at 5/95 by year (Table II); forecast errors at 1/99 by year (Figure 4).
