# Output

Generated figures, tables, and intermediate datasets.

**Note**: Figures and tables are fully reproducible from the scripts in `../code/`. However, 6 of the 7 intermediate datasets (all except `compustat_eps_panel.parquet`) were built interactively and do not yet have standalone build scripts — see `../README.md` for details.

## Figures

| File | Description |
|------|-------------|
| `Figure1.png` | Geometric mean annual EW returns by LTG decile (1981–2015) |
| `Figure2.png` | Evolution of EPS around portfolio formation for HLTG and LLTG |
| `Figure3.png` | Evolution of LTG (analyst growth forecasts) around formation |
| `Figure4.png` | Forecast errors: realized EPS growth minus LTG |
| `Figure5.png` | Earnings announcement returns by LTG decile |
| `Figure6.png` | Simulated model moments (6-panel: autocorrelations, CG coefficients, returns) |
| `Figure7.png` | Kernel density of 5-year realized gross EPS growth (HLTG vs. non-HLTG) |
| `Figure8.png` | Kernel density of realized vs. expected EPS growth for HLTG |

## Tables

| File | Description |
|------|-------------|
| `Table1.txt` | Characteristics of LTG decile portfolios |
| `Table2.txt` | Coibion-Gorodnichenko overreaction regressions |
| `Table3.txt` | SMM parameter estimates for the diagnostic Kalman filter model |
| `Table4.txt` | Double-sorted returns on LTG and earnings persistence/volatility |

## Intermediate Datasets

| File | Description |
|------|-------------|
| `descriptive.parquet` | Core firm-year panel: LTG deciles, EPS leads/lags, returns, characteristics |
| `compustat_eps_panel.parquet` | COMPUSTAT `ib`-based per-share EPS, split-adjusted (PERMNO × year) |
| `crsp_monthly_filtered.parquet` | Monthly returns filtered to sample firms and period |
| `daily_ret_filtered.parquet` | Daily returns for earnings announcement event study |
| `portfolio_assignments.parquet` | LTG decile assignments by firm-year |
| `portfolio_returns.parquet` | Monthly portfolio returns by LTG decile |
| `ltg_all_months.parquet` | LTG forecasts at all available monthly dates |
