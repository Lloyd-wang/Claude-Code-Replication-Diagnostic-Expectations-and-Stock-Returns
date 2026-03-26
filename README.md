# Replication: Diagnostic Expectations and Stock Returns (BGLS 2019)

This repository replicates the main empirical results (Tables I--IV, Figures 1--8) of:

> Bordalo, P., Gennaioli, N., La Porta, R., & Shleifer, A. (2019). **Diagnostic expectations and stock returns.** *Journal of Finance*, 74(6), 2839--2874. [https://doi.org/10.1111/jofi.12833](https://doi.org/10.1111/jofi.12833)

- Overall, all figures and results are successfully reproduced. Any minor discrepancies are likely attributable to differences in data processing procedures and data versions.

## About This Project

This replication was developed collaboratively with **[Claude Code](https://claude.ai/claude-code)** (Anthropic), serving as an experiment in AI-assisted replication of empirical finance research. All Python scripts --- from data downloading to figure generation --- were written through iterative human--AI dialogue, with the original authors' Stata code as the reference implementation.

**Disclaimer**: This repository is intended solely for **academic and educational purposes**. It is an independent replication effort and is not affiliated with or endorsed by the original authors. No proprietary data is distributed; users must obtain their own data access through WRDS and other sources listed below.

## Replication Summary

All 8 figures and 4 tables from the paper are successfully replicated, with patterns and magnitudes closely matching the published results. Minor numerical differences (e.g., kernel density peak heights, individual t-statistics) are attributable to data vintage differences between the authors' original extract and the current WRDS data.

## Replicated Results

| Output | Paper Reference | Description |
|--------|-----------------|-------------|
| Figure 1, Table I | Section II | Geometric mean returns by LTG decile; descriptive statistics |
| Figure 2 | Section II | Evolution of EPS around portfolio formation |
| Figure 3 | Section II | Evolution of analyst LTG forecasts around formation |
| Figure 4 | Section II | Forecast errors: realized EPS growth minus LTG |
| Figure 5 | Section III | Earnings announcement returns by LTG decile |
| Table II | Section III | LTG forecast revisions and errors |
| Figures 7--8 | Section IV | Kernel densities of realized vs. expected EPS growth |
| Figure 6 | Section V | Simulated diagnostic expectations model moments |
| Table III | Section V | SMM parameter estimates for the diagnostic Kalman filter |
| Table IV | Section VI | Double sorts on LTG and earnings persistence/volatility |

## Project Structure

All scripts assume `BGLS2019_rep/` as the working directory.

```
BGLS2019_rep/                  Repository root (run scripts from here)
├── README.md
├── LICENSE
├── data/                      Raw data files (gitignored; user must populate)
└── replication/
    ├── code/                  11 Python scripts
    │   ├── 00_download_data.py    WRDS download + EPS panel build
    │   ├── 11_figure1_table1.py
    │   ├── 12_figure2.py
    │   ├── 12_figure3.py
    │   ├── 12_figure4.py
    │   ├── 13_figure5.py
    │   ├── 13_table2.py
    │   ├── 14_figures7_8.py
    │   ├── 15_figure6.py
    │   ├── 15_table3.py
    │   └── 16_table4.py
    ├── output/                Generated figures (.png), tables (.txt), intermediate datasets (.parquet)
    └── original/              Authors' original Stata and Python code (JF replication package)
```

## Data Requirements

All data must be obtained independently by users with appropriate access. **No data files are included in this repository.**

### Downloaded via script (`00_download_data.py`)

Requires a [WRDS](https://wrds-www.wharton.upenn.edu/) account.

| Source | Dataset | Description |
|--------|---------|-------------|
| COMPUSTAT | `funda` | Annual fundamentals |
| COMPUSTAT | `fundq` | Quarterly fundamentals |
| CRSP | Monthly stock file | Returns, prices, shares outstanding |
| CRSP | `cfacshr` | Cumulative factor to adjust shares |
| CRSP | `stocknames` | PERMNO--ticker mapping |
| CCM | Linking table | CRSP--COMPUSTAT link |

### Manually obtained

| File | Source | Description |
|------|--------|-------------|
| `EPS_unadj_act.dta`, `EPS_unadj_forecast.dta` | IBES via WRDS | Unadjusted actual and forecast EPS |
| `iclink_updated.dta` | WRDS | IBES--CRSP identifier link |
| `FF_5_month.csv` | Kenneth French's website | Monthly Fama-French 5 factors |
| `daily_FF3.csv` | Kenneth French's website | Daily Fama-French 3 factors |
| `daily_ret.parquet` | CRSP via WRDS | Daily stock returns |

## How to Run

### 1. Set up environment

```bash
pip install pandas numpy scipy statsmodels wrds matplotlib
```

### 2. Download data

```bash
cd BGLS2019_rep/

# Set WRDS credentials
export WRDS_USER='your_wrds_username'
export PGPASSWORD='your_wrds_password'

# Download CRSP/COMPUSTAT from WRDS and build EPS panel
python replication/code/00_download_data.py

# Manually place IBES and other files in data/ (see table above)
```

### 3. Pre-built intermediate datasets

Six intermediate datasets (`descriptive.parquet`, `crsp_monthly_filtered.parquet`, `portfolio_returns.parquet`, `portfolio_assignments.parquet`, `ltg_all_months.parquet`, `daily_ret_filtered.parquet`) are required by most scripts but do not yet have standalone build scripts. These were constructed interactively during the replication process. See `replication/README.md` for details.

### 4. Generate figures and tables

Each numbered script can be run independently:

```bash
python replication/code/11_figure1_table1.py   # Figure 1, Table I
python replication/code/12_figure2.py           # Figure 2
python replication/code/12_figure3.py           # Figure 3
python replication/code/12_figure4.py           # Figure 4
python replication/code/13_figure5.py           # Figure 5
python replication/code/13_table2.py            # Table II
python replication/code/14_figures7_8.py        # Figures 7-8
python replication/code/15_figure6.py           # Figure 6
python replication/code/15_table3.py            # Table III (slow: SMM grid search)
python replication/code/16_table4.py            # Table IV
```

Outputs are written to `replication/output/`.

## Methodology Notes

- **EPS measure**: COMPUSTAT income before extraordinary items (`ib`) per share, split-adjusted via CRSP `cfacshr`, matching `earnings.do` from the original code.
- **LTG timing (Table II)**: Analyst forecasts selected 30--90 days after annual earnings announcements, following `revisions03.do`.
- **Rolling AR(1) (Table IV)**: 20-quarter rolling regressions with calendar-quarter alignment (`qofd(date)`) and duplicate averaging within (PERMNO, quarter), matching `Table04_A.do`. Missing values filled by Fama-French 48 industry-year medians.
- **SMM (Table III)**: Diagnostic expectations formula with grid search over $(a, b, \sigma_f, \sigma_e, \theta, s)$.
- **Kernel densities (Figures 7--8)**: Epanechnikov kernel with absolute bandwidth $h = 0.15$, matching Stata's `kdensity` defaults in `Kernels_JF.do`.
- **Cohort design (Figure 4)**: 3-year non-overlapping cohorts for forecast error computation, matching `JFsubmission.do`.

## Citation

If you use or reference this replication code, please cite the original paper:

```bibtex
@article{BGLS2019,
  title     = {Diagnostic Expectations and Stock Returns},
  author    = {Bordalo, Pedro and Gennaioli, Nicola and La Porta, Rafael and Shleifer, Andrei},
  journal   = {Journal of Finance},
  volume    = {74},
  number    = {6},
  pages     = {2839--2874},
  year      = {2019},
  doi       = {10.1111/jofi.12833}
}
```

## License

This replication code is released under the [MIT License](LICENSE). The original paper, data, and authors' replication code remain the intellectual property of their respective owners.
