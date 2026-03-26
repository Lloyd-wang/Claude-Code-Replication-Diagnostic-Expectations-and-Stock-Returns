# Data

Raw datasets for the BGLS (2019) replication. Due to WRDS licensing restrictions, data files are not included in the repository.

## WRDS-Downloaded Data

Downloaded by `BGLS2019_rep/replication/code/00_download_data.py` using a WRDS account:

| File | Source | Rows | Description |
|------|--------|------|-------------|
| `comp_annual.parquet` | COMPUSTAT `funda` | 471K | Annual fundamentals: `ib`, `ni`, `sale`, `ceq`, `seq`, `sich`, etc. |
| `comp_quarterly.parquet` | COMPUSTAT `fundq` | 1.7M | Quarterly fundamentals: `niq`, `ibq`, `cshprq`, `saleq`, `rdq` |
| `crsp_monthly.parquet` | CRSP `msf`+`msenames`+`msedelist` | 3.9M | Monthly returns, prices, shares, exchange/SIC codes, delisting |
| `crsp_cfacshr.parquet` | Derived from CRSP `msf` | 3.9M | Cumulative share adjustment factor (`cfacshr`) |
| `crsp_stocknames.parquet` | CRSP `stocknames` | 83K | Stock identifiers, listing/delisting dates |
| `ccm_link.parquet` | CRSP-COMPUSTAT `ccmxpf_lnkhist` | 123K | `gvkey`-`permno` linking with dates |

## Manually Uploaded Data

These files were obtained separately (IBES requires special WRDS access; FF factors from Kenneth French's website):

| File | Source | Description |
|------|--------|-------------|
| `EPS_unadj_forecast.dta` | IBES `statsumu` | Analyst consensus forecasts (EPS and LTG) by firm-month |
| `EPS_unadj_act.dta` | IBES `actu` | Actual earnings and announcement dates (`ANNDATS`) |
| `iclink_updated.dta` | WRDS | IBES-CRSP identifier link (`TICKER`-`PERMNO`) |
| `daily_ret.parquet` | CRSP `dsf` | Daily stock returns (83M rows) for event studies |
| `FF_5_month.csv` | Kenneth French | Fama-French five-factor monthly returns |
| `daily_FF3.csv` | Kenneth French | Fama-French three-factor daily returns |

## Key Variables

- **IBES `FPI`**: Forecast period indicator. `0` = LTG, `1`/`2`/`3` = 1/2/3-year ahead EPS.
- **CRSP `shrout`**: Shares outstanding in thousands.
- **CRSP `cfacshr`**: Cumulative factor to adjust shares for stock splits.
- **COMPUSTAT `ib`**: Income before extraordinary items (millions).
- **COMPUSTAT `niq`**: Net income, quarterly (millions).
