# tdcladder

tdcladder builds monthly and quarterly panels of liquidity-weighted US Treasury supply by maturity bucket and security type, and compares them with the Treasury Deposit Component (TDC), the part of the change in deposits at US banks and credit unions that comes from Treasury payments and trades in Treasury securities, as measured by [tdcest](https://github.com/smkwray/tdcest). It reads gross accepted Treasury auction issuance (plus optional month-end outstanding stocks and turnover data), applies seven liquidity-weight schemes across 11 maturity buckets from 0–3 months to 30+ years, and writes clean CSV panels, data-quality reports, and descriptive comparison tables.

For each period `t`, liquidity-weighted Treasury supply is:

```text
LTS_t = sum_m w_{m,t} * TS_{m,t}
```

where `TS_{m,t}` is Treasury supply in maturity and security bucket `m` (bills, floating-rate notes, nominal coupons, and Treasury Inflation-Protected Securities (TIPS) remain separate) and `w_{m,t}` is the liquidity weight for that bucket.

## Data

Input definitions live under `config/source_contracts/` and expect local CSV files under `data/raw/<upstream>/`. Run `tdcladder describe-contracts` to list every configured path.

- Required: monthly gross accepted auction issuance by maturity bucket from [buycurve](https://github.com/smkwray/buycurve) (`data/raw/buycurve/monthly_issuance_maturity_panel.csv`).
- Optional Treasury stock: month-end marketable debt outstanding (`data/raw/treasury/monthly_outstanding_by_maturity.csv`), downloaded directly from the US Treasury Fiscal Data API (Monthly Statement of the Public Debt, Table 3 Marketable) via `tdcladder download-treasury-stock`. When present, `tdcladder build` produces both outstanding-stock and issuance-flow series.
- Optional turnover: secondary-market volume by security type (`data/raw/market_liquidity/turnover_by_maturity.csv`), built from the public Treasury aggregate workbooks (`.xlsx`/`.xls`/`.zip`) of the Financial Industry Regulatory Authority's Trade Reporting and Compliance Engine (FINRA TRACE) via `tdcladder build-trace-turnover`.
- Optional comparison exports under `data/raw/`: CSVs from sibling repositories for auction allotments and Financial Accounts of the United States (Z.1) Treasury holders (`buycurve`), monthly deposits, money market funds (MMFs), overnight reverse repo (ON RRP), reserves, and money-market rate spreads ([liqsub](https://github.com/smkwray/liqsub)), quarterly measured TDC and components ([tdcest](https://github.com/smkwray/tdcest)), and quarterly deposit pass-through series ([tdcpass](https://github.com/smkwray/tdcpass)).

A real build therefore needs a `buycurve` checkout; the test suite runs on synthetic fixtures without any sibling repository. No API keys are required. Raw extracts and generated outputs are gitignored.

## Install

Requires Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -B -m pip install -e '.[dev,analysis]'
```

## Usage

Inspect configured inputs and copy local sibling exports into `data/raw/<upstream>/`:

```bash
tdcladder describe-contracts
tdcladder copy-upstream --upstream buycurve --source-root ../buycurve
tdcladder copy-upstream --upstream liqsub   --source-root ../liqsub
tdcladder copy-upstream --upstream tdcest   --source-root ../tdcest
tdcladder copy-upstream --upstream tdcpass  --source-root ../tdcpass
tdcladder validate-sources
```

`copy-upstream` accepts `--overwrite`; `validate-sources` accepts `--required-only` and `--upstream`.

Optionally fetch Treasury stock data from Fiscal Data or parse FINRA TRACE workbooks:

```bash
tdcladder download-treasury-stock
tdcladder build-trace-turnover --source-path PATH_TO_TRACE_WORKBOOKS
```

Pass `--from-raw` to `download-treasury-stock` to rebuild from an existing `data/raw/treasury/mspd_table_3_market.csv` without downloading.

Build the monthly supply panel, joined monthly panel, quarterly TDC comparison panel, and data-quality reports:

```bash
tdcladder build
```

Pass `--strict` to fail if optional comparison inputs are absent, or `--skip-qa` to omit quality reports.

Run the descriptive comparison and weight-sensitivity scripts after `tdcladder build`:

```bash
python scripts/analysis/compare_liquidity_measures.py
python scripts/analysis/sensitivity_weight_ladders.py
```

Without the `tdcest` export, `compare_liquidity_measures.py` writes the monthly tables, skips the TDC tables, and exits nonzero.

`config/weight_families.yaml` activates seven liquidity-weight schemes: three fixed maturity ladders (a baseline ladder, a steeper duration penalty, and higher weights on short maturities) plus four data-driven schemes blending the baseline ladder with money-market convenience yields (the spreads of bill yields and the Secured Overnight Financing Rate over the interest rate on reserve balances), FINRA TRACE turnover with a duration penalty, auction buyer composition, and quarterly Z.1 holder stocks.

## Output

`tdcladder build` writes clean monthly and quarterly CSV panels under `data/clean/` and quality checks under `output/qa/`. The analysis scripts write comparison tables and charts under `output/tables/` and `output/figures/`.

Main files:

| File | Description |
|---|---|
| `data/clean/monthly_ladder_panel.csv` | Monthly raw Treasury supply, liquidity-weighted supply, bill share, and weighted-average maturity (WAM), in US dollars and years |
| `data/clean/quarterly_tdc_comparison_panel.csv` | Quarterly Treasury supply aligned with quarterly measured TDC (supply in US dollars, TDC in USD millions) |
| `data/clean/liquidity_weights.csv` | Monthly liquidity weights by weighting scheme, maturity bucket, and security type |
| `output/qa/qa_report.md` | Summary of missingness, duplicate keys, weight bounds, and unit checks |
| `output/tables/latest_results_overview.md` | Summary of descriptive comparisons and weight-sensitivity checks |

Bucket-level weighted supply, joined monthly deposit and money-market panels, individual quality CSVs, correlation and event-window tables, and raw-versus-liquid supply charts are written alongside them. Analysis tables scale Treasury supply (stored in US dollars) and macro balance-sheet series (stored in USD millions) to USD billions.

## Limits

- Liquidity weights are assumptions; the seven schemes show how results change across them. Short Treasuries serve as near-money for institutional cash investors while remaining distinct from bank deposits.
- High secondary-market turnover in long bonds reflects trading volume under duration risk, so the turnover-based weighting scheme keeps a duration penalty.
- Quarterly TDC comparisons use complete quarters aligned with quarterly measured TDC (in USD millions) from the optional `tdcest` export.
- Correlations, standardized model diagnostics, episode flags, and event windows written by `scripts/analysis/` are descriptive and do not identify causal effects.

## Tests

The test suite runs offline on synthetic fixtures:

```bash
python -B -m pytest -q
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
