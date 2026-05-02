# `data/raw/`

Place explicit local copies of upstream exports here. Production code reads from these project-local paths only; it does not reach into machine-specific upstream locations.

Expected core files:

```text
data/raw/buycurve/monthly_issuance_maturity_panel.csv
data/raw/buycurve/auction_allotment_panel_base_slim.csv
data/raw/buycurve/z1_treasury_holders_clean.csv
data/raw/liqsub/monthly_liquidity_substitution_panel.csv
data/raw/liqsub/monthly_liquidity_substitution_panel_analysis.csv
data/raw/tdcest/tdc_estimates.csv
data/raw/tdcest/tdc_components.csv
data/raw/tdcpass/quarterly_panel.csv
```

Optional stock-basis sensitivity input:

```text
data/raw/treasury/monthly_outstanding_by_maturity.csv
data/raw/market_liquidity/turnover_by_maturity.csv
```

The buycurve Z.1 holder file is optional for source validation but feeds the
`holder_stock_context_z1` diagnostic weight family when present. It is aggregate
quarterly holder-stock context, not maturity-specific final-holder evidence.

When present, the build publishes `outstanding_stock` rows alongside the default
`issuance_flow` rows. This optional file should be a project-local copy of an
MSPD/FiscalData-style month-end outstanding panel with `month`, `security_type`,
`maturity_bucket`, and `outstanding_amount`.

Use `tdcladder download-treasury-stock` to download FiscalData MSPD Table 3
marketable detail to `data/raw/treasury/mspd_table_3_market.csv` and normalize
it into the optional monthly outstanding-stock input.

Use `tdcladder build-trace-turnover --source-path <path-to-trace-workbooks>` to
build the optional market-liquidity turnover input from FINRA public TRACE
Treasury aggregate workbooks. TRACE aggregate categories are broad security
types, not maturity buckets, so the turnover weight family still applies
duration discipline from the fixed ladder.

Use `tdcladder copy-upstream --upstream <name> --source-root <path-to-that-upstream-repo>` to copy files from an explicit source root, or copy them manually and run `tdcladder validate-sources`.
