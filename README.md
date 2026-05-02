# tdcladder

`tdcladder` builds a Liquidity-Weighted TDC / Treasury Moneyness Ladder: a companion measure to the Tier 2 interest-corrected bank-only TDC anchor that asks whether the same Treasury finance flow has different monetary meaning depending on whether investors receive bills, FRNs, short coupons, or long-duration coupons.

The headline object is not a replacement for deposit-accounting TDC. Ordinary TDC tracks the Treasury-attributed component of deposits; `tdcladder` adds a measurement layer:

\[
LTS_t = \sum_m w_{m,t} \cdot TS_{m,t}
\]

where `TS` is Treasury supply by maturity/security bucket and `w` is an explicit moneyness/liquidity assumption. The weights are labeled measurement assumptions, not structural facts.

## Project Metadata

- Package: `tdcladder`
- Current release status: research toolkit, not a production data service
- Primary measurement layer: liquidity-weighted Treasury supply by maturity/security bucket
- Primary TDC anchor: Tier 2 interest-corrected bank-only RU flow
- Default Treasury basis: gross accepted auction issuance flow
- Optional Treasury basis: month-end outstanding stock from FiscalData/MSPD-style inputs
- Data policy: raw upstream extracts and generated outputs are local artifacts and are ignored by git
- Claim boundary: descriptive measurement and episode-design surfaces only; no structural moneyness or causal claims are implied by the weights

## Repository layout

```text
config/                 source contracts, maturity buckets, weight assumptions
config/source_contracts/ YAML contracts for upstream exports
src/tdcladder/          package code and CLI
data/raw/               explicit local upstream copies; ignored by git
data/interim/           normalized source-level files; ignored by git
data/clean/             analysis-ready panels; ignored by git
output/                 QA, manifests, tables, figures; ignored by git
scripts/analysis/       starter analysis scripts
tests/                  synthetic fixtures and focused tests
```

Local planning notes, raw data, and generated outputs are intentionally excluded from version control. The public repository should contain code, contracts, configuration, tests, and lightweight placeholders only.

## Install

```bash
python -B -m pip install -e '.[dev,analysis]'
python -B -m pytest
```

## Supply upstream files locally

Production code never hard-codes machine-specific upstream paths. Copy the upstream exports into `data/raw/<upstream>/` either manually or through the import gate.

```bash
tdcladder copy-upstream --upstream buycurve --source-root ../buycurve
tdcladder copy-upstream --upstream liqsub   --source-root ../liqsub
tdcladder copy-upstream --upstream tdcest   --source-root ../tdcest
tdcladder copy-upstream --upstream tdcpass  --source-root ../tdcpass

tdcladder validate-sources
```

Expected core local copies are documented in `data/raw/README.md` and `config/source_contracts/*.yaml`.
When present, buycurve's normalized Z.1 holder panel is copied to
`data/raw/buycurve/z1_treasury_holders_clean.csv` and used only as aggregate
holder-stock context.

The default flow basis is `issuance_flow`, using buycurve gross accepted auction
amounts by month and maturity bucket. If a project-local
`data/raw/treasury/monthly_outstanding_by_maturity.csv` file is supplied, the
same build also emits an `outstanding_stock` basis in parallel. The stock path is
intended for MSPD/FiscalData-style month-end outstanding amounts and is a
sensitivity surface, not a replacement for the flow basis.

To refresh the optional stock-basis input directly from FiscalData:

```bash
tdcladder download-treasury-stock
```

To build the optional turnover input from FINRA public TRACE Treasury aggregate
workbooks:

```bash
tdcladder build-trace-turnover --source-path <path-to-trace-workbooks>
```

## Build the ladder

```bash
tdcladder build
```

Main outputs:

```text
data/interim/raw_treasury_supply_by_maturity.csv
data/clean/liquidity_weights.csv
data/clean/liquidity_weighted_treasury_supply_by_bucket.csv
data/clean/monthly_ladder_panel.csv
data/clean/monthly_ladder_liqsub_panel.csv
data/clean/quarterly_tdc_comparison_panel.csv
output/qa/source_manifest.csv
output/qa/dataset_summary.csv
output/qa/missingness.csv
output/qa/duplicate_keys.csv
output/qa/weight_bounds.csv
output/qa/weight_source_status.csv
output/qa/family_support_matrix.csv
output/qa/unit_checks.csv
output/qa/analysis_unit_assumptions.csv
output/qa/qa_report.md
```

The monthly panel is the primary frequency. Quarterly aggregation is used for TDC comparison and Z.1-aligned interpretation.

## Weight families

The current build includes seven active families. `price_first_pass` remains in
code/config as an inactive continuity path; public analysis outputs use the more
explicit `price_convenience_yield` family when price sensitivity is needed.

| Family | Role |
|---|---|
| `fixed_baseline` | Transparent maturity/security-type ladder for exposition. |
| `fixed_conservative_duration` | Sensitivity that penalizes duration more aggressively. |
| `fixed_cash_like_generous` | Sensitivity that treats short Treasury instruments more generously. |
| `price_convenience_yield` | Explicit convenience-yield sensitivity using bill-IORB specialness and SOFR-IORB funding-pressure penalty. |
| `turnover_duration_adjusted` | Stub/first-pass turnover family that never lets high long-bond trading volume erase duration risk. |
| `holder_revealed_first_pass` | First-pass buyer-composition weight using auction allotments, blended with the fixed ladder to retain duration discipline. |
| `holder_stock_context_z1` | Diagnostic quarterly Z.1 holder-stock context blended with the fixed ladder; aggregate holders do not identify security-level moneyness. |

## Starter analysis

After `tdcladder build`:

```bash
python scripts/analysis/compare_liquidity_measures.py
python scripts/analysis/sensitivity_weight_ladders.py
```

These scripts create descriptive correlations/tables and simple figures comparing raw Treasury supply, liquid Treasury supply, Tier 2 interest-corrected bank-only TDC, deposits, MMFs, ON RRP, and reserves. Treasury supply columns are stored in dollars and now get explicit `_usd` / `_usd_billions` display columns; TDC, tdcpass, and liqsub/H.8/OFR-style macro columns are treated as stored in USD millions and get explicit `_usd_millions` / `_usd_billions` columns. `output/qa/analysis_unit_assumptions.csv` documents those scale rules.

`output/tables/ladder_interpretation_summary.csv` is the broad summary of raw supply, liquidity-weighted supply, basis-specific bill share, basis-specific WAM, and the Tier 2 TDC anchor. `output/tables/core_measurement_summary.csv` is the narrower headline table: issuance-flow only, fixed baseline plus fixed sensitivity band. `output/tables/exploratory_sidecar_summary.csv` keeps price, turnover, auction-buyer, and Z.1 holder-context families separate as diagnostics. `output/tables/tdc_model_comparison_diagnostics.csv` adds standardized descriptive fit diagnostics for raw supply, liquid supply, bill share, and WAM model variants. Treat all of these as measurement diagnostics, not causal claims.
`output/tables/latest_results_overview.csv` and `output/tables/latest_results_overview.md` provide a short plain-language snapshot for nontechnical review. `output/tables/monthly_episode_design.csv` and `output/tables/quarterly_tier2_episode_design.csv` are first-pass episode scaffolds for bill-heavy, coupon-heavy, WAM-extension, duration-gap, reserve-drain, MMF-absorption, ON-RRP-drain, TGA-refill, high-rate, and QT regimes. `output/tables/monthly_episode_event_windows.csv` and `output/tables/quarterly_tier2_episode_event_windows.csv` convert those flags into descriptive event windows; their paired `_summary.csv` and `_catalog.csv` files summarize average paths and event counts. These event-window outputs are descriptive triage surfaces, not causal local projections.

For output navigation, start with `output/tables/release_readiness_report.md` and `output/tables/output_tiers.csv`. They mark which outputs are headline, diagnostic, sidecar, appendix, QA, or episode-design surfaces.

## CLI reference

```bash
tdcladder validate-sources [--required-only]
tdcladder copy-upstream --upstream buycurve --source-root ../buycurve [--overwrite]
tdcladder download-treasury-stock
tdcladder build-trace-turnover --source-path <path-to-trace-workbooks>
tdcladder build [--strict] [--skip-qa]
tdcladder describe-contracts
```

`--strict` makes missing optional comparison inputs fatal. Without it, the build proceeds with available local files and records missingness in QA outputs.

## Interpretation guardrails

- Liquidity weights are assumptions and sensitivity surfaces, not structural constants.
- Bills, FRNs, and short coupons can be money-like for institutional cash investors, but they are not checking deposits.
- High trading volume in a long bond is market liquidity, not the same as money-likeness; duration risk remains central.
- Weekly `liqsub` outputs are intentionally excluded from the headline build unless explicit claim-readiness gates are added.
