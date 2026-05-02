from __future__ import annotations

import importlib.util
from pathlib import Path

from tdcladder.build import build_all


def _load_analysis_module():
    script = Path(__file__).resolve().parents[1] / "scripts/analysis/compare_liquidity_measures.py"
    spec = importlib.util.spec_from_file_location("compare_liquidity_measures", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_all_on_synthetic_fixture(fixture_project) -> None:
    outputs = build_all(fixture_project)
    assert outputs["raw_treasury_supply_by_maturity"].shape[0] == 6
    assert set(outputs["raw_treasury_supply_by_maturity"]["supply_basis"]) == {"issuance_flow"}
    assert "raw_weighted_maturity_years_q" in outputs["quarterly_tdc_comparison_panel"].columns
    assert "raw_treasury_supply_q_usd_billions" in outputs["quarterly_tdc_comparison_panel"].columns
    assert "tdc_tier2_interest_corrected_bank_only_ru_flow_usd_billions" in outputs["quarterly_tdc_comparison_panel"].columns
    assert "price_first_pass" not in set(outputs["liquidity_weights"]["weight_family"])
    assert set(outputs["liquidity_weights"]["weight_family"]).issuperset({
        "fixed_baseline",
        "price_convenience_yield",
        "turnover_duration_adjusted",
        "holder_revealed_first_pass",
    })
    assert not outputs["monthly_ladder_liqsub_panel"].empty
    assert not outputs["quarterly_tdc_comparison_panel"].empty
    assert (fixture_project / "output/qa/weight_bounds.csv").exists()
    assert (fixture_project / "output/qa/weight_source_status.csv").exists()
    assert (fixture_project / "output/qa/family_support_matrix.csv").exists()


def test_analysis_outputs_claim_hierarchy_and_episode_scaffolds(fixture_project) -> None:
    outputs = build_all(fixture_project)
    analysis = _load_analysis_module()
    out_dir = fixture_project / "output/tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis.write_ladder_interpretation_table(outputs["quarterly_tdc_comparison_panel"], out_dir)
    analysis.write_monthly_episode_candidates(outputs["monthly_ladder_liqsub_panel"], out_dir)
    analysis.write_quarterly_episode_design(outputs["quarterly_tdc_comparison_panel"], out_dir)

    assert (out_dir / "core_measurement_summary.csv").exists()
    assert (out_dir / "exploratory_sidecar_summary.csv").exists()
    assert (out_dir / "monthly_episode_design.csv").exists()
    assert (out_dir / "quarterly_tier2_episode_design.csv").exists()
    assert (out_dir / "monthly_episode_event_catalog.csv").exists()


def test_build_all_keeps_optional_stock_basis_parallel(fixture_project) -> None:
    stock_dir = fixture_project / "data/raw/treasury"
    stock_dir.mkdir(parents=True)
    (stock_dir / "monthly_outstanding_by_maturity.csv").write_text(
        "\n".join(
            [
                "month,security_type,maturity_bucket,outstanding_amount,weighted_maturity_years",
                "2023-01-01,Bill,0_3m,400000000000,0.12",
                "2023-01-01,Note,3_5y,900000000000,3.50",
                "2023-02-01,Bill,0_3m,420000000000,0.12",
                "2023-02-01,Bond,20_30y,800000000000,22.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = build_all(fixture_project)
    supply = outputs["raw_treasury_supply_by_maturity"]
    monthly = outputs["monthly_ladder_panel"]
    quarterly = outputs["quarterly_tdc_comparison_panel"]

    assert set(supply["supply_basis"]) == {"issuance_flow", "outstanding_stock"}
    assert supply[supply["supply_basis"].eq("outstanding_stock")]["outstanding_amount"].notna().all()
    assert set(monthly["supply_basis"]) == {"issuance_flow", "outstanding_stock"}
    assert set(quarterly["supply_basis"]) == {"issuance_flow", "outstanding_stock"}
    stock_month = monthly[
        monthly["supply_basis"].eq("outstanding_stock")
        & monthly["weight_family"].eq("fixed_baseline")
        & monthly["month"].eq("2023-02-01")
    ].iloc[0]
    flow_month = monthly[
        monthly["supply_basis"].eq("issuance_flow")
        & monthly["weight_family"].eq("fixed_baseline")
        & monthly["month"].eq("2023-02-01")
    ].iloc[0]
    assert stock_month["raw_bill_share"] != flow_month["raw_bill_share"]
    stock_q = quarterly[
        quarterly["supply_basis"].eq("outstanding_stock")
        & quarterly["weight_family"].eq("fixed_baseline")
    ].iloc[0]
    assert stock_q["supply_aggregation"] == "quarter_end"
    assert stock_q["raw_treasury_supply_q"] == 1_220_000_000_000
