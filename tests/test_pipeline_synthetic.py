from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tdcladder.build import build_all
from tdcladder.qa import TDC_PRIMARY_ANCHOR, eligible_tdc_rows, family_support_matrix


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
    assert "tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow_usd_billions" in outputs["quarterly_tdc_comparison_panel"].columns
    assert "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow_usd_billions" in outputs["quarterly_tdc_comparison_panel"].columns
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
    # The base fixture has only January and February; complete it for analysis.
    path = fixture_project / "data/raw/buycurve/monthly_issuance_maturity_panel.csv"
    monthly = pd.read_csv(path)
    march = monthly[monthly["month"].eq("2023-02-01")].assign(month="2023-03-01")
    pd.concat([monthly, march]).to_csv(path, index=False)
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


def test_missing_primary_anchor_does_not_admit_legacy_rows(fixture_project) -> None:
    (fixture_project / "data/raw/tdcest/tdc_tier2_regression_series.csv").unlink()
    quarterly = build_all(fixture_project)["quarterly_tdc_comparison_panel"]
    assert quarterly["tdc_tier2_interest_corrected_bank_only_ru_flow"].notna().any()
    assert not quarterly["is_tdc_aligned"].any()
    assert quarterly["is_future_or_partial"].all()


def test_tdc_unit_scaling_uses_usd_millions(fixture_project) -> None:
    quarterly = build_all(fixture_project)["quarterly_tdc_comparison_panel"]
    row = quarterly.iloc[0]
    assert row[TDC_PRIMARY_ANCHOR] == -49_700
    assert row[f"{TDC_PRIMARY_ANCHOR}_usd_millions"] == -49_700
    assert row[f"{TDC_PRIMARY_ANCHOR}_usd_billions"] == pytest.approx(-49.7)
    modern = "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow"
    assert row[f"{modern}_usd_billions"] == pytest.approx(-49.6)
    assert row["raw_treasury_supply_q_usd_billions"] == pytest.approx(
        row["raw_treasury_supply_q"] / 1_000_000_000
    )


@pytest.mark.parametrize("missing", ["source", "quarterly_panel", "complete_quarters"])
def test_missing_primary_anchor_blocks_tdc_release_outputs(fixture_project, monkeypatch, missing) -> None:
    if missing == "source":
        (fixture_project / "data/raw/tdcest/tdc_tier2_regression_series.csv").unlink()
    build_all(fixture_project)
    if missing == "quarterly_panel":
        (fixture_project / "data/clean/quarterly_tdc_comparison_panel.csv").unlink()
    analysis = _load_analysis_module()
    out_dir = fixture_project / "output/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Simulate all of the previous valid run's retained TDC artifacts.
    for name in analysis.TDC_DERIVED_FILES:
        (out_dir / name).write_text("stale endorsed result")
    monkeypatch.chdir(fixture_project)
    assert analysis.main() == 1
    assert all(not (out_dir / name).exists() for name in analysis.TDC_DERIVED_FILES)
    status = pd.read_csv(out_dir / "release_readiness_status.csv")
    assert set(status["status"]) == {"blocked_missing_primary_anchor"}
    manifest = pd.read_csv(out_dir / "output_tiers.csv")
    assert not manifest["claim_status"].isin(["ready", "promote"]).any()
    assert "stale endorsed result" not in (out_dir / "release_readiness_report.md").read_text()


def test_partial_quarter_is_excluded_from_all_tdc_outputs(fixture_project, monkeypatch) -> None:
    outputs = build_all(fixture_project)
    base = outputs["quarterly_tdc_comparison_panel"]
    base = base[base["weight_family"].eq("fixed_baseline")].iloc[[0]].copy()
    quarters = []
    for i, date in enumerate(pd.date_range("2023-03-31", periods=4, freq="QE")):
        q = base.copy()
        q["quarter"] = date
        q["is_tdc_aligned"] = True
        q["is_complete_quarter"] = i < 3
        q[TDC_PRIMARY_ANCHOR] = [-49_700, 13_000, -20_000, 999_999][i]
        q["raw_treasury_supply_q"] *= i + 1
        q["liquid_treasury_supply_q"] *= (i + 1) ** 2
        quarters.append(q)
    quarterly = pd.concat(quarters, ignore_index=True)
    analysis = _load_analysis_module()
    out_dir = fixture_project / "output/tables"
    panel_path = fixture_project / "data/clean/quarterly_tdc_comparison_panel.csv"
    monkeypatch.chdir(fixture_project)
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", None)
    expected = {}
    for panel in [quarterly.iloc[:3], quarterly]:
        panel.to_csv(panel_path, index=False)
        assert analysis.main() == 0
        current = {name: (out_dir / name).read_bytes() for name in analysis.TDC_DERIVED_FILES if (out_dir / name).exists()}
        if expected:
            assert current == expected
        expected = current
    summary = pd.read_csv(out_dir / "ladder_interpretation_summary.csv")
    assert summary["quarters_with_tdc"].eq(3).all()
    assert summary["last_quarter"].eq("2023-09-30").all()
    models = pd.read_csv(out_dir / "tdc_model_comparison_diagnostics.csv")
    assert models["n_obs"].eq(3).all()
    weighted = pd.DataFrame({
        "month": pd.date_range("2023-03-01", periods=4, freq="3MS"),
        "weight_family": "fixed_baseline", "supply_basis": "issuance_flow",
        "source_status": "observed", "treasury_supply_raw": [1, 2, 3, 999],
    })
    support = family_support_matrix(weighted, quarterly=quarterly)
    tdc = support[support["sample_window"].eq("tdc_sample")].iloc[0]
    assert tdc["rows"] == 3
    assert tdc["raw_supply"] == 6
    assert tdc["date_max"] == "2023-09-01"
    latest = support[support["sample_window"].eq("latest_quarter")].iloc[0]
    assert latest["date_max"] == "2023-09-01"


def test_tdc_eligibility_requires_both_flags_and_finite_anchor() -> None:
    q = pd.DataFrame({TDC_PRIMARY_ANCHOR: [1, 2, 3, float("inf"), 4],
                      "is_tdc_aligned": [True, False, True, True, True],
                      "is_complete_quarter": [True, True, False, True, None]})
    assert eligible_tdc_rows(q).index.tolist() == [0]
    assert eligible_tdc_rows(q.drop(columns="is_complete_quarter")).empty
