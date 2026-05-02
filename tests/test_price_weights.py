from __future__ import annotations

import pandas as pd

from tdcladder.weights.price import build_price_convenience_yield_weights


def test_convenience_yield_boosts_short_bills_more_than_long_bonds(fixture_project) -> None:
    supply = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "security_type": ["Bill", "Bond"],
            "maturity_bucket": ["0_3m", "20_30y"],
        }
    )
    liqsub = pd.DataFrame(
        {
            "month": ["2024-01-01"],
            "iorb_rate": [5.0],
            "bill_yield_3mo": [4.7],
            "bill_iorb_spread": [-0.30],
            "sofr_iorb_spread": [0.00],
        }
    )

    weights = build_price_convenience_yield_weights(supply, liqsub=liqsub, root=fixture_project)

    bill = weights.loc[weights["security_type"].eq("Bill")].iloc[0]
    bond = weights.loc[weights["security_type"].eq("Bond")].iloc[0]
    assert bill["source_status"] == "first_pass_convenience_yield_model"
    assert bill["liquidity_weight"] >= bill["base_fixed_weight"]
    assert bond["liquidity_weight"] <= 0.22


def test_convenience_yield_falls_back_without_rate_inputs(fixture_project) -> None:
    supply = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01"]),
            "security_type": ["Bill"],
            "maturity_bucket": ["0_3m"],
        }
    )
    weights = build_price_convenience_yield_weights(supply, liqsub=pd.DataFrame(), root=fixture_project)
    row = weights.iloc[0]
    assert row["source_status"] == "fallback_fixed_missing_convenience_yield_inputs"
    assert row["liquidity_weight"] == row["base_fixed_weight"]
