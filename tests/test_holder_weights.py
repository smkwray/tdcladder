from __future__ import annotations

import pandas as pd

from tdcladder.build.pipeline import build_raw_treasury_supply
from tdcladder.weights.holder import (
    build_holder_revealed_weights,
    build_holder_stock_context_weights,
)


def test_holder_revealed_weights_use_auction_mix(fixture_project) -> None:
    supply = build_raw_treasury_supply(fixture_project)
    auction = pd.read_csv(fixture_project / "data/raw/buycurve/auction_allotment_panel_base_slim.csv")
    weights = build_holder_revealed_weights(supply, auction_allotments=auction, root=fixture_project)
    assert weights["liquidity_weight"].between(0, 1).all()
    jan_bill = weights[(weights["month"].eq(pd.Timestamp("2023-01-01"))) & (weights["security_type"].eq("Bill")) & (weights["maturity_bucket"].eq("0_3m"))].iloc[0]
    jan_note = weights[(weights["month"].eq(pd.Timestamp("2023-01-01"))) & (weights["security_type"].eq("Note")) & (weights["maturity_bucket"].eq("3_5y"))].iloc[0]
    assert jan_bill["liquidity_weight"] > jan_note["liquidity_weight"]


def test_holder_stock_context_weights_merge_by_quarter(fixture_project) -> None:
    supply = pd.DataFrame(
        {
            "month": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "security_type": ["Bill", "Note"],
            "maturity_bucket": ["0_3m", "3_5y"],
        }
    )
    z1 = pd.DataFrame(
        {
            "quarter": ["2023-03-31", "2023-03-31", "2023-03-31"],
            "broad_investor_class": ["banks", "money_market_cash", "total"],
            "measure": ["level", "level", "level"],
            "value": [100.0, 100.0, 200.0],
        }
    )

    weights = build_holder_stock_context_weights(supply, z1_holder_context=z1, root=fixture_project)

    assert weights["source_status"].eq("diagnostic_z1_holder_stock_blend").all()
    bill = weights.loc[weights["security_type"].eq("Bill")].iloc[0]
    note = weights.loc[weights["security_type"].eq("Note")].iloc[0]
    assert bill["liquidity_weight"] < bill["base_fixed_weight"]
    assert note["liquidity_weight"] > note["base_fixed_weight"]
