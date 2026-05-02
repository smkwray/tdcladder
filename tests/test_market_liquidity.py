from __future__ import annotations

import pandas as pd

from tdcladder.weights.turnover import build_turnover_weights


def test_turnover_weights_accept_security_type_turnover(fixture_project) -> None:
    supply = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "security_type": ["Bill", "Note"],
            "maturity_bucket": ["0_3m", "7_10y"],
        }
    )
    turnover = pd.DataFrame(
        {
            "month": ["2024-01-01", "2024-01-01"],
            "security_type": ["Bill", "Coupon"],
            "turnover": [2.0, 1.0],
        }
    )

    weights = build_turnover_weights(supply, turnover=turnover, root=fixture_project)

    bill_weight = weights.loc[weights["security_type"].eq("Bill"), "liquidity_weight"].iloc[0]
    note_weight = weights.loc[weights["security_type"].eq("Note"), "liquidity_weight"].iloc[0]
    assert weights["source_status"].eq("first_pass_turnover_duration_blend").all()
    assert bill_weight > note_weight


def test_turnover_weights_fall_back_when_month_unmatched(fixture_project) -> None:
    supply = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-02-01"]),
            "security_type": ["Bill"],
            "maturity_bucket": ["0_3m"],
        }
    )
    turnover = pd.DataFrame(
        {
            "month": ["2024-01-01"],
            "security_type": ["Bill"],
            "turnover": [2.0],
        }
    )

    weights = build_turnover_weights(supply, turnover=turnover, root=fixture_project)

    row = weights.iloc[0]
    assert row["source_status"] == "fallback_fixed_no_matching_turnover"
    assert row["liquidity_weight"] == row["base_fixed_weight"]
