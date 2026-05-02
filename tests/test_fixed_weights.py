from __future__ import annotations

import pandas as pd

from tdcladder.weights.fixed import fixed_weight_for, fixed_weights_for_frame


def test_baseline_weights_rank_duration(fixture_project) -> None:
    bill = fixed_weight_for("0_3m", "Bill", root=fixture_project)
    short_note = fixed_weight_for("1_2y", "Note", root=fixture_project)
    long_bond = fixed_weight_for("20_30y", "Bond", root=fixture_project)
    assert 0 <= long_bond < short_note < bill <= 1


def test_alias_bucket_is_supported(fixture_project) -> None:
    assert fixed_weight_for("2_5y", "Note", root=fixture_project) == fixed_weight_for("3_5y", "Note", root=fixture_project)


def test_batch_fixed_weights_match_scalar_api(fixture_project) -> None:
    frame = pd.DataFrame(
        {
            "maturity_bucket": ["0_3m", "2_5y", "20_30y"],
            "security_type": ["Bill", "Note", "Bond"],
        }
    )
    batch = fixed_weights_for_frame(frame, root=fixture_project, ladder_name="fixed_baseline").tolist()
    scalar = [
        fixed_weight_for(row.maturity_bucket, row.security_type, root=fixture_project, ladder_name="fixed_baseline")
        for row in frame.itertuples(index=False)
    ]
    assert batch == scalar
