from __future__ import annotations

import pandas as pd
import pytest

from tdcladder.treasury_stock import build_monthly_outstanding_by_maturity


def test_build_monthly_outstanding_by_maturity_excludes_total_rows(fixture_project) -> None:
    raw_dir = fixture_project / "data/raw/treasury"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "mspd_table_3_market.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2026-02-28",
                "security_type_desc": "Marketable",
                "security_class1_desc": "Bills Maturity Value",
                "security_class2_desc": "912797ST5",
                "maturity_date": "2026-03-03",
                "outstanding_amt": "254840.9417",
            },
            {
                "record_date": "2026-02-28",
                "security_type_desc": "Marketable",
                "security_class1_desc": "Bills Maturity Value",
                "security_class2_desc": "Total Treasury Bills",
                "maturity_date": "",
                "outstanding_amt": "6808105.8252",
            },
            {
                "record_date": "2026-02-28",
                "security_type_desc": "Marketable",
                "security_class1_desc": "Notes",
                "security_class2_desc": "912828Z78",
                "maturity_date": "2027-01-31",
                "outstanding_amt": "97787.8406",
            },
        ]
    ).to_csv(raw_path, index=False)

    out = build_monthly_outstanding_by_maturity(fixture_project)

    assert set(out["security_type"]) == {"Bill", "Note"}
    assert set(out["maturity_bucket"]) == {"0_3m", "6_12m"}
    assert out["outstanding_amount"].sum() == pytest.approx((254840.9417 + 97787.8406) * 1_000_000)
    assert (raw_dir / "monthly_outstanding_by_maturity.csv").exists()
