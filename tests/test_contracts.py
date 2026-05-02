from __future__ import annotations

from tdcladder.contracts import validate_all


def test_contracts_validate_fixture_project(fixture_project) -> None:
    manifest = validate_all(fixture_project)
    core = manifest[manifest["name"].eq("buycurve_monthly_issuance_maturity_panel")].iloc[0]
    assert core["status"] == "pass"
    assert core["rows"] == 6
    assert core["duplicate_key_count"] == 0

    allotments = manifest[manifest["name"].eq("buycurve_auction_allotment_panel_base_slim")].iloc[0]
    assert allotments["date_min"] == "2023-01-05"
    assert allotments["date_max"] == "2023-02-15"
