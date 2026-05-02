from __future__ import annotations

from tdcladder.cli import main


def test_cli_describe_contracts(fixture_project, capsys) -> None:
    code = main(["describe-contracts", "--root", str(fixture_project), "--upstream", "buycurve"])
    captured = capsys.readouterr()
    assert code == 0
    assert "monthly_issuance_maturity_panel" in captured.out
