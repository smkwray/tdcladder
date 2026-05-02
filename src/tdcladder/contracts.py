from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_yaml, project_root
from .hashing import sha256_file
from .io import write_csv

CONTRACT_DIR = "config/source_contracts"


def load_contracts(
    root: str | Path | None = None,
    *,
    upstream: str | None = None,
    include_optional: bool = True,
) -> list[dict[str, Any]]:
    base = project_root(root) / CONTRACT_DIR
    contracts: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.yaml")):
        contract = load_yaml(path)
        contract["_contract_path"] = str(path.relative_to(project_root(root)))
        if upstream and contract.get("upstream") != upstream:
            continue
        if not include_optional and not contract.get("required", False):
            continue
        contracts.append(contract)
    return contracts


def get_contract(root: str | Path | None, name: str) -> dict[str, Any]:
    matches = [c for c in load_contracts(root) if c.get("name") == name]
    if not matches:
        raise KeyError(f"No source contract named {name!r}")
    return matches[0]


def contract_local_path(root: str | Path | None, contract: dict[str, Any]) -> Path:
    return project_root(root) / str(contract["local_path"])


def _csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def validate_contract(root: str | Path | None, contract: dict[str, Any], *, sample_rows: int | None = None) -> dict[str, Any]:
    p = contract_local_path(root, contract)
    result: dict[str, Any] = {
        "name": contract.get("name"),
        "upstream": contract.get("upstream"),
        "required": bool(contract.get("required", False)),
        "required_stage": contract.get("required_stage", ""),
        "local_path": contract.get("local_path"),
        "contract_path": contract.get("_contract_path", ""),
        "exists": p.exists(),
        "status": "missing",
        "rows": None,
        "columns": None,
        "sha256": None,
        "bytes": None,
        "missing_required_columns": "",
        "duplicate_key_count": None,
        "date_min": "",
        "date_max": "",
    }
    if not p.exists():
        result["status"] = "missing_required" if contract.get("required") else "missing_optional"
        return result

    result["sha256"] = sha256_file(p)
    result["bytes"] = p.stat().st_size
    try:
        columns = _csv_columns(p)
    except Exception as exc:  # pragma: no cover - defensive branch
        result["status"] = "unreadable"
        result["error"] = str(exc)
        return result

    result["columns"] = len(columns)
    required_columns = list(contract.get("required_columns", []))
    missing = [col for col in required_columns if col not in columns]
    result["missing_required_columns"] = ",".join(missing)
    if missing:
        result["status"] = "failed_missing_columns"
        return result

    try:
        read_kwargs: dict[str, Any] = {}
        if sample_rows is not None:
            read_kwargs["nrows"] = sample_rows
        df = pd.read_csv(p, low_memory=False, **read_kwargs)
    except Exception as exc:  # pragma: no cover - defensive branch
        result["status"] = "unreadable"
        result["error"] = str(exc)
        return result

    result["rows"] = int(len(df))
    key_cols = [col for col in contract.get("primary_key", []) if col in df.columns]
    if key_cols:
        result["duplicate_key_count"] = int(df.duplicated(key_cols).sum())
    else:
        result["duplicate_key_count"] = 0

    date_values = []
    for col in contract.get("date_columns", []):
        if col not in df.columns:
            continue
        parsed = _parse_contract_date_series(df[col], contract.get("frequency"))
        if parsed.notna().any():
            date_values.append(parsed)
    if date_values:
        combined = pd.concat(date_values)
        result["date_min"] = str(combined.min().date())
        result["date_max"] = str(combined.max().date())

    allow_dupes = bool(contract.get("qa", {}).get("allow_duplicate_primary_key", False))
    if result["duplicate_key_count"] and not allow_dupes:
        result["status"] = "failed_duplicate_keys"
    else:
        result["status"] = "pass"
    return result


def _parse_contract_date_series(series: pd.Series, frequency: str | None) -> pd.Series:
    s = series.astype("string")
    if frequency == "quarterly":
        qmask = s.str.match(r"^\d{4}Q[1-4]$", na=False)
        parsed = pd.to_datetime(s.where(~qmask), errors="coerce")
        if qmask.any():
            qvals = pd.PeriodIndex(s[qmask], freq="Q").to_timestamp(how="end").normalize()
            parsed.loc[qmask] = qvals
        return parsed
    return pd.to_datetime(s, errors="coerce")


def validate_all(
    root: str | Path | None = None,
    *,
    upstream: str | None = None,
    required_only: bool = False,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    contracts = load_contracts(root, upstream=upstream, include_optional=not required_only)
    rows = [validate_contract(root, c, sample_rows=sample_rows) for c in contracts]
    return pd.DataFrame(rows)


def write_source_manifest(root: str | Path | None = None, *, required_only: bool = False) -> Path:
    df = validate_all(root, required_only=required_only)
    return write_csv(df, project_root(root) / "output/qa/source_manifest.csv")
