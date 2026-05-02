from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import project_root
from .hashing import sha256_file
from .io import ensure_parent, read_csv, write_csv
from .maturities import load_maturity_buckets

FISCALDATA_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
MSPD_TABLE_3_MARKET_ENDPOINT = "v1/debt/mspd/mspd_table_3_market"
MSPD_TABLE_3_MARKET_URL = f"{FISCALDATA_BASE}/{MSPD_TABLE_3_MARKET_ENDPOINT}"


def download_mspd_table_3_market(
    root: str | Path | None = None,
    *,
    page_size: int = 10_000,
) -> tuple[Path, Path]:
    """Download FiscalData MSPD marketable detail into project-local data/raw."""
    r = project_root(root)
    csv_path = r / "data/raw/treasury/mspd_table_3_market.csv"
    meta_path = r / "data/raw/treasury/mspd_table_3_market_meta.json"

    rows: list[dict[str, Any]] = []
    pages_fetched = 0
    page = 1
    last_payload: dict[str, Any] = {}
    started_at = datetime.now(UTC).isoformat()

    while True:
        params = {
            "format": "json",
            "sort": "record_date,src_line_nbr",
            "page[size]": str(page_size),
            "page[number]": str(page),
        }
        url = f"{MSPD_TABLE_3_MARKET_URL}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "tdcladder/0.2.0"})
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))

        last_payload = payload
        rows.extend(payload.get("data", []) or [])
        pages_fetched += 1

        links = payload.get("links", {}) or {}
        if not links.get("next"):
            break
        page += 1

    raw = pd.DataFrame(rows)
    write_csv(raw, csv_path)
    meta = {
        "source": "FiscalData Monthly Statement of the Public Debt, Table 3 Marketable",
        "endpoint": MSPD_TABLE_3_MARKET_ENDPOINT,
        "url": MSPD_TABLE_3_MARKET_URL,
        "download_started_at_utc": started_at,
        "download_finished_at_utc": datetime.now(UTC).isoformat(),
        "page_size": page_size,
        "pages_fetched": pages_fetched,
        "rows_fetched": len(raw),
        "csv_sha256": sha256_file(csv_path),
        "meta": last_payload.get("meta", {}),
    }
    ensure_parent(meta_path).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, meta_path


def _security_type(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text.startswith("bills"):
        return "Bill"
    if text == "notes":
        return "Note"
    if text == "bonds":
        return "Bond"
    if "floating rate" in text:
        return "FRN"
    if "inflation" in text:
        return "TIPS"
    return "Unknown"


def _remaining_maturity_bucket(days: float, buckets: list[dict[str, Any]]) -> str:
    if pd.isna(days) or days < 0:
        return "unknown"
    for bucket in buckets:
        min_days = bucket.get("min_days")
        max_days = bucket.get("max_days")
        if min_days is not None and days < float(min_days):
            continue
        if max_days is not None and days > float(max_days):
            continue
        return str(bucket["name"])
    return "unknown"


def _write_stock_reconciliation(raw: pd.DataFrame, root: Path) -> None:
    df = raw.copy()
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    df["outstanding_amount"] = pd.to_numeric(df["outstanding_amt"], errors="coerce") * 1_000_000
    class1 = df["security_class1_desc"].astype("string")
    class2 = df["security_class2_desc"].astype("string")
    detail_classes = {
        "Bills Maturity Value",
        "Notes",
        "Bonds",
        "Floating Rate Notes",
        "Inflation-Protected Securities",
        "Inflation-Indexed Notes",
        "Inflation-Indexed Bonds",
    }
    detail = df[class1.isin(detail_classes) & ~class2.str.contains("Total", case=False, na=False)]
    detail = detail.groupby("record_date", as_index=False).agg(
        detail_outstanding_amount=("outstanding_amount", "sum"),
        detail_security_rows=("outstanding_amount", "count"),
    )
    total = df[class1.eq("Total Marketable")].groupby("record_date", as_index=False).agg(
        total_marketable_outstanding_amount=("outstanding_amount", "sum")
    )
    out = detail.merge(total, on="record_date", how="outer").sort_values("record_date")
    out["detail_to_total_marketable_ratio"] = (
        out["detail_outstanding_amount"] / out["total_marketable_outstanding_amount"]
    )
    write_csv(out, root / "output/qa/treasury_stock_reconciliation.csv")


def build_monthly_outstanding_by_maturity(
    root: str | Path | None = None,
    *,
    input_path: str | Path | None = None,
) -> pd.DataFrame:
    r = project_root(root)
    src_path = Path(input_path) if input_path is not None else r / "data/raw/treasury/mspd_table_3_market.csv"
    raw = read_csv(src_path, low_memory=False)
    required = {
        "record_date",
        "security_type_desc",
        "security_class1_desc",
        "security_class2_desc",
        "maturity_date",
        "outstanding_amt",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(f"MSPD table 3 market file is missing columns: {missing}")
    _write_stock_reconciliation(raw, r)

    df = raw.copy()
    df = df[df["security_type_desc"].astype("string").str.casefold().eq("marketable")]
    df = df[~df["security_class2_desc"].astype("string").str.contains("total", case=False, na=False)]
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    df["month"] = df["record_date"].dt.to_period("M").dt.to_timestamp()
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df["outstanding_amount"] = pd.to_numeric(df["outstanding_amt"], errors="coerce") * 1_000_000
    df["security_type"] = df["security_class1_desc"].map(_security_type)
    df = df.dropna(subset=["month", "maturity_date", "outstanding_amount"])
    df = df[(df["outstanding_amount"] > 0) & df["security_type"].ne("Unknown")]

    df["remaining_maturity_days"] = (df["maturity_date"] - df["record_date"]).dt.days
    buckets = load_maturity_buckets(r)
    df["maturity_bucket"] = df["remaining_maturity_days"].map(lambda days: _remaining_maturity_bucket(days, buckets))
    df = df[df["maturity_bucket"].ne("unknown")]
    df["weighted_amount_years"] = df["outstanding_amount"] * (df["remaining_maturity_days"] / 365.25)

    out = df.groupby(["month", "security_type", "maturity_bucket"], as_index=False).agg(
        outstanding_amount=("outstanding_amount", "sum"),
        weighted_amount_years=("weighted_amount_years", "sum"),
        cusip_count=("security_class2_desc", "nunique"),
        record_date=("record_date", "max"),
    )
    out["weighted_maturity_years"] = out["weighted_amount_years"] / out["outstanding_amount"]
    out = out.drop(columns=["weighted_amount_years"])
    out = out.sort_values(["month", "security_type", "maturity_bucket"]).reset_index(drop=True)
    write_csv(out, r / "data/raw/treasury/monthly_outstanding_by_maturity.csv")
    return out


def download_and_build_treasury_stock(root: str | Path | None = None) -> pd.DataFrame:
    download_mspd_table_3_market(root)
    return build_monthly_outstanding_by_maturity(root)
