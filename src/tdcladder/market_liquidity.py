from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from .config import project_root
from .io import write_csv

TRACE_CATEGORY_TO_SECURITY_TYPE = {
    "Bills": "Bill",
    "FRNs": "FRN",
    "Nominal Coupons": "Coupon",
    "TIPS": "TIPS",
    "Total": "Total",
}


def _trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = [
        candidate
        for candidate in sorted(path.rglob("*"))
        if candidate.suffix.lower() in {".xlsx", ".xls", ".zip"}
    ]
    if not files:
        raise FileNotFoundError(f"No FINRA TRACE aggregate workbooks found under {path}")
    return files


def _read_trace_summary(source: object) -> pd.DataFrame:
    frame = pd.read_excel(source, sheet_name="Summary", header=None)
    title = str(frame.iloc[0, 0])
    if title.startswith("TRACE Volumes - Week of "):
        report_date = pd.to_datetime(title.replace("TRACE Volumes - Week of ", ""), errors="raise")
        table = frame.iloc[4:45, [0, 3]].copy()
        table.columns = ["trace_category", "trace_volume_bn"]
        table["trace_trade_count"] = pd.NA
        source_granularity = "weekly"
    elif title.startswith("TRACE Volumes - "):
        report_date = pd.to_datetime(title.replace("TRACE Volumes - ", ""), errors="raise")
        table = frame.iloc[5:45, [0, 5, 6]].copy()
        table.columns = ["trace_category", "trace_trade_count", "trace_volume_bn"]
        source_granularity = "monthly"
    else:
        raise ValueError(f"Unsupported TRACE workbook title: {title}")

    table["trace_category"] = table["trace_category"].astype("string").str.strip()
    table = table[table["trace_category"].isin(TRACE_CATEGORY_TO_SECURITY_TYPE)]
    table["security_type"] = table["trace_category"].map(TRACE_CATEGORY_TO_SECURITY_TYPE)
    table["month"] = report_date.to_period("M").to_timestamp()
    table["trace_volume"] = pd.to_numeric(table["trace_volume_bn"], errors="coerce") * 1_000_000_000
    table["trace_trade_count"] = pd.to_numeric(table["trace_trade_count"], errors="coerce")
    table["source_granularity"] = source_granularity
    return table.dropna(subset=["trace_volume"])[
        ["month", "security_type", "trace_category", "trace_volume", "trace_trade_count", "source_granularity"]
    ]


def _read_trace_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        frames = []
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                member for member in archive.namelist() if Path(member).suffix.lower() in {".xlsx", ".xls"}
            )
            for member in members:
                with archive.open(member) as handle:
                    frames.append(_read_trace_summary(BytesIO(handle.read())))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    return _read_trace_summary(path)


def build_trace_turnover(
    root: str | Path | None = None,
    *,
    source_path: str | Path,
) -> pd.DataFrame:
    r = project_root(root)
    frames = [_read_trace_file(path) for path in _trace_files(Path(source_path).expanduser().resolve())]
    trace = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if trace.empty:
        raise ValueError("No usable FINRA TRACE aggregate rows found.")

    trace = trace.groupby(["month", "security_type"], as_index=False).agg(
        trace_volume=("trace_volume", "sum"),
        trace_trade_count=("trace_trade_count", "sum"),
        source_granularity=("source_granularity", lambda s: ",".join(sorted(set(s.dropna().astype(str))))),
    )

    stock_path = r / "data/raw/treasury/monthly_outstanding_by_maturity.csv"
    if not stock_path.exists():
        raise FileNotFoundError("Run `tdcladder download-treasury-stock` before building TRACE turnover.")
    stock = pd.read_csv(stock_path, parse_dates=["month"])
    stock_by_type = stock.groupby(["month", "security_type"], as_index=False).agg(
        outstanding_amount=("outstanding_amount", "sum")
    )
    coupon = (
        stock[stock["security_type"].isin(["Note", "Bond"])]
        .groupby("month", as_index=False)
        .agg(outstanding_amount=("outstanding_amount", "sum"))
    )
    coupon["security_type"] = "Coupon"
    total = stock.groupby("month", as_index=False).agg(outstanding_amount=("outstanding_amount", "sum"))
    total["security_type"] = "Total"
    denominator = pd.concat([stock_by_type, coupon, total], ignore_index=True)

    out = trace.merge(denominator, on=["month", "security_type"], how="left")
    out["turnover"] = out["trace_volume"] / out["outstanding_amount"]
    out["maturity_bucket"] = "all"
    out["source_detail"] = "FINRA TRACE Treasury aggregate volume / MSPD outstanding stock"
    out = out.dropna(subset=["turnover"]).sort_values(["month", "security_type"]).reset_index(drop=True)
    write_csv(out, r / "data/raw/market_liquidity/turnover_by_maturity.csv")
    return out
