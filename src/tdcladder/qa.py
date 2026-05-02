from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .config import project_root
from .contracts import load_contracts
from .io import write_csv


def dataset_summary(name: str, df: pd.DataFrame, *, date_col: str | None = None, key_cols: Iterable[str] = ()) -> dict[str, object]:
    out: dict[str, object] = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_key_count": None,
        "date_min": "",
        "date_max": "",
    }
    keys = [c for c in key_cols if c in df.columns]
    if keys:
        out["duplicate_key_count"] = int(df.duplicated(keys).sum())
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().any():
            out["date_min"] = str(dates.min().date())
            out["date_max"] = str(dates.max().date())
    return out


def missingness_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset", "column", "missing_count", "missing_share"])
    rows = []
    for col in df.columns:
        count = int(df[col].isna().sum())
        rows.append({"dataset": name, "column": col, "missing_count": count, "missing_share": count / len(df)})
    return pd.DataFrame(rows)


def duplicate_key_table(name: str, df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in key_cols if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["dataset", "key", "duplicate_count"])
    dupes = df[df.duplicated(cols, keep=False)].copy()
    if dupes.empty:
        return pd.DataFrame([{"dataset": name, "key": ",".join(cols), "duplicate_count": 0}])
    return pd.DataFrame([{"dataset": name, "key": ",".join(cols), "duplicate_count": int(len(dupes))}])


def weight_bounds_table(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty or "liquidity_weight" not in weights.columns:
        return pd.DataFrame(columns=["weight_family", "rows", "min_weight", "max_weight", "outside_unit_interval"])
    rows = []
    for fam, g in weights.groupby("weight_family", dropna=False):
        w = pd.to_numeric(g["liquidity_weight"], errors="coerce")
        rows.append(
            {
                "weight_family": fam,
                "rows": int(len(g)),
                "min_weight": float(w.min()) if w.notna().any() else None,
                "max_weight": float(w.max()) if w.notna().any() else None,
                "outside_unit_interval": int(((w < 0) | (w > 1)).sum()),
                "missing_weight": int(w.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def weight_source_status_table(weights: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "weight_family",
        "source_status",
        "rows",
        "row_share_within_family",
        "date_min",
        "date_max",
        "mean_liquidity_weight",
        "min_liquidity_weight",
        "max_liquidity_weight",
    ]
    required = {"weight_family", "source_status", "liquidity_weight"}
    if weights.empty or not required.issubset(weights.columns):
        return pd.DataFrame(columns=columns)

    frame = weights.copy()
    frame["source_status"] = frame["source_status"].fillna("missing_source_status")
    frame["liquidity_weight"] = pd.to_numeric(frame["liquidity_weight"], errors="coerce")
    if "month" in frame.columns:
        frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    else:
        frame["month"] = pd.NaT

    family_rows = frame.groupby("weight_family", dropna=False).size().rename("family_rows")
    grouped = frame.groupby(["weight_family", "source_status"], dropna=False).agg(
        rows=("liquidity_weight", "size"),
        date_min=("month", "min"),
        date_max=("month", "max"),
        mean_liquidity_weight=("liquidity_weight", "mean"),
        min_liquidity_weight=("liquidity_weight", "min"),
        max_liquidity_weight=("liquidity_weight", "max"),
    )
    out = grouped.join(family_rows, on="weight_family").reset_index()
    out["row_share_within_family"] = out["rows"] / out["family_rows"].replace(0, pd.NA)
    for col in ["date_min", "date_max"]:
        out[col] = out[col].dt.date.astype("string").fillna("")
    return out[columns].sort_values(["weight_family", "source_status"]).reset_index(drop=True)


def family_support_matrix(weighted_by_bucket: pd.DataFrame, quarterly: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = [
        "weight_family",
        "supply_basis",
        "sample_window",
        "date_min",
        "date_max",
        "rows",
        "supported_rows",
        "row_share_model_backed",
        "raw_supply",
        "supported_raw_supply",
        "dollar_share_model_backed",
    ]
    required = {"month", "weight_family", "supply_basis", "source_status", "treasury_supply_raw"}
    if weighted_by_bucket.empty or not required.issubset(weighted_by_bucket.columns):
        return pd.DataFrame(columns=columns)

    frame = weighted_by_bucket.copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    frame["quarter"] = frame["month"].dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    frame["treasury_supply_raw"] = pd.to_numeric(frame["treasury_supply_raw"], errors="coerce").fillna(0)
    status = frame["source_status"].astype("string").fillna("missing_source_status").str.lower()
    frame["is_model_backed"] = ~status.str.contains("fallback|missing|stub", regex=True)

    windows: list[tuple[str, pd.DataFrame]] = [("full_sample", frame)]
    if quarterly is not None and not quarterly.empty and {"quarter", "is_tdc_aligned"}.issubset(quarterly.columns):
        q = quarterly.copy()
        q["quarter"] = pd.to_datetime(q["quarter"], errors="coerce").dt.normalize()
        tdc_quarters = set(q.loc[q["is_tdc_aligned"].fillna(False).astype(bool), "quarter"].dropna())
        if tdc_quarters:
            windows.append(("tdc_sample", frame[frame["quarter"].isin(tdc_quarters)]))
            complete = q["is_complete_quarter"].fillna(False).astype(bool) if "is_complete_quarter" in q.columns else pd.Series(True, index=q.index)
            latest_vals = q.loc[q["is_tdc_aligned"].fillna(False).astype(bool) & complete, "quarter"].dropna()
            if not latest_vals.empty:
                latest_quarter = latest_vals.max()
                latest = frame[frame["quarter"].eq(latest_quarter)].copy()
                if not latest.empty:
                    stock = latest["supply_basis"].eq("outstanding_stock")
                    stock_latest_month = latest.loc[stock].groupby(["weight_family", "supply_basis"], dropna=False)["month"].transform("max")
                    latest = latest.loc[~stock | latest["month"].eq(stock_latest_month)]
                    windows.append(("latest_quarter", latest))

    rows: list[dict[str, object]] = []
    for window_name, window in windows:
        if window.empty:
            continue
        for (family, basis), g in window.groupby(["weight_family", "supply_basis"], dropna=False):
            supported = g[g["is_model_backed"]]
            raw = float(g["treasury_supply_raw"].sum())
            supported_raw = float(supported["treasury_supply_raw"].sum())
            rows.append(
                {
                    "weight_family": family,
                    "supply_basis": basis,
                    "sample_window": window_name,
                    "date_min": g["month"].min().date().isoformat() if g["month"].notna().any() else "",
                    "date_max": g["month"].max().date().isoformat() if g["month"].notna().any() else "",
                    "rows": int(len(g)),
                    "supported_rows": int(len(supported)),
                    "row_share_model_backed": len(supported) / len(g) if len(g) else None,
                    "raw_supply": raw,
                    "supported_raw_supply": supported_raw,
                    "dollar_share_model_backed": supported_raw / raw if raw else None,
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(["sample_window", "supply_basis", "weight_family"]).reset_index(drop=True)


def unit_checks_table(root: str | Path | None = None) -> pd.DataFrame:
    rows = []
    for contract in load_contracts(root):
        for col, unit in (contract.get("amount_columns", {}) or {}).items():
            unit_text = str(unit)
            multiplier = 1_000_000 if "million" in unit_text.lower() else 1
            rows.append(
                {
                    "source_contract": contract.get("name"),
                    "upstream": contract.get("upstream"),
                    "column": col,
                    "declared_unit": unit,
                    "multiplier_to_usd": multiplier,
                    "local_path": contract.get("local_path"),
                }
            )
    return pd.DataFrame(rows)


def analysis_unit_assumptions_table() -> pd.DataFrame:
    rows = [
        {
            "column_group": "Treasury auction issuance, outstanding stock, and liquidity-weighted supply",
            "stored_unit": "USD",
            "generated_suffixes": "_usd, _usd_billions",
            "notes": "Buycurve issuance inputs and normalized MSPD stock inputs are stored as dollars.",
        },
        {
            "column_group": "TDC estimates, TDC components, tdcpass flow comparisons",
            "stored_unit": "USD millions",
            "generated_suffixes": "_usd_millions, _usd_billions",
            "notes": "Magnitude checks show source values are million-dollar units; analysis tables keep stored values and add display-scale columns.",
        },
        {
            "column_group": "Liqsub/H.8/OFR macro levels and changes",
            "stored_unit": "USD millions",
            "generated_suffixes": "_usd_millions, _usd_billions",
            "notes": "Deposits, reserves, MMF, ON RRP, and TGA context are treated as million-dollar units.",
        },
        {
            "column_group": "Shares, weights, maturity, regimes, and indicators",
            "stored_unit": "unitless or years",
            "generated_suffixes": "none",
            "notes": "Liquidity weights are measurement assumptions; maturity is in years; flags/regimes are classifiers.",
        },
    ]
    return pd.DataFrame(rows)


def write_qa_outputs(
    root: str | Path | None,
    *,
    datasets: dict[str, tuple[pd.DataFrame, str | None, list[str]]],
    weights: pd.DataFrame,
    weighted_by_bucket: pd.DataFrame | None = None,
    quarterly: pd.DataFrame | None = None,
    source_manifest: pd.DataFrame | None = None,
) -> dict[str, Path]:
    r = project_root(root)
    paths: dict[str, Path] = {}
    summary_rows = [dataset_summary(name, df, date_col=date_col, key_cols=keys) for name, (df, date_col, keys) in datasets.items()]
    paths["dataset_summary"] = write_csv(pd.DataFrame(summary_rows), r / "output/qa/dataset_summary.csv")

    missing_frames = [missingness_table(name, df) for name, (df, _, _) in datasets.items()]
    paths["missingness"] = write_csv(pd.concat(missing_frames, ignore_index=True), r / "output/qa/missingness.csv")

    dup_frames = [duplicate_key_table(name, df, keys) for name, (df, _, keys) in datasets.items() if keys]
    paths["duplicate_keys"] = write_csv(pd.concat(dup_frames, ignore_index=True), r / "output/qa/duplicate_keys.csv")
    paths["weight_bounds"] = write_csv(weight_bounds_table(weights), r / "output/qa/weight_bounds.csv")
    paths["weight_source_status"] = write_csv(weight_source_status_table(weights), r / "output/qa/weight_source_status.csv")
    if weighted_by_bucket is not None:
        paths["family_support_matrix"] = write_csv(family_support_matrix(weighted_by_bucket, quarterly), r / "output/qa/family_support_matrix.csv")
    paths["unit_checks"] = write_csv(unit_checks_table(root), r / "output/qa/unit_checks.csv")
    paths["analysis_unit_assumptions"] = write_csv(analysis_unit_assumptions_table(), r / "output/qa/analysis_unit_assumptions.csv")
    if source_manifest is not None:
        paths["source_manifest"] = write_csv(source_manifest, r / "output/qa/source_manifest.csv")
    paths["qa_report"] = write_qa_report(r, paths)
    return paths


def write_qa_report(root: Path, paths: dict[str, Path]) -> Path:
    lines = [
        "# tdcladder QA report",
        "",
        "This report is generated by `tdcladder build`. It summarizes source contracts, date ranges, duplicate keys, missingness, declared units, and weight bounds.",
        "",
    ]
    for label, path in sorted(paths.items()):
        if label == "qa_report":
            continue
        rel = path.relative_to(root)
        lines.append(f"- `{label}`: `{rel}`")
    out = root / "output/qa/qa_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
