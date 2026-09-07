from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import project_root
from ..contracts import get_contract, validate_all
from ..io import read_csv, write_csv
from ..maturities import add_bucket_metadata, normalize_bucket
from ..qa import TDC_PRIMARY_ANCHOR, write_qa_outputs
from ..weights import build_all_weight_families
from ..weights.fixed import _normalise_security_type

TDC_LEGACY_BANK_ANCHOR = "tdc_tier2_interest_corrected_bank_only_ru_flow"
TDC_MODERN_DI_MMF_RRP_ANCHOR = "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow"
TDC_LONG_HISTORY_BANK_ANCHOR = TDC_PRIMARY_ANCHOR
TDC_LONG_HISTORY_DI_ANCHOR = "tdc_tier2_regression_mmf_rrp_prop_depository_institution_np_cu_ru_flow"

TREASURY_USD_COLUMNS = [
    "raw_treasury_supply_q",
    "liquid_treasury_supply_q",
    "duration_gap_supply_q",
    "bill_raw_supply_q",
    "frn_raw_supply_q",
    "coupon_raw_supply_q",
    "bill_liquid_supply_q",
    "frn_liquid_supply_q",
    "coupon_liquid_supply_q",
    "d_raw_treasury_supply_q",
    "d_liquid_treasury_supply_q",
]
USD_MILLION_COLUMNS = [
    "deposits_quarter_end",
    "domestic_deposits_quarter_end",
    "total_mmf_assets_quarter_end",
    "mmf_treasury_holdings_quarter_end",
    "on_rrp_quarter_end",
    "reserves_quarter_end",
    "tga_quarter_end",
    "tdc_base_bank_only_ru_flow",
    "tdc_base_broad_depository_np_cu_ru_flow",
    TDC_LEGACY_BANK_ANCHOR,
    TDC_MODERN_DI_MMF_RRP_ANCHOR,
    TDC_LONG_HISTORY_BANK_ANCHOR,
    TDC_LONG_HISTORY_DI_ANCHOR,
    "tdc_tier3_fiscal_corrected_bank_only_ru_flow",
    "tdc_level_bank_only_sensitivity",
    "fed_tsy_tx",
    "row_tsy_tx",
    "minus_treasury_operating_cash_tx",
    "tdc_bank_only_qoq",
    "tdc_broad_depository_qoq",
    "tdc_domestic_bank_only_qoq",
    "total_deposits_bank_qoq",
    "domestic_nonbank_deposits_qoq",
    "d_deposits_q",
    "d_domestic_deposits_q",
    "d_total_mmf_assets_q",
    "d_mmf_treasury_holdings_q",
    "d_on_rrp_q",
    "d_reserves_q",
    "d_tga_q",
]


def _path(root: str | Path | None, rel: str) -> Path:
    return project_root(root) / rel


def _read_contract(root: str | Path | None, name: str, *, required: bool = False) -> pd.DataFrame | None:
    contract = get_contract(root, name)
    path = _path(root, str(contract["local_path"]))
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required source {name} is missing at {contract['local_path']}")
        return None
    return read_csv(path)


def _monthify(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _quarterify_dates(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    qmask = s.str.match(r"^\d{4}Q[1-4]$", na=False)
    out = pd.to_datetime(s.where(~qmask), errors="coerce")
    if qmask.any():
        out.loc[qmask] = pd.PeriodIndex(s[qmask], freq="Q").to_timestamp(how="end").normalize()
    return out.dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()


def _standardize_supply_basis(
    src: pd.DataFrame,
    *,
    root: str | Path | None,
    amount_col: str,
    supply_basis: str,
) -> pd.DataFrame:
    df = src.copy()
    df["month"] = _monthify(df["month"])
    df["security_type"] = df["security_type"].map(_normalise_security_type)
    df["maturity_bucket"] = df["maturity_bucket"].map(lambda x: normalize_bucket(x, root))

    if amount_col not in df.columns:
        raise KeyError(f"Supply panel needs {amount_col}")
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
    df["treasury_supply_raw"] = df[amount_col]
    df["supply_basis"] = supply_basis

    for col in ["accepted_amount_sum", "outstanding_amount", "offering_amount_sum", "auction_count", "weighted_maturity_years", "bill_share_by_accepted_amount"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def wavg(g: pd.DataFrame, value_col: str, weight_col: str) -> float:
        vals = pd.to_numeric(g[value_col], errors="coerce")
        weights = pd.to_numeric(g[weight_col], errors="coerce").fillna(0)
        mask = vals.notna() & (weights > 0)
        if not mask.any():
            return float(vals.mean()) if vals.notna().any() else np.nan
        return float(np.average(vals[mask], weights=weights[mask]))

    grouped_rows: list[dict[str, Any]] = []
    def nansafe_sum(g: pd.DataFrame, col: str) -> float:
        return float(g[col].sum()) if g[col].notna().any() else np.nan

    for keys, g in df.groupby(["month", "security_type", "maturity_bucket", "supply_basis"], dropna=False):
        month, security_type, maturity_bucket, basis = keys
        grouped_rows.append(
            {
                "month": month,
                "security_type": security_type,
                "maturity_bucket": maturity_bucket,
                "supply_basis": basis,
                "treasury_supply_raw": float(g["treasury_supply_raw"].sum()),
                "accepted_amount_sum": nansafe_sum(g, "accepted_amount_sum"),
                "outstanding_amount": nansafe_sum(g, "outstanding_amount"),
                "offering_amount_sum": nansafe_sum(g, "offering_amount_sum"),
                "auction_count": nansafe_sum(g, "auction_count"),
                "weighted_maturity_years": wavg(g, "weighted_maturity_years", "treasury_supply_raw"),
                "bill_share_by_accepted_amount": wavg(g, "bill_share_by_accepted_amount", "treasury_supply_raw"),
                "supply_amount_column": amount_col,
            }
        )
    return pd.DataFrame(grouped_rows)


def build_raw_treasury_supply(root: str | Path | None = None) -> pd.DataFrame:
    flow_src = _read_contract(root, "buycurve_monthly_issuance_maturity_panel", required=True)
    assert flow_src is not None
    frames = [
        _standardize_supply_basis(
            flow_src,
            root=root,
            amount_col="accepted_amount_sum",
            supply_basis="issuance_flow",
        )
    ]

    stock_src = _read_contract(root, "treasury_monthly_outstanding_by_maturity", required=False)
    if stock_src is not None:
        frames.append(
            _standardize_supply_basis(
                stock_src,
                root=root,
                amount_col="outstanding_amount",
                supply_basis="outstanding_stock",
            )
        )

    out = pd.concat(frames, ignore_index=True)
    out = add_bucket_metadata(out, root)
    out["is_bill"] = out["security_type"].isin(["Bill", "CMB"])
    out["is_frn"] = out["security_type"].eq("FRN")
    out["is_coupon"] = out["security_type"].isin(["Note", "Bond", "TIPS"])
    out = out.sort_values(["supply_basis", "month", "maturity_rank", "security_type"]).reset_index(drop=True)
    write_csv(out, _path(root, "data/interim/raw_treasury_supply_by_maturity.csv"))
    return out


def _load_liqsub(root: str | Path | None = None) -> pd.DataFrame | None:
    analysis = _read_contract(root, "liqsub_monthly_liquidity_substitution_panel_analysis", required=False)
    base = _read_contract(root, "liqsub_monthly_liquidity_substitution_panel", required=False)
    df = analysis if analysis is not None else base
    if df is None:
        return None
    df = df.copy()
    df["month"] = _monthify(df["month"])
    return df.drop_duplicates("month")


def _load_turnover(root: str | Path | None = None) -> pd.DataFrame | None:
    return _read_contract(root, "market_liquidity_turnover_by_maturity", required=False)


def _load_auction(root: str | Path | None = None) -> pd.DataFrame | None:
    return _read_contract(root, "buycurve_auction_allotment_panel_base_slim", required=False)


def _load_z1_holder_context(root: str | Path | None = None) -> pd.DataFrame | None:
    return _read_contract(root, "buycurve_z1_treasury_holders_clean", required=False)


def build_liquidity_weights(root: str | Path | None, supply: pd.DataFrame, liqsub: pd.DataFrame | None) -> pd.DataFrame:
    weights = build_all_weight_families(
        supply,
        root=root,
        liqsub=liqsub,
        auction_allotments=_load_auction(root),
        z1_holder_context=_load_z1_holder_context(root),
        turnover=_load_turnover(root),
    )
    write_csv(weights, _path(root, "data/clean/liquidity_weights.csv"))
    return weights


def build_liquidity_weighted_supply(root: str | Path | None, supply: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["month", "security_type", "maturity_bucket"]
    supply_keys = [*keys, "supply_basis"]
    weight_keys = [*keys, "weight_family"]
    if supply.duplicated(supply_keys).any():
        raise ValueError("Supply rows must be unique by month/security_type/maturity_bucket/supply_basis.")
    if weights.duplicated(weight_keys).any():
        raise ValueError("Weight rows must be unique by month/security_type/maturity_bucket/weight_family.")

    merged = supply.merge(weights, on=keys, how="inner", validate="many_to_many")
    merged["liquid_treasury_supply"] = merged["treasury_supply_raw"] * merged["liquidity_weight"]
    merged["duration_gap_supply"] = merged["treasury_supply_raw"] - merged["liquid_treasury_supply"]
    merged = merged.sort_values(["supply_basis", "weight_family", "month", "maturity_rank", "security_type"]).reset_index(drop=True)
    write_csv(merged, _path(root, "data/clean/liquidity_weighted_treasury_supply_by_bucket.csv"))

    rows: list[dict[str, Any]] = []
    for (month, family, basis), g in merged.groupby(["month", "weight_family", "supply_basis"], dropna=False):
        raw = float(g["treasury_supply_raw"].sum())
        liquid = float(g["liquid_treasury_supply"].sum())
        bill_raw = float(g.loc[g["is_bill"], "treasury_supply_raw"].sum())
        frn_raw = float(g.loc[g["is_frn"], "treasury_supply_raw"].sum())
        coupon_raw = float(g.loc[g["is_coupon"], "treasury_supply_raw"].sum())
        bill_liquid = float(g.loc[g["is_bill"], "liquid_treasury_supply"].sum())
        frn_liquid = float(g.loc[g["is_frn"], "liquid_treasury_supply"].sum())
        coupon_liquid = float(g.loc[g["is_coupon"], "liquid_treasury_supply"].sum())
        maturity = pd.to_numeric(g["weighted_maturity_years"], errors="coerce")
        raw_wam = _weighted_average(maturity, g["treasury_supply_raw"])
        liquid_wam = _weighted_average(maturity, g["liquid_treasury_supply"])
        rows.append(
            {
                "month": month,
                "weight_family": family,
                "supply_basis": basis,
                "raw_treasury_supply": raw,
                "liquid_treasury_supply": liquid,
                "duration_gap_supply": raw - liquid,
                "average_liquidity_weight": liquid / raw if raw else np.nan,
                "bill_raw_supply": bill_raw,
                "frn_raw_supply": frn_raw,
                "coupon_raw_supply": coupon_raw,
                "bill_liquid_supply": bill_liquid,
                "frn_liquid_supply": frn_liquid,
                "coupon_liquid_supply": coupon_liquid,
                "raw_bill_share": bill_raw / raw if raw else np.nan,
                "raw_frn_share": frn_raw / raw if raw else np.nan,
                "raw_coupon_share": coupon_raw / raw if raw else np.nan,
                "liquid_bill_share": bill_liquid / liquid if liquid else np.nan,
                "liquid_frn_share": frn_liquid / liquid if liquid else np.nan,
                "liquid_coupon_share": coupon_liquid / liquid if liquid else np.nan,
                "raw_weighted_maturity_years": raw_wam,
                "liquid_weighted_maturity_years": liquid_wam,
                "bucket_count": int(g[["security_type", "maturity_bucket"]].drop_duplicates().shape[0]),
            }
        )
    monthly = pd.DataFrame(rows).sort_values(["supply_basis", "weight_family", "month"]).reset_index(drop=True)
    for col in ["raw_treasury_supply", "liquid_treasury_supply", "duration_gap_supply"]:
        monthly[f"d_{col}"] = monthly.groupby(["weight_family", "supply_basis"])[col].diff()
    write_csv(monthly, _path(root, "data/clean/monthly_ladder_panel.csv"))
    return merged, monthly


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0)
    mask = vals.notna() & (w > 0)
    if not mask.any():
        return float(vals.mean()) if vals.notna().any() else np.nan
    return float(np.average(vals[mask], weights=w[mask]))


def _add_quarterly_unit_columns(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    for col in TREASURY_USD_COLUMNS:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_usd"] = values
            out[f"{col}_usd_billions"] = values / 1_000_000_000
    for col in USD_MILLION_COLUMNS:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_usd_millions"] = values
            out[f"{col}_usd_billions"] = values / 1_000
    return out


def join_liqsub_outcomes(root: str | Path | None, monthly: pd.DataFrame, liqsub: pd.DataFrame | None) -> pd.DataFrame:
    if liqsub is None or liqsub.empty:
        out = monthly.copy()
        out["liqsub_join_status"] = "missing_liqsub_monthly_panel"
    else:
        renamed = liqsub.rename(
            columns={
                "bill_share": "issuance_context_bill_share",
                "weighted_maturity_years": "issuance_context_weighted_maturity_years",
                "gross_bill_issuance": "issuance_context_gross_bill_issuance",
                "coupon_issuance": "issuance_context_coupon_issuance",
            }
        )
        out = monthly.merge(renamed, on="month", how="left", suffixes=("", "_liqsub"))
        out["liqsub_join_status"] = out["deposits"].notna().map({True: "matched", False: "unmatched_month"}) if "deposits" in out.columns else "matched_no_deposits_column"
    write_csv(out, _path(root, "data/clean/monthly_ladder_liqsub_panel.csv"))
    return out


def build_quarterly_tdc_comparison(root: str | Path | None, monthly_joined: pd.DataFrame) -> pd.DataFrame:
    df = monthly_joined.copy()
    df["quarter"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    rows = []
    for (quarter, family, basis), g in df.groupby(["quarter", "weight_family", "supply_basis"], dropna=False):
        g = g.sort_values("month").copy()
        if basis == "outstanding_stock":
            supply_g = g
            last = supply_g.iloc[-1]
            raw = float(pd.to_numeric(supply_g["raw_treasury_supply"], errors="coerce").dropna().iloc[-1])
            liquid = float(pd.to_numeric(supply_g["liquid_treasury_supply"], errors="coerce").dropna().iloc[-1])
            supply_aggregation = "quarter_end"
            bill_raw = float(last.get("bill_raw_supply", np.nan))
            frn_raw = float(last.get("frn_raw_supply", np.nan))
            coupon_raw = float(last.get("coupon_raw_supply", np.nan))
            bill_liquid = float(last.get("bill_liquid_supply", np.nan))
            frn_liquid = float(last.get("frn_liquid_supply", np.nan))
            coupon_liquid = float(last.get("coupon_liquid_supply", np.nan))
            raw_wam = float(last.get("raw_weighted_maturity_years", np.nan))
            liquid_wam = float(last.get("liquid_weighted_maturity_years", np.nan))
        else:
            raw = float(g["raw_treasury_supply"].sum())
            liquid = float(g["liquid_treasury_supply"].sum())
            supply_aggregation = "quarter_sum"
            bill_raw = float(g["bill_raw_supply"].sum())
            frn_raw = float(g["frn_raw_supply"].sum())
            coupon_raw = float(g["coupon_raw_supply"].sum())
            bill_liquid = float(g["bill_liquid_supply"].sum())
            frn_liquid = float(g["frn_liquid_supply"].sum())
            coupon_liquid = float(g["coupon_liquid_supply"].sum())
            raw_wam = _weighted_average(g["raw_weighted_maturity_years"], g["raw_treasury_supply"])
            liquid_wam = _weighted_average(g["liquid_weighted_maturity_years"], g["liquid_treasury_supply"])
        months_observed = int(g["month"].nunique())
        row: dict[str, Any] = {
            "quarter": quarter,
            "weight_family": family,
            "supply_basis": basis,
            "supply_aggregation": supply_aggregation,
            "raw_treasury_supply_q": raw,
            "liquid_treasury_supply_q": liquid,
            "duration_gap_supply_q": raw - liquid,
            "average_liquidity_weight_q": liquid / raw if raw else np.nan,
            "bill_raw_supply_q": bill_raw,
            "frn_raw_supply_q": frn_raw,
            "coupon_raw_supply_q": coupon_raw,
            "bill_liquid_supply_q": bill_liquid,
            "frn_liquid_supply_q": frn_liquid,
            "coupon_liquid_supply_q": coupon_liquid,
            "bill_share_raw_q": bill_raw / raw if raw else np.nan,
            "frn_share_raw_q": frn_raw / raw if raw else np.nan,
            "coupon_share_raw_q": coupon_raw / raw if raw else np.nan,
            "bill_share_liquid_q": bill_liquid / liquid if liquid else np.nan,
            "frn_share_liquid_q": frn_liquid / liquid if liquid else np.nan,
            "coupon_share_liquid_q": coupon_liquid / liquid if liquid else np.nan,
            "raw_weighted_maturity_years_q": raw_wam,
            "liquid_weighted_maturity_years_q": liquid_wam,
            "months_observed": months_observed,
            "is_complete_quarter": months_observed == 3,
        }
        for col in ["deposits", "domestic_deposits", "total_mmf_assets", "mmf_treasury_holdings", "on_rrp", "reserves", "tga"]:
            if col in g.columns:
                values = pd.to_numeric(g[col], errors="coerce").dropna()
                row[f"{col}_quarter_end"] = values.iloc[-1] if not values.empty else np.nan
        for col in [
            "issuance_context_bill_share",
            "issuance_context_weighted_maturity_years",
            "issuance_context_gross_bill_issuance",
            "issuance_context_coupon_issuance",
        ]:
            if col in g.columns:
                values = pd.to_numeric(g[col], errors="coerce").dropna()
                row[f"{col}_quarter_end"] = values.iloc[-1] if not values.empty else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["supply_basis", "weight_family", "quarter"]).reset_index(drop=True)

    tdc = _read_contract(root, "tdcest_tdc_estimates", required=False)
    if tdc is not None and "date" in tdc.columns:
        tdc = tdc.copy()
        tdc["quarter"] = _quarterify_dates(tdc["date"])
        keep = [c for c in [
            "quarter",
            "tdc_base_bank_only_ru_flow",
            "tdc_base_broad_depository_np_cu_ru_flow",
            TDC_LEGACY_BANK_ANCHOR,
            TDC_MODERN_DI_MMF_RRP_ANCHOR,
            "tdc_tier3_fiscal_corrected_bank_only_ru_flow",
            "tdc_level_bank_only_sensitivity",
        ] if c in tdc.columns]
        out = out.merge(tdc[keep].drop_duplicates("quarter"), on="quarter", how="left")

    regression_tdc = _read_contract(root, "tdcest_tier2_regression_series", required=False)
    if regression_tdc is not None and "date" in regression_tdc.columns:
        regression_tdc = regression_tdc.copy()
        regression_tdc["quarter"] = _quarterify_dates(regression_tdc["date"])
        keep = [c for c in [
            "quarter",
            TDC_LONG_HISTORY_BANK_ANCHOR,
            TDC_LONG_HISTORY_DI_ANCHOR,
            "tier2_regression_bank_row_method_tier",
            "tier2_regression_di_method_tier",
        ] if c in regression_tdc.columns]
        out = out.merge(regression_tdc[keep], on="quarter", how="left", validate="many_to_one")

    components = _read_contract(root, "tdcest_tdc_components", required=False)
    if components is not None and "date" in components.columns:
        components = components.copy()
        components["quarter"] = _quarterify_dates(components["date"])
        keep = [c for c in ["quarter", "fed_tsy_tx", "row_tsy_tx", "minus_treasury_operating_cash_tx"] if c in components.columns]
        out = out.merge(components[keep].drop_duplicates("quarter"), on="quarter", how="left")

    tdcpass = _read_contract(root, "tdcpass_quarterly_panel", required=False)
    if tdcpass is not None and "quarter" in tdcpass.columns:
        tdcpass = tdcpass.copy()
        tdcpass["quarter"] = _quarterify_dates(tdcpass["quarter"])
        keep = [c for c in [
            "quarter",
            "tdc_bank_only_qoq",
            "tdc_broad_depository_qoq",
            "tdc_domestic_bank_only_qoq",
            "total_deposits_bank_qoq",
            "domestic_nonbank_deposits_qoq",
            "bill_share",
            "reserve_drain_pressure",
        ] if c in tdcpass.columns]
        tdcpass_keep = tdcpass[keep].drop_duplicates("quarter").rename(columns={"bill_share": "tdcpass_bill_share"})
        out = out.merge(tdcpass_keep, on="quarter", how="left")

    for col in ["raw_treasury_supply_q", "liquid_treasury_supply_q"]:
        out[f"d_{col}"] = out.groupby(["weight_family", "supply_basis"])[col].diff()
    for col in ["deposits", "domestic_deposits", "total_mmf_assets", "mmf_treasury_holdings", "on_rrp", "reserves", "tga"]:
        end_col = f"{col}_quarter_end"
        if end_col in out.columns:
            out[f"d_{col}_q"] = out.groupby(["weight_family", "supply_basis"])[end_col].diff()
    tdc_anchor = TDC_PRIMARY_ANCHOR
    out["is_tdc_aligned"] = out[tdc_anchor].notna() if tdc_anchor in out.columns else False
    out["is_future_or_partial"] = ~out["is_complete_quarter"] | ~out["is_tdc_aligned"]
    out = _add_quarterly_unit_columns(out)
    write_csv(out, _path(root, "data/clean/quarterly_tdc_comparison_panel.csv"))
    return out


def build_all(root: str | Path | None = None, *, strict: bool = False, write_qa: bool = True) -> dict[str, pd.DataFrame]:
    r = project_root(root)
    source_manifest = validate_all(r)
    if strict:
        bad = source_manifest[source_manifest["status"].isin(["missing_required", "failed_missing_columns", "failed_duplicate_keys", "unreadable"])]
        optional_needed = source_manifest[(source_manifest["required_stage"].isin(["monthly_outcome_join", "quarterly_tdc_comparison"])) & source_manifest["exists"].eq(False)]
        if not bad.empty or not optional_needed.empty:
            problems = pd.concat([bad, optional_needed], ignore_index=True)
            raise RuntimeError("Source validation failed in strict mode:\n" + problems[["name", "status", "local_path"]].to_string(index=False))

    supply = build_raw_treasury_supply(r)
    liqsub = _load_liqsub(r)
    weights = build_liquidity_weights(r, supply, liqsub)
    weighted_by_bucket, monthly = build_liquidity_weighted_supply(r, supply, weights)
    monthly_joined = join_liqsub_outcomes(r, monthly, liqsub)
    quarterly = build_quarterly_tdc_comparison(r, monthly_joined)

    outputs = {
        "raw_treasury_supply_by_maturity": supply,
        "liquidity_weights": weights,
        "liquidity_weighted_treasury_supply_by_bucket": weighted_by_bucket,
        "monthly_ladder_panel": monthly,
        "monthly_ladder_liqsub_panel": monthly_joined,
        "quarterly_tdc_comparison_panel": quarterly,
    }
    if write_qa:
        write_qa_outputs(
            r,
            datasets={
                "raw_treasury_supply_by_maturity": (supply, "month", ["month", "security_type", "maturity_bucket", "supply_basis"]),
                "liquidity_weights": (weights, "month", ["month", "security_type", "maturity_bucket", "weight_family"]),
                "liquidity_weighted_treasury_supply_by_bucket": (weighted_by_bucket, "month", ["month", "security_type", "maturity_bucket", "weight_family", "supply_basis"]),
                "monthly_ladder_panel": (monthly, "month", ["month", "weight_family", "supply_basis"]),
                "monthly_ladder_liqsub_panel": (monthly_joined, "month", ["month", "weight_family", "supply_basis"]),
                "quarterly_tdc_comparison_panel": (quarterly, "quarter", ["quarter", "weight_family", "supply_basis"]),
            },
            weights=weights,
            weighted_by_bucket=weighted_by_bucket,
            quarterly=quarterly,
            source_manifest=source_manifest,
        )
    return outputs
