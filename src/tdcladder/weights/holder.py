from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import load_config
from ..maturities import normalize_bucket
from .fixed import KEY_COLS, _normalise_security_type, fixed_weights_for_frame


def build_holder_revealed_weights(
    supply: pd.DataFrame,
    *,
    auction_allotments: pd.DataFrame | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    frame = supply[KEY_COLS].drop_duplicates().copy()
    frame["maturity_bucket"] = frame["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
    frame["security_type"] = frame["security_type"].map(_normalise_security_type)
    frame["weight_family"] = "holder_revealed_first_pass"
    frame["base_fixed_weight"] = fixed_weights_for_frame(frame, root=root, ladder_name="fixed_baseline")

    if auction_allotments is None or auction_allotments.empty:
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "fallback_fixed_no_auction_allotments"
        frame["measurement_assumption"] = "Holder-revealed family fallback: fixed baseline because allotment data are unavailable."
        frame["weight_source_detail"] = "requires buycurve auction_allotment_panel_base_slim.csv"
        return frame

    required = {"security_type", "maturity_bucket", "broad_investor_class", "allotment_amount"}
    if not required.issubset(auction_allotments.columns):
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "fallback_fixed_missing_holder_columns"
        frame["measurement_assumption"] = "Holder-revealed family fallback: allotment data missing required columns."
        frame["weight_source_detail"] = ",".join(sorted(required - set(auction_allotments.columns)))
        return frame

    cfg = load_config(root, "holder_revealed_weights.yaml")
    scores = {str(k): float(v) for k, v in (cfg.get("cash_like_class_scores", {}) or {}).items()}
    fixed_share = float(cfg.get("blend", {}).get("fixed_weight_share", 0.60))
    holder_share = float(cfg.get("blend", {}).get("holder_score_share", 0.40))

    a = auction_allotments.copy()
    date_col = "issue_date" if "issue_date" in a.columns else "auction_date"
    a["month"] = pd.to_datetime(a[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    a["maturity_bucket"] = a["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
    a["security_type"] = a["security_type"].map(_normalise_security_type)
    a["broad_investor_class"] = a["broad_investor_class"].astype("string").str.strip().fillna("unknown")
    a["class_score"] = a["broad_investor_class"].map(lambda x: scores.get(str(x), scores.get("unknown", 0.20)))
    a["allotment_amount"] = pd.to_numeric(a["allotment_amount"], errors="coerce").fillna(0)
    a = a.dropna(subset=["month", "maturity_bucket"])

    if a.empty or a["allotment_amount"].sum() <= 0:
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "fallback_fixed_no_positive_allotments"
        frame["measurement_assumption"] = "Holder-revealed family fallback: no positive allotment amounts."
        frame["weight_source_detail"] = "auction allotment panel"
        return frame

    a["score_amount"] = a["allotment_amount"] * a["class_score"]
    grouped = a.groupby(["month", "security_type", "maturity_bucket"], as_index=False).agg(
        total_allotment_amount=("allotment_amount", "sum"),
        score_amount=("score_amount", "sum"),
    )
    grouped["holder_cashlike_score"] = grouped["score_amount"] / grouped["total_allotment_amount"].replace(0, pd.NA)
    grouped = grouped.drop(columns=["score_amount"])

    merged = frame.merge(grouped, on=KEY_COLS, how="left")
    has_holder = merged["holder_cashlike_score"].notna()
    merged["holder_cashlike_score"] = merged["holder_cashlike_score"].fillna(merged["base_fixed_weight"])
    merged["liquidity_weight"] = (
        fixed_share * merged["base_fixed_weight"] + holder_share * merged["holder_cashlike_score"]
    ).clip(lower=0, upper=1)
    merged["source_status"] = has_holder.map({True: "first_pass_auction_buyer_blend", False: "fallback_fixed_no_matching_allotment"})
    merged["measurement_assumption"] = "Auction buyer-composition score blended with fixed duration ladder; initial buyers are not final holders."
    merged["weight_source_detail"] = "buycurve auction allotment broad investor classes"
    return merged[KEY_COLS + ["weight_family", "liquidity_weight", "base_fixed_weight", "source_status", "measurement_assumption", "weight_source_detail"]]


def build_holder_stock_context_weights(
    supply: pd.DataFrame,
    *,
    z1_holder_context: pd.DataFrame | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    frame = supply[KEY_COLS].drop_duplicates().copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    frame["quarter"] = frame["month"].dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    frame["maturity_bucket"] = frame["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
    frame["security_type"] = frame["security_type"].map(_normalise_security_type)
    frame["weight_family"] = "holder_stock_context_z1"
    frame["base_fixed_weight"] = fixed_weights_for_frame(frame, root=root, ladder_name="fixed_baseline")

    if z1_holder_context is None or z1_holder_context.empty:
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "fallback_fixed_no_z1_holder_context"
        frame["measurement_assumption"] = "Z.1 holder-stock context fallback: fixed baseline because holder context is unavailable."
        frame["weight_source_detail"] = "requires buycurve z1_treasury_holders_clean.csv"
        return _holder_output(frame)

    required = {"quarter", "broad_investor_class", "measure", "value"}
    if not required.issubset(z1_holder_context.columns):
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "fallback_fixed_missing_z1_holder_columns"
        frame["measurement_assumption"] = "Z.1 holder-stock context fallback: holder context is missing required columns."
        frame["weight_source_detail"] = ",".join(sorted(required - set(z1_holder_context.columns)))
        return _holder_output(frame)

    cfg = load_config(root, "holder_revealed_weights.yaml")
    scores = {str(k): float(v) for k, v in (cfg.get("cash_like_class_scores", {}) or {}).items()}
    blend = cfg.get("z1_stock_context_blend", {}) or {}
    fixed_share = float(blend.get("fixed_weight_share", 0.75))
    holder_share = float(blend.get("holder_score_share", 0.25))
    excluded = set(str(x) for x in (cfg.get("private_sector_exclusions", []) or []))

    z1 = z1_holder_context.copy()
    z1["quarter"] = pd.to_datetime(z1["quarter"], errors="coerce").dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    z1["broad_investor_class"] = z1["broad_investor_class"].astype("string").str.strip().fillna("unknown")
    z1["measure"] = z1["measure"].astype("string").str.strip().str.lower()
    z1["value"] = pd.to_numeric(z1["value"], errors="coerce")
    z1 = z1[
        z1["quarter"].notna()
        & z1["measure"].eq("level")
        & z1["value"].gt(0)
        & ~z1["broad_investor_class"].eq("total")
        & ~z1["broad_investor_class"].isin(excluded)
    ].copy()

    if z1.empty:
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "fallback_fixed_no_positive_z1_holder_levels"
        frame["measurement_assumption"] = "Z.1 holder-stock context fallback: no positive private holder levels."
        frame["weight_source_detail"] = "buycurve Z.1 holder levels"
        return _holder_output(frame)

    z1["class_score"] = z1["broad_investor_class"].map(lambda x: scores.get(str(x), scores.get("unknown", 0.20)))
    z1["score_amount"] = z1["value"] * z1["class_score"]
    context = z1.groupby("quarter", as_index=False).agg(
        z1_holder_stock_amount=("value", "sum"),
        score_amount=("score_amount", "sum"),
    )
    context["z1_holder_stock_cashlike_score"] = context["score_amount"] / context["z1_holder_stock_amount"].replace(0, pd.NA)
    context = context.drop(columns=["score_amount"])

    merged = frame.merge(context, on="quarter", how="left")
    has_context = merged["z1_holder_stock_cashlike_score"].notna()
    merged["z1_holder_stock_cashlike_score"] = merged["z1_holder_stock_cashlike_score"].fillna(merged["base_fixed_weight"])
    merged["liquidity_weight"] = (
        fixed_share * merged["base_fixed_weight"] + holder_share * merged["z1_holder_stock_cashlike_score"]
    ).clip(lower=0, upper=1)
    merged["source_status"] = has_context.map({True: "diagnostic_z1_holder_stock_blend", False: "fallback_fixed_no_matching_z1_quarter"})
    merged["measurement_assumption"] = "Quarterly Z.1 holder-stock context blended with fixed duration ladder; aggregate holders do not identify security-level moneyness."
    merged["weight_source_detail"] = "buycurve Z.1 Treasury holder levels by broad investor class"
    return _holder_output(merged)


def _holder_output(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        KEY_COLS
        + [
            "weight_family",
            "liquidity_weight",
            "base_fixed_weight",
            "source_status",
            "measurement_assumption",
            "weight_source_detail",
        ]
    ]
