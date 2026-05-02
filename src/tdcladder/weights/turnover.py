from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..maturities import normalize_bucket
from .fixed import KEY_COLS, _normalise_security_type, fixed_weights_for_frame


def _turnover_security_type(value: object) -> str:
    sec = _normalise_security_type(value)
    if sec in {"Note", "Bond"}:
        return "Coupon"
    return sec


def build_turnover_weights(
    supply: pd.DataFrame,
    *,
    turnover: pd.DataFrame | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    frame = supply[KEY_COLS].drop_duplicates().copy()
    frame["maturity_bucket"] = frame["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
    frame["security_type"] = frame["security_type"].map(_normalise_security_type)
    frame["weight_family"] = "turnover_duration_adjusted"
    frame["base_fixed_weight"] = fixed_weights_for_frame(frame, root=root, ladder_name="fixed_baseline")

    if turnover is None or turnover.empty or "turnover" not in turnover.columns or "month" not in turnover.columns:
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "stub_fallback_duration_penalty_no_turnover"
        frame["measurement_assumption"] = "Turnover family fallback: duration-adjusted fixed baseline because turnover input is unavailable."
        frame["weight_source_detail"] = "optional data/raw/market_liquidity/turnover_by_maturity.csv"
        return frame

    t = turnover.copy()
    t["month"] = pd.to_datetime(t["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    t["turnover"] = pd.to_numeric(t["turnover"], errors="coerce")
    if "security_type" in t.columns:
        t["turnover_security_type"] = t["security_type"].map(_turnover_security_type)
        merge_cols = ["month", "turnover_security_type"]
    elif "maturity_bucket" in t.columns:
        t["maturity_bucket"] = t["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
        merge_cols = ["month", "maturity_bucket"]
    else:
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "stub_fallback_turnover_missing_keys"
        frame["measurement_assumption"] = "Turnover input was present but missing security_type or maturity_bucket keys; fixed baseline used."
        frame["weight_source_detail"] = "optional turnover input"
        return frame

    t = t.dropna(subset=[*merge_cols, "turnover"])
    if t.empty:
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "stub_fallback_turnover_all_missing"
        frame["measurement_assumption"] = "Turnover input was present but unusable; fixed baseline used."
        frame["weight_source_detail"] = "optional turnover input"
        return frame

    t["turnover_norm"] = t.groupby("month")["turnover"].transform(lambda x: x / x.max() if x.max() else 0)
    t = t.groupby(merge_cols, as_index=False)["turnover_norm"].mean()
    if "turnover_security_type" in merge_cols:
        frame["turnover_security_type"] = frame["security_type"].map(_turnover_security_type)
    merged = frame.merge(t, on=merge_cols, how="left")
    has_turnover = merged["turnover_norm"].notna()
    merged["turnover_norm"] = merged["turnover_norm"].fillna(0)

    # Duration discipline: turnover is informative only after multiplying by the fixed duration score.
    turnover_weight = (
        0.70 * merged["base_fixed_weight"] + 0.30 * merged["turnover_norm"] * merged["base_fixed_weight"]
    ).clip(lower=0, upper=1)
    merged["liquidity_weight"] = turnover_weight.where(has_turnover, merged["base_fixed_weight"])
    merged["source_status"] = has_turnover.map({
        True: "first_pass_turnover_duration_blend",
        False: "fallback_fixed_no_matching_turnover",
    })
    merged["measurement_assumption"] = "Turnover blended with fixed duration ladder; high long-bond turnover is not treated as cash-likeness."
    merged["weight_source_detail"] = "data/raw/market_liquidity/turnover_by_maturity.csv"
    return merged[KEY_COLS + ["weight_family", "liquidity_weight", "base_fixed_weight", "source_status", "measurement_assumption", "weight_source_detail"]]
