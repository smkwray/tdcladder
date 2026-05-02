from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..maturities import bucket_shortness_score, normalize_bucket
from .fixed import KEY_COLS, _normalise_security_type, fixed_weights_for_frame


def _rate_score(series: pd.Series) -> pd.Series:
    # Rates in liqsub are percentage points. A 5% short-rate environment maps near 1.
    return pd.to_numeric(series, errors="coerce").clip(lower=0, upper=5).fillna(0) / 5.0


def build_price_first_pass_weights(
    supply: pd.DataFrame,
    *,
    liqsub: pd.DataFrame | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    frame = supply[KEY_COLS].drop_duplicates().copy()
    frame["maturity_bucket"] = frame["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
    frame["security_type"] = frame["security_type"].map(_normalise_security_type)
    frame["weight_family"] = "price_first_pass"
    frame["base_fixed_weight"] = fixed_weights_for_frame(frame, root=root, ladder_name="fixed_baseline")

    if liqsub is None or liqsub.empty or "month" not in liqsub.columns:
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "fallback_fixed_no_price_inputs"
        frame["measurement_assumption"] = "Price family fallback: fixed baseline because monthly rate inputs are unavailable."
        frame["weight_source_detail"] = "requires liqsub monthly rate columns such as bill_yield_3mo, iorb_rate, bill_iorb_spread"
        return frame

    rates = liqsub.copy()
    rates["month"] = pd.to_datetime(rates["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    rates = rates.drop_duplicates("month")

    # Choose available rate columns defensively.
    if "bill_yield_3mo" in rates.columns:
        rates["short_rate_score"] = _rate_score(rates["bill_yield_3mo"])
    elif "bill_yield" in rates.columns:
        rates["short_rate_score"] = _rate_score(rates["bill_yield"])
    elif "iorb_rate" in rates.columns:
        rates["short_rate_score"] = _rate_score(rates["iorb_rate"])
    else:
        rates["short_rate_score"] = 0.0

    if "bill_iorb_spread" in rates.columns:
        # Negative bill-IORB spreads are a crude specialness/convenience-yield proxy.
        spread = pd.to_numeric(rates["bill_iorb_spread"], errors="coerce")
        rates["bill_specialness_score"] = (-spread / 0.50).clip(lower=0, upper=1).fillna(0)
    else:
        rates["bill_specialness_score"] = 0.0

    merged = frame.merge(rates[["month", "short_rate_score", "bill_specialness_score"]], on="month", how="left")
    merged[["short_rate_score", "bill_specialness_score"]] = merged[["short_rate_score", "bill_specialness_score"]].fillna(0)
    shortness = merged["maturity_bucket"].map(lambda b: bucket_shortness_score(b, root))
    adjustment = 1.0 + shortness * (0.08 * merged["short_rate_score"] + 0.07 * merged["bill_specialness_score"])
    merged["liquidity_weight"] = (merged["base_fixed_weight"] * adjustment).clip(lower=0, upper=1)

    # Keep the duration guardrail: long coupons cannot become cash-like through price adjustment alone.
    long_cap = np.where(shortness < 0.35, 0.22, 1.0)
    merged["liquidity_weight"] = np.minimum(merged["liquidity_weight"], long_cap)
    merged["source_status"] = "first_pass_rate_adjustment"
    merged["measurement_assumption"] = "Rate/specialness-adjusted fixed ladder; exploratory measurement assumption, not structural liquidity."
    merged["weight_source_detail"] = "liqsub monthly rates blended with fixed_baseline"
    return merged[KEY_COLS + ["weight_family", "liquidity_weight", "base_fixed_weight", "source_status", "measurement_assumption", "weight_source_detail"]]


def _score_positive_spread(series: pd.Series, *, scale: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values / scale).clip(lower=0, upper=1).fillna(0)


def _security_convenience_loading(security_type: object) -> float:
    sec = _normalise_security_type(security_type)
    if sec in {"Bill", "CMB"}:
        return 1.00
    if sec == "FRN":
        return 0.60
    if sec == "Note":
        return 0.30
    if sec in {"Bond", "TIPS"}:
        return 0.08
    return 0.20


def _security_price_cap(security_type: object, shortness: float) -> float:
    sec = _normalise_security_type(security_type)
    caps = {
        "Bill": 1.00,
        "CMB": 1.00,
        "FRN": 0.96,
        "Note": 0.82,
        "Bond": 0.22,
        "TIPS": 0.20,
    }
    base_cap = caps.get(sec, 0.75)
    if shortness < 0.35:
        base_cap = min(base_cap, 0.24)
    return base_cap


def build_price_convenience_yield_weights(
    supply: pd.DataFrame,
    *,
    liqsub: pd.DataFrame | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    frame = supply[KEY_COLS].drop_duplicates().copy()
    frame["maturity_bucket"] = frame["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
    frame["security_type"] = frame["security_type"].map(_normalise_security_type)
    frame["weight_family"] = "price_convenience_yield"
    frame["base_fixed_weight"] = fixed_weights_for_frame(frame, root=root, ladder_name="fixed_baseline")

    needed = {"month", "iorb_rate", "bill_yield_3mo", "bill_iorb_spread", "sofr_iorb_spread"}
    if liqsub is None or liqsub.empty or not needed.issubset(liqsub.columns):
        frame["liquidity_weight"] = frame["base_fixed_weight"]
        frame["source_status"] = "fallback_fixed_missing_convenience_yield_inputs"
        frame["measurement_assumption"] = "Convenience-yield family fallback: fixed baseline because required monthly rate/spread inputs are unavailable."
        frame["weight_source_detail"] = "requires liqsub iorb_rate, bill_yield_3mo, bill_iorb_spread, and sofr_iorb_spread"
        return frame

    rates = liqsub.copy()
    rates["month"] = pd.to_datetime(rates["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    rates = rates.drop_duplicates("month")
    required_value_cols = ["iorb_rate", "bill_yield_3mo", "bill_iorb_spread", "sofr_iorb_spread"]
    rates["has_convenience_inputs"] = rates[required_value_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    rates["policy_rate_score"] = _rate_score(rates["iorb_rate"])
    # Positive when bills trade below IORB, a first-pass convenience/specialness proxy.
    rates["bill_convenience_score"] = _score_positive_spread(-pd.to_numeric(rates["bill_iorb_spread"], errors="coerce"), scale=0.50)
    # Positive SOFR-IORB spread is treated as a funding-pressure penalty, not money-likeness.
    rates["funding_pressure_score"] = _score_positive_spread(rates["sofr_iorb_spread"], scale=0.25)

    merged = frame.merge(
        rates[["month", "has_convenience_inputs", "policy_rate_score", "bill_convenience_score", "funding_pressure_score"]],
        on="month",
        how="left",
    )
    score_cols = ["policy_rate_score", "bill_convenience_score", "funding_pressure_score"]
    has_inputs = merged["has_convenience_inputs"].fillna(False).astype(bool)
    merged[score_cols] = merged[score_cols].fillna(0)
    shortness = merged["maturity_bucket"].map(lambda b: bucket_shortness_score(b, root))
    loading = merged["security_type"].map(_security_convenience_loading)
    convenience_boost = shortness * loading * (
        0.04 * merged["policy_rate_score"] + 0.12 * merged["bill_convenience_score"]
    )
    funding_penalty = (1.0 - 0.50 * shortness) * (0.06 * merged["funding_pressure_score"])
    candidate = merged["base_fixed_weight"] * (1.0 + convenience_boost - funding_penalty)
    caps = [
        _security_price_cap(row.security_type, float(bucket_shortness_score(row.maturity_bucket, root)))
        for row in merged.itertuples(index=False)
    ]
    merged["liquidity_weight"] = np.minimum(candidate, caps).clip(lower=0, upper=1)
    merged["liquidity_weight"] = merged["liquidity_weight"].where(has_inputs, merged["base_fixed_weight"])
    merged["source_status"] = has_inputs.map({
        True: "first_pass_convenience_yield_model",
        False: "fallback_fixed_no_matching_price_inputs",
    })
    merged["measurement_assumption"] = "Convenience-yield model using bill-IORB specialness and SOFR-IORB funding-pressure penalty; exploratory measurement assumption."
    merged["weight_source_detail"] = "liqsub monthly IORB, bill yield, bill-IORB spread, and SOFR-IORB spread"
    return merged[
        KEY_COLS
        + ["weight_family", "liquidity_weight", "base_fixed_weight", "source_status", "measurement_assumption", "weight_source_detail"]
    ]
