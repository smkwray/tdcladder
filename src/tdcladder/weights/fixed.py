from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config
from ..maturities import normalize_bucket

KEY_COLS = ["month", "security_type", "maturity_bucket"]


def _normalise_security_type(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    raw = str(value).strip()
    lookup = {
        "bill": "Bill",
        "bills": "Bill",
        "cmb": "CMB",
        "cash management bill": "CMB",
        "frn": "FRN",
        "floating rate note": "FRN",
        "note": "Note",
        "notes": "Note",
        "bond": "Bond",
        "bonds": "Bond",
        "tips": "TIPS",
        "tip": "TIPS",
    }
    return lookup.get(raw.lower(), raw or "Unknown")


def _clip_weight(value: float, sec_type: str, ladder: dict[str, Any]) -> float:
    caps = ladder.get("security_type_caps", {}) or {}
    floors = ladder.get("security_type_floors", {}) or {}
    cap = float(caps.get(sec_type, caps.get("Unknown", 1.0)))
    floor = floors.get(sec_type)
    if floor is not None:
        value = max(value, float(floor))
    return max(0.0, min(1.0, min(value, cap)))


def fixed_weight_for(
    maturity_bucket: object,
    security_type: object,
    *,
    root: str | Path | None = None,
    ladder_name: str = "fixed_baseline",
) -> float:
    cfg = load_config(root, "baseline_weights.yaml")
    ladder = cfg["weight_ladders"][ladder_name]
    bucket = normalize_bucket(maturity_bucket, root)
    sec_type = _normalise_security_type(security_type)
    base = float(ladder.get("maturity_bucket_weights", {}).get(bucket, 0.0))
    multiplier = float(ladder.get("security_type_multipliers", {}).get(sec_type, ladder.get("security_type_multipliers", {}).get("Unknown", 1.0)))
    return _clip_weight(base * multiplier, sec_type, ladder)


def fixed_weights_for_frame(
    frame: pd.DataFrame,
    *,
    root: str | Path | None = None,
    ladder_name: str = "fixed_baseline",
) -> pd.Series:
    """Return fixed weights for a normalized or raw security/bucket frame.

    This is the batch equivalent of `fixed_weight_for`; it loads the YAML ladder
    once and avoids thousands of config reads during real-data builds.
    """
    cfg = load_config(root, "baseline_weights.yaml")
    ladder = cfg["weight_ladders"][ladder_name]
    buckets = frame["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
    sec_types = frame["security_type"].map(_normalise_security_type)
    bucket_weights = ladder.get("maturity_bucket_weights", {}) or {}
    multipliers = ladder.get("security_type_multipliers", {}) or {}
    caps = ladder.get("security_type_caps", {}) or {}
    floors = ladder.get("security_type_floors", {}) or {}

    unknown_multiplier = float(multipliers.get("Unknown", 1.0))
    unknown_cap = float(caps.get("Unknown", 1.0))
    base = buckets.map(lambda b: float(bucket_weights.get(b, 0.0))).astype(float)
    multiplier = sec_types.map(lambda s: float(multipliers.get(s, unknown_multiplier))).astype(float)
    values = base * multiplier
    floor_values = sec_types.map(lambda s: floors.get(s))
    floor_mask = floor_values.notna()
    if floor_mask.any():
        values = pd.Series(np.where(floor_mask, np.maximum(values, floor_values.astype(float)), values), index=frame.index)
    cap_values = sec_types.map(lambda s: float(caps.get(s, unknown_cap))).astype(float)
    return pd.Series(np.minimum(values, cap_values), index=frame.index).clip(lower=0, upper=1)


def build_fixed_weights(
    supply: pd.DataFrame,
    *,
    root: str | Path | None = None,
    ladder_names: list[str] | None = None,
) -> pd.DataFrame:
    cfg = load_config(root, "baseline_weights.yaml")
    ladders = cfg.get("weight_ladders", {})
    names = ladder_names or list(ladders.keys())
    rows = []
    for ladder_name in names:
        if ladder_name not in ladders:
            raise KeyError(f"Unknown fixed ladder {ladder_name!r}")
        frame = supply[KEY_COLS].drop_duplicates().copy()
        frame["maturity_bucket"] = frame["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
        frame["security_type"] = frame["security_type"].map(_normalise_security_type)
        frame["weight_family"] = ladder_name
        frame["liquidity_weight"] = fixed_weights_for_frame(frame, root=root, ladder_name=ladder_name)
        frame["base_fixed_weight"] = frame["liquidity_weight"]
        frame["source_status"] = "configured_fixed_assumption"
        frame["measurement_assumption"] = ladders[ladder_name].get("description", "fixed measurement assumption")
        frame["weight_source_detail"] = "config/baseline_weights.yaml"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
