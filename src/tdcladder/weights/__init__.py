from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import load_config
from .fixed import build_fixed_weights
from .holder import build_holder_revealed_weights, build_holder_stock_context_weights
from .price import build_price_convenience_yield_weights, build_price_first_pass_weights
from .turnover import build_turnover_weights


def build_all_weight_families(
    supply: pd.DataFrame,
    *,
    root: str | Path | None = None,
    liqsub: pd.DataFrame | None = None,
    auction_allotments: pd.DataFrame | None = None,
    z1_holder_context: pd.DataFrame | None = None,
    turnover: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = load_config(root, "weight_families.yaml")
    families = cfg.get("weight_families", {}) or {}
    active = {name for name, meta in families.items() if bool((meta or {}).get("active", True))}
    active_fixed = [name for name, meta in families.items() if name in active and (meta or {}).get("class") == "fixed"]

    frames = [
        build_fixed_weights(supply, root=root, ladder_names=active_fixed) if active_fixed else None,
        build_price_first_pass_weights(supply, liqsub=liqsub, root=root) if "price_first_pass" in active else None,
        build_price_convenience_yield_weights(supply, liqsub=liqsub, root=root) if "price_convenience_yield" in active else None,
        build_turnover_weights(supply, turnover=turnover, root=root) if "turnover_duration_adjusted" in active else None,
        build_holder_revealed_weights(supply, auction_allotments=auction_allotments, root=root) if "holder_revealed_first_pass" in active else None,
        build_holder_stock_context_weights(supply, z1_holder_context=z1_holder_context, root=root) if "holder_stock_context_z1" in active else None,
    ]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["liquidity_weight"] = out["liquidity_weight"].clip(lower=0, upper=1)
    return out
