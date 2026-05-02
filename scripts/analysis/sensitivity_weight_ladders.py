#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(".")
    weights_path = root / "data/clean/liquidity_weights.csv"
    if not weights_path.exists():
        raise FileNotFoundError("Run `tdcladder build` before sensitivity analysis.")
    weights = pd.read_csv(weights_path, parse_dates=["month"])
    table = weights.groupby(["weight_family", "maturity_bucket", "security_type"], as_index=False).agg(
        mean_weight=("liquidity_weight", "mean"),
        min_weight=("liquidity_weight", "min"),
        max_weight=("liquidity_weight", "max"),
        rows=("liquidity_weight", "size"),
    )
    out_dir = root / "output/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "weight_ladder_sensitivity_summary.csv", index=False)
    print("Wrote output/tables/weight_ladder_sensitivity_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
