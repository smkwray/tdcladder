#!/usr/bin/env python
from __future__ import annotations

from tdcladder.build import build_all

if __name__ == "__main__":
    outputs = build_all(".")
    for name, df in outputs.items():
        print(f"{name}: {len(df):,} rows")
