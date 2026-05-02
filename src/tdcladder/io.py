from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: str | Path, *, parse_dates: Iterable[str] | None = None, **kwargs) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, **kwargs)
    for col in parse_dates or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    p = ensure_parent(path)
    df.to_csv(p, index=False, **kwargs)
    return p


def maybe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return read_csv(p, **kwargs)
