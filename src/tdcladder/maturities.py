from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config


def _root_key(root: str | Path | None = None) -> str:
    return str(Path(root or ".").resolve())


@cache
def _maturity_config(root_key: str) -> tuple[tuple[tuple[tuple[str, Any], ...], ...], dict[str, str], dict[str, int]]:
    cfg = load_config(Path(root_key), "maturity_buckets.yaml")
    buckets: list[tuple[tuple[str, Any], ...]] = []
    aliases: dict[str, str] = {}
    order: dict[str, int] = {}
    for idx, bucket in enumerate(cfg.get("maturity_buckets", []) or []):
        name = str(bucket["name"])
        bucket_aliases = tuple(str(alias) for alias in (bucket.get("aliases", []) or []))
        rank = int(bucket.get("liquidity_rank", idx + 1))
        normalized_bucket = dict(bucket)
        normalized_bucket["name"] = name
        normalized_bucket["aliases"] = list(bucket_aliases)
        normalized_bucket["liquidity_rank"] = rank
        buckets.append(tuple(normalized_bucket.items()))
        aliases[name] = name
        for alias in bucket_aliases:
            aliases[alias] = name
        order[name] = rank
    return tuple(buckets), aliases, order


def load_maturity_buckets(root: str | Path | None = None) -> list[dict[str, Any]]:
    buckets, _, _ = _maturity_config(_root_key(root))
    return [dict(bucket) for bucket in buckets]


def bucket_alias_map(root: str | Path | None = None) -> dict[str, str]:
    _, aliases, _ = _maturity_config(_root_key(root))
    return dict(aliases)


def bucket_order(root: str | Path | None = None) -> dict[str, int]:
    _, _, order = _maturity_config(_root_key(root))
    return dict(order)


def normalize_bucket(value: object, root: str | Path | None = None) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value).strip()
    _, aliases, _ = _maturity_config(_root_key(root))
    return aliases.get(s, s)


def add_bucket_metadata(df: pd.DataFrame, root: str | Path | None = None) -> pd.DataFrame:
    out = df.copy()
    order = bucket_order(root)
    out["maturity_bucket"] = out["maturity_bucket"].map(lambda x: normalize_bucket(x, root))
    out["maturity_rank"] = out["maturity_bucket"].map(order).fillna(999).astype(int)
    return out


def bucket_shortness_score(bucket: object, root: str | Path | None = None) -> float:
    order = bucket_order(root)
    rank = order.get(normalize_bucket(bucket, root), 999)
    if rank == 999:
        return 0.0
    max_rank = max(order.values()) if order else 11
    return max(0.0, min(1.0, 1.0 - (rank - 1) / max(max_rank - 1, 1)))
