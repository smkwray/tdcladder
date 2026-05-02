from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .contracts import contract_local_path, load_contracts, validate_all
from .hashing import sha256_file
from .io import ensure_parent, write_csv


def copy_upstream_artifacts(
    root: str | Path | None,
    *,
    upstream: str,
    source_root: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Copy expected artifacts from an explicit upstream repo root into data/raw/<upstream>/.

    `source_root` is supplied by the caller. No machine-specific paths are hard-coded.
    """
    src_root = Path(source_root).expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(src_root)

    rows: list[dict[str, Any]] = []
    for contract in load_contracts(root, upstream=upstream):
        rel = contract.get("upstream_relative_path")
        if not rel:
            continue
        src = src_root / str(rel)
        dest = contract_local_path(root, contract)
        status = "missing_source"
        if src.exists():
            ensure_parent(dest)
            if dest.exists() and not overwrite:
                status = "exists_skipped"
            else:
                shutil.copy2(src, dest)
                status = "copied"
        rows.append(
            {
                "name": contract.get("name"),
                "upstream": upstream,
                "source_relative_path": rel,
                "dest_local_path": contract.get("local_path"),
                "status": status,
                "source_sha256": sha256_file(src) if src.exists() else "",
                "dest_sha256": sha256_file(dest) if dest.exists() else "",
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, project_root(root) / f"output/qa/import_gate_copy_{upstream}.csv")
    return result


def validate_project_local_sources(
    root: str | Path | None,
    *,
    upstream: str | None = None,
    required_only: bool = False,
) -> pd.DataFrame:
    df = validate_all(root, upstream=upstream, required_only=required_only)
    write_csv(df, project_root(root) / "output/qa/source_manifest.csv")
    return df
