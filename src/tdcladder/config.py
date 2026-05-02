from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root(root: str | Path | None = None) -> Path:
    """Return the project root, defaulting to the current working directory."""
    return Path(root or ".").resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in YAML file {path}")
    return data


def config_path(root: str | Path | None, *parts: str) -> Path:
    return project_root(root) / "config" / Path(*parts)


def load_config(root: str | Path | None, filename: str) -> dict[str, Any]:
    return load_yaml(config_path(root, filename))
