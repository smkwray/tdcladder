from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def fixture_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    repo = Path(__file__).resolve().parents[1]
    for rel in ["config", "src"]:
        shutil.copytree(repo / rel, root / rel)
    shutil.copytree(repo / "tests/fixtures/data", root / "data")
    (root / "output").mkdir(parents=True, exist_ok=True)
    return root
