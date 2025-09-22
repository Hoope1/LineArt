"""Ensure dependency declarations are in sync."""

import importlib
from pathlib import Path

try:
    tomllib = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover
    tomllib = importlib.import_module("tomli")


def test_requirements_match_pyproject() -> None:
    """All declared dependencies should appear in requirements.txt."""
    data = tomllib.loads(Path("pyproject.toml").read_text())
    deps = set(data["project"]["dependencies"])
    reqs = {
        line.strip()
        for line in Path("requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }

    missing = deps - reqs
    assert not missing, f"Missing dependencies in requirements.txt: {sorted(missing)}"
