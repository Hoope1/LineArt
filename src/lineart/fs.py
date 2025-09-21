"""Filesystem helpers for locating and preparing assets."""

from __future__ import annotations

from pathlib import Path

from .constants import IMG_EXTS


def find_model_dirs(name: str, root: Path | None = None) -> list[Path]:
    """Return directories named *name* within *root* and its subfolders."""
    base = Path(__file__).resolve().parent.parent if root is None else root
    return [p for p in base.rglob(name) if p.is_dir()]


def list_images(folder: Path) -> list[Path]:
    """Return all image paths in *folder* with a supported extension."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def ensure_dir(path: Path) -> Path:
    """Create directory *path* if needed and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
