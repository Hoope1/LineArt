"""Edge case image handling tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

import lineart.pipeline as pipeline

CFG = pipeline.PipelineConfig(
    use_sd=False,
    save_svg=False,
    steps=1,
    guidance=1.0,
    ctrl=1.0,
    strength=0.5,
    seed=0,
    max_long=2048,
    batch_size=1,
)


def _run(path: Path, tmp_path: Path) -> list[str]:
    logs: list[str] = []
    pipeline.process_one(path, tmp_path, CFG, logs.append)
    return logs


def test_too_small_image(tmp_path: Path) -> None:
    """Images below minimum size are rejected."""
    p = tmp_path / "small.png"
    Image.new("RGB", (1, 1)).save(p)
    logs = _run(p, tmp_path)
    assert any("ungültige Abmessungen" in s for s in logs)


def test_too_large_image(tmp_path: Path) -> None:
    """Images above maximum size are rejected."""
    p = tmp_path / "large.png"
    Image.new("RGB", (10000, 10000)).save(p)
    logs = _run(p, tmp_path)
    assert any("ungültige Abmessungen" in s for s in logs)


def test_corrupt_image(tmp_path: Path) -> None:
    """Corrupt image files log an error and are skipped."""
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image")
    logs = _run(p, tmp_path)
    assert any("FEHLER" in s for s in logs)
