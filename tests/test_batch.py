"""Tests for batch processing logic."""

from pathlib import Path

from PIL import Image

from src import pipeline


def test_batch_fallback(tmp_path, monkeypatch) -> None:
    """GPU OOM reduces batch size to one and continues."""
    for i in range(2):
        Image.new("RGB", (64, 64)).save(tmp_path / f"im{i}.png")

    calls = {"count": 0}

    def fake_process_one(path: Path, out_dir: Path, cfg: pipeline.Config, log):
        if calls["count"] == 0:
            calls["count"] += 1
            raise RuntimeError("GPU out of memory")
        calls["count"] += 1

    monkeypatch.setattr(pipeline, "process_one", fake_process_one)

    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    cfg: pipeline.Config = {
        "use_sd": False,
        "save_svg": False,
        "steps": 1,
        "guidance": 1.0,
        "ctrl": 1.0,
        "strength": 0.5,
        "seed": 0,
        "max_long": 64,
        "batch_size": 2,
    }
    pipeline.process_folder(tmp_path, tmp_path, cfg, log, lambda: None)

    assert any("reduziere Batch-Size" in m for m in logs)
    assert calls["count"] == 3
