"""Tests for folder processing utilities."""

from types import SimpleNamespace

from PIL import Image

from src import pipeline


def test_process_folder_creates_output(tmp_path, monkeypatch) -> None:
    """process_folder creates the output directory if missing."""
    inp = tmp_path / "in"
    inp.mkdir()
    Image.new("RGB", (64, 64)).save(inp / "a.png")

    out = tmp_path / "out"

    monkeypatch.setattr(pipeline, "process_one", lambda *_a, **_k: None)
    monkeypatch.setattr("src.lineart.processing.process_one", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "cleanup_models", lambda: None)
    disk = SimpleNamespace(free=pipeline.MIN_DISK_SPACE + 1)
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _: disk)

    cfg = pipeline.PipelineConfig(
        use_sd=False,
        save_svg=False,
        steps=1,
        guidance=1.0,
        ctrl=1.0,
        strength=0.5,
        seed=0,
        max_long=64,
        batch_size=1,
    )

    pipeline.process_folder(inp, out, cfg, lambda _: None, lambda: None)
    assert out.exists()
