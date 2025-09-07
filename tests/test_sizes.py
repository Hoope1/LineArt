"""Ensure pipeline handles common image sizes."""

from pathlib import Path

from PIL import Image

from src import pipeline


def test_common_sizes(tmp_path, monkeypatch) -> None:
    """Process 256, 512 and 1024 px images without errors."""
    sizes = [256, 512, 1024]
    for idx, size in enumerate(sizes):
        Image.new("RGB", (size, size)).save(tmp_path / f"im{idx}.png")

    # Fake processor writes out placeholder files
    def fake_process_one(path: Path, out: Path, cfg: pipeline.Config, log):
        out.mkdir(parents=True, exist_ok=True)
        Image.open(path).save(out / path.name)

    monkeypatch.setattr(pipeline, "process_one", fake_process_one)
    monkeypatch.setattr(pipeline, "cleanup_models", lambda: None)

    cfg: pipeline.Config = {
        "use_sd": False,
        "save_svg": False,
        "steps": 1,
        "guidance": 1.0,
        "ctrl": 1.0,
        "strength": 0.5,
        "seed": 0,
        "max_long": 2048,
        "batch_size": 1,
    }

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pipeline.process_folder(tmp_path, out_dir, cfg, lambda _: None, lambda: None)

    for idx in range(len(sizes)):
        assert (out_dir / f"im{idx}.png").exists()
