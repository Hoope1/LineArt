"""GUI responsiveness tests."""

import time
from pathlib import Path

from PIL import Image

from src import pipeline


def test_progress_moves(tmp_path, monkeypatch) -> None:
    """Processing 100 images keeps progress callbacks flowing."""
    for i in range(100):
        Image.new("RGB", (32, 32)).save(tmp_path / f"im{i}.png")

    monkeypatch.setattr(pipeline, "process_one", lambda _p, _o, _c, _log: None)
    monkeypatch.setattr(pipeline, "cleanup_models", lambda: None)

    cfg: pipeline.Config = {
        "use_sd": False,
        "save_svg": False,
        "steps": 1,
        "guidance": 1.0,
        "ctrl": 1.0,
        "strength": 0.5,
        "seed": 0,
        "max_long": 64,
        "batch_size": 10,
    }

    prog: list[int] = []

    start = time.time()

    def progress(i: int, total: int, path: Path) -> None:  # noqa: ARG001
        prog.append(i)

    pipeline.process_folder(
        tmp_path, tmp_path, cfg, lambda _: None, lambda: None, None, progress
    )
    duration = time.time() - start

    assert prog and prog[-1] == 100
    assert duration < 5
