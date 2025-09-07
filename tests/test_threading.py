"""Threading robustness tests."""

import threading
import time
from pathlib import Path

from PIL import Image

from src import pipeline


def test_start_stop_multiple(tmp_path, monkeypatch) -> None:
    """Rapid start/stop cycles finish without deadlock."""
    Image.new("RGB", (64, 64)).save(tmp_path / "im.png")

    def fake_process_one(path: Path, out: Path, cfg: pipeline.Config, log):
        time.sleep(0.05)

    cleanup_calls = {"n": 0}

    def fake_cleanup() -> None:
        cleanup_calls["n"] += 1

    monkeypatch.setattr(pipeline, "process_one", fake_process_one)
    monkeypatch.setattr(pipeline, "cleanup_models", fake_cleanup)

    cfg: pipeline.Config = {
        "use_sd": False,
        "save_svg": False,
        "steps": 1,
        "guidance": 1.0,
        "ctrl": 1.0,
        "strength": 0.5,
        "seed": 0,
        "max_long": 128,
        "batch_size": 1,
    }

    def run_once() -> None:
        stop_event = threading.Event()
        t = threading.Thread(
            target=pipeline.process_folder,
            args=(tmp_path, tmp_path, cfg, lambda _: None, lambda: None, stop_event),
        )
        t.start()
        stop_event.set()
        t.join(timeout=2)
        assert not t.is_alive()

    for _ in range(3):
        run_once()

    assert cleanup_calls["n"] == 3
