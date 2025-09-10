"""Error recovery tests."""

from pathlib import Path

from PIL import Image

from src import pipeline


def test_recovery_after_failure(tmp_path, monkeypatch) -> None:
    """After exception processing can restart cleanly."""
    Image.new("RGB", (64, 64)).save(tmp_path / "im.png")

    def failing_process(path: Path, out: Path, cfg: pipeline.Config, _log):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "process_one", failing_process)
    monkeypatch.setattr(pipeline, "cleanup_models", lambda: None)

    logs: list[str] = []
    cfg: pipeline.Config = {
        "use_sd": False,
        "save_svg": False,
        "steps": 1,
        "guidance": 1.0,
        "ctrl": 1.0,
        "strength": 0.5,
        "seed": 0,
        "max_long": 64,
        "batch_size": 1,
    }

    pipeline.process_folder(tmp_path, tmp_path, cfg, logs.append, lambda: None)
    assert any("FEHLER" in m for m in logs)

    # Second run with working process_one
    monkeypatch.setattr(pipeline, "process_one", lambda _p, _o, _c, _log: None)
    logs2: list[str] = []
    pipeline.process_folder(tmp_path, tmp_path, cfg, logs2.append, lambda: None)
    assert any("ALLE BILDER ERLEDIGT" in m for m in logs2)
