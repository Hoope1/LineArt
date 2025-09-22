"""Tests for prefetching Hugging Face models."""

from __future__ import annotations

from pathlib import Path

import huggingface_hub

import lineart.pipeline as pipeline


def test_prefetch_models_download(monkeypatch) -> None:
    """prefetch_models uses huggingface_hub to download repositories."""
    calls: list[tuple[str, Path]] = []

    def fake_snapshot(repo_id: str, local_dir: Path, **_kwargs) -> None:  # noqa: D401
        """Capture download calls."""
        calls.append((repo_id, Path(local_dir)))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(pipeline, "load_dexined", lambda **_k: None)
    monkeypatch.setattr("lineart.models.dexined.load_dexined", lambda **_k: None)
    monkeypatch.setattr("lineart.prefetch.load_dexined", lambda **_k: None)
    monkeypatch.setattr(pipeline, "load_sd15_lineart", lambda **_k: None)
    monkeypatch.setattr("lineart.models.diffusion.load_sd15_lineart", lambda **_k: None)
    monkeypatch.setattr("lineart.prefetch.load_sd15_lineart", lambda **_k: None)

    pipeline.prefetch_models(lambda _msg: None)

    expected = {
        ("stable-diffusion-v1-5/stable-diffusion-v1-5", Path("models") / "sd15"),
        ("lllyasviel/Annotators", Path("models") / "Annotators"),
        (
            "lllyasviel/control_v11p_sd15_lineart",
            Path("models") / "control_v11p_sd15_lineart",
        ),
    }
    assert set(calls) == expected
