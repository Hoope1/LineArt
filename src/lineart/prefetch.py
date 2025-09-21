"""Model prefetch and cleanup helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .models.dexined import load_dexined
from .models.diffusion import load_sd15_lineart


def _download_repo(repo_id: str, target: Path) -> None:
    """Download *repo_id* into *target* using ``huggingface_hub``."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id,
            local_dir=target,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"Download fehlgeschlagen: {repo_id}") from exc


# pragma: no cover
def prefetch_models(log: Callable[[str], None]) -> None:
    """Download all required models ahead of time."""
    log("Lade Modelle vom Hub … (einmalig)")
    _download_repo(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        Path("models") / "sd15",
    )
    _download_repo(
        "lllyasviel/Annotators",
        Path("models") / "Annotators",
    )
    _download_repo(
        "lllyasviel/control_v11p_sd15_lineart",
        Path("models") / "control_v11p_sd15_lineart",
    )
    _ = load_dexined(device="cpu", local_dir=Path("models") / "Annotators")
    _ = load_sd15_lineart(
        local_model_dir=Path("models") / "sd15",
        local_controlnet_dir=Path("models") / "control_v11p_sd15_lineart",
    )
    log("Modelle vorhanden.\n")


def cleanup_models() -> None:
    """Release loaded models and free GPU memory."""
    import gc

    import torch

    load_dexined.cache_clear()
    load_sd15_lineart.cache_clear()
    gc.collect()
    if torch.cuda.is_available():  # pragma: no cover - hardware specific
        torch.cuda.empty_cache()
