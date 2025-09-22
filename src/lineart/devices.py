"""Device and dtype detection helpers."""

from __future__ import annotations

from contextlib import suppress
from functools import lru_cache
from typing import TYPE_CHECKING

from .constants import DEFAULT_MAX_LONG, LOW_VRAM_BYTES, LOW_VRAM_MAX_LONG

if TYPE_CHECKING:
    import torch


@lru_cache(maxsize=1)
def detect_device() -> str:
    """Return the best available torch device string."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=3)
def detect_dtype(device: str) -> torch.dtype:
    """Return the preferred torch dtype for *device*."""
    import torch

    if device == "cuda":
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
        return torch.bfloat16 if is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    cpu_bf16_supported = getattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)
    return torch.bfloat16 if cpu_bf16_supported() else torch.float32


@lru_cache(maxsize=1)
def detect_max_long() -> int:
    """Return a recommended ``max_long`` value based on available VRAM."""
    with suppress(Exception):
        import torch

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory
            if vram < LOW_VRAM_BYTES:
                return LOW_VRAM_MAX_LONG
    return DEFAULT_MAX_LONG
