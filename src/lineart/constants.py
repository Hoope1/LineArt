"""Shared constants for the Dexi LineArt pipeline."""

from __future__ import annotations

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

MIN_IMG_SIZE = 64
MAX_IMG_SIZE = 4096
LOW_VRAM_MAX_LONG = 640
DEFAULT_MAX_LONG = 896
LOW_VRAM_BYTES = 5 * 1024**3  # 5 GiB
SEQUENTIAL_OFFLOAD_VRAM = 6 * 1024**3  # 6 GiB
MIN_DISK_SPACE = 100 * 1024 * 1024  # 100 MiB
DEFAULT_STEPS = 32
DEFAULT_GUIDANCE = 6.0
DEFAULT_CTRL_SCALE = 1.0
DEFAULT_STRENGTH = 0.70
DEFAULT_SEED = 42
