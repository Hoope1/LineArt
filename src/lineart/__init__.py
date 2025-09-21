"""Public API for the LineArt processing package."""

from __future__ import annotations

from .config import PipelineConfig
from .constants import (
    DEFAULT_CTRL_SCALE,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_LONG,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STRENGTH,
    IMG_EXTS,
    LOW_VRAM_BYTES,
    LOW_VRAM_MAX_LONG,
    MAX_IMG_SIZE,
    MIN_DISK_SPACE,
    MIN_IMG_SIZE,
    SEQUENTIAL_OFFLOAD_VRAM,
)
from .devices import detect_device, detect_dtype, detect_max_long
from .fs import ensure_dir, find_model_dirs, list_images
from .image_ops import (
    ensure_rgb,
    guided_smooth_if_available,
    postprocess_lineart,
    rescale_edge,
    resize_img,
)
from .models.dexined import get_dexined, load_dexined
from .models.diffusion import load_sd15_lineart, sd_refine
from .prefetch import cleanup_models, prefetch_models
from .processing import process_folder, process_one
from .svg import save_svg_vtracer

__all__ = [
    "PipelineConfig",
    "DEFAULT_CTRL_SCALE",
    "DEFAULT_GUIDANCE",
    "DEFAULT_MAX_LONG",
    "DEFAULT_SEED",
    "DEFAULT_STEPS",
    "DEFAULT_STRENGTH",
    "IMG_EXTS",
    "LOW_VRAM_BYTES",
    "LOW_VRAM_MAX_LONG",
    "MAX_IMG_SIZE",
    "MIN_DISK_SPACE",
    "MIN_IMG_SIZE",
    "SEQUENTIAL_OFFLOAD_VRAM",
    "detect_device",
    "detect_dtype",
    "detect_max_long",
    "ensure_dir",
    "find_model_dirs",
    "list_images",
    "ensure_rgb",
    "guided_smooth_if_available",
    "postprocess_lineart",
    "resize_img",
    "rescale_edge",
    "get_dexined",
    "load_dexined",
    "load_sd15_lineart",
    "sd_refine",
    "cleanup_models",
    "prefetch_models",
    "process_folder",
    "process_one",
    "save_svg_vtracer",
]
