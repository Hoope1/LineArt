"""Backwards compatible facade for the LineArt pipeline."""

from __future__ import annotations

import shutil as _shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .lineart import (
    DEFAULT_CTRL_SCALE,
    DEFAULT_GUIDANCE,
    DEFAULT_MAX_LONG,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STRENGTH,
    IMG_EXTS,
    MIN_DISK_SPACE,
    MIN_IMG_SIZE,
    PipelineConfig,
    cleanup_models,
    detect_device,
    detect_dtype,
    detect_max_long,
    ensure_dir,
    ensure_rgb,
    find_model_dirs,
    get_dexined,
    guided_smooth_if_available,
    list_images,
    load_dexined,
    load_sd15_lineart,
    postprocess_lineart,
    prefetch_models,
    rescale_edge,
    resize_img,
    save_svg_vtracer,
    sd_refine,
)
from .lineart import (
    process_folder as _process_folder_impl,
)
from .lineart import (
    process_one as _process_one_impl,
)
from .lineart.models.diffusion import _autocast_context as diffusion_autocast_context

Config = PipelineConfig
shutil = _shutil

__all__ = [
    "DEFAULT_CTRL_SCALE",
    "DEFAULT_GUIDANCE",
    "DEFAULT_MAX_LONG",
    "DEFAULT_SEED",
    "DEFAULT_STEPS",
    "DEFAULT_STRENGTH",
    "IMG_EXTS",
    "MIN_DISK_SPACE",
    "MIN_IMG_SIZE",
    "PipelineConfig",
    "Config",
    "cleanup_models",
    "detect_device",
    "detect_dtype",
    "detect_max_long",
    "ensure_dir",
    "ensure_rgb",
    "find_model_dirs",
    "get_dexined",
    "guided_smooth_if_available",
    "list_images",
    "load_dexined",
    "load_sd15_lineart",
    "sd_refine",
    "postprocess_lineart",
    "prefetch_models",
    "process_folder",
    "process_one",
    "resize_img",
    "rescale_edge",
    "save_svg_vtracer",
    "shutil",
    "_autocast_context",
]


def process_folder_compat(
    inp: Path,
    out: Path,
    cfg: PipelineConfig | dict[str, Any],
    log: Callable[[str], None],
    done_cb: Callable[[], None] | None = None,
    stop_event: Any | None = None,
    progress_cb: Callable[[int, int, Path], None] | None = None,
) -> None:
    """Compatibility wrapper that also accepts mapping configs."""
    config = cfg if isinstance(cfg, PipelineConfig) else PipelineConfig.from_mapping(cfg)
    _process_folder_impl(
        inp,
        out,
        config,
        log,
        progress_cb=progress_cb,
        done_cb=done_cb,
        stop_event=stop_event,
    )


def process_one_compat(
    path: Path,
    out_dir: Path,
    cfg: PipelineConfig | dict[str, Any],
    log: Callable[[str], None],
) -> None:
    """Compatibility wrapper that also accepts mapping configs."""
    config = cfg if isinstance(cfg, PipelineConfig) else PipelineConfig.from_mapping(cfg)
    _process_one_impl(path, out_dir, config, log)


# keep legacy names
process_folder = process_folder_compat
process_one = process_one_compat
_autocast_context = diffusion_autocast_context
