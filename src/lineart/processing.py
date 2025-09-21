"""High level orchestration for batch processing."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from .config import PipelineConfig
from .constants import MAX_IMG_SIZE, MIN_DISK_SPACE, MIN_IMG_SIZE
from .fs import ensure_dir, list_images
from .image_ops import ensure_rgb
from .models.dexined import get_dexined
from .models.diffusion import sd_refine
from .prefetch import cleanup_models
from .svg import save_svg_vtracer

logger = logging.getLogger(__name__)


def process_one(path: Path, out_dir: Path, cfg: PipelineConfig, log: Callable[[str], None]) -> None:
    """Process a single image and write outputs to *out_dir*."""
    start = time.perf_counter()
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            src = ensure_rgb(img)
    except Exception as exc:
        log(f"FEHLER: {path.name} – {exc}")
        return

    if (
        src.width < MIN_IMG_SIZE
        or src.height < MIN_IMG_SIZE
        or src.width > MAX_IMG_SIZE
        or src.height > MAX_IMG_SIZE
    ):
        log(f"FEHLER: {path.name} hat ungültige Abmessungen")
        return

    edges = get_dexined(src)
    ensure_dir(out_dir)
    prep_dir = ensure_dir(out_dir / "preprocessed")
    out_edges = prep_dir / f"{path.stem}_dexi.png"
    edges.save(out_edges)

    if cfg.use_sd:
        try:
            refined, bw = sd_refine(src, edges, cfg)
        except Exception as exc:
            log(f"FEHLER bei SD-Refinement: {exc}")
            return
        ref_path = out_dir / f"{path.stem}_refined.png"
        bw_path = out_dir / f"{path.stem}_refined_bw.png"
        refined.save(ref_path)
        bw.save(bw_path)

        if cfg.save_svg:
            svg_target = out_dir / f"{path.stem}_refined_bw.svg"
            if not save_svg_vtracer(bw_path, svg_target):
                log("SVG-Export fehlgeschlagen – ist 'vtracer' installiert?")
    duration = time.perf_counter() - start
    log(f"{path.name} erledigt in {duration:.1f}s")


# pragma: no cover


def process_folder(
    inp: Path,
    out: Path,
    cfg: PipelineConfig,
    log: Callable[[str], None],
    progress_cb: Callable[[int, int, Path], None] | None = None,
    done_cb: Callable[[], None] | None = None,
    stop_event: Any | None = None,
) -> None:
    """Process *inp* recursively and write results to *out*."""
    images = _collect_images(inp, log, done_cb)
    if images is None:
        return
    if not _check_disk_space(out, log, done_cb):
        return

    try:
        _run_batches(images, out, cfg, log, progress_cb, done_cb, stop_event)
    finally:
        cleanup_models()


def _collect_images(
    inp: Path,
    log: Callable[[str], None],
    done_cb: Callable[[], None] | None,
) -> list[Path] | None:
    """Return images inside *inp* or ``None`` if processing should abort."""
    try:
        images = list_images(inp)
    except FileNotFoundError:
        log("FEHLER: Eingabeordner nicht gefunden")
        _notify_done(done_cb)
        return None
    if not images:
        log("Keine Eingabebilder gefunden.")
        _notify_done(done_cb)
        return None
    return images


def _check_disk_space(
    out: Path,
    log: Callable[[str], None],
    done_cb: Callable[[], None] | None,
) -> bool:
    """Ensure the output directory exists and has sufficient free space."""
    ensure_dir(out)
    if shutil.disk_usage(out).free < MIN_DISK_SPACE:
        log("FEHLER: Zu wenig Speicherplatz im Ausgabeverzeichnis")
        _notify_done(done_cb)
        return False
    return True


def _run_batches(
    images: list[Path],
    out: Path,
    cfg: PipelineConfig,
    log: Callable[[str], None],
    progress_cb: Callable[[int, int, Path], None] | None,
    done_cb: Callable[[], None] | None,
    stop_event: Any | None,
) -> None:
    """Process *images* in batches while honouring callbacks and stop requests."""
    import torch

    batch_size = max(1, cfg.batch_size)
    total = len(images)
    index = 0
    while index < total:
        batch = images[index : index + batch_size]
        try:
            for img_path in batch:
                if stop_event is not None and stop_event.is_set():
                    log("Verarbeitung abgebrochen.")
                    _notify_done(done_cb)
                    return
                process_one(img_path, out, cfg, log)
                index += 1
                if progress_cb:
                    progress_cb(index, total, img_path)
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            if _should_retry_oom(exc, batch_size):
                log("GPU OOM – reduziere Batch-Size auf 1 und versuche erneut …")
                batch_size = 1
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.empty_cache()
                continue
            log(f"FEHLER: {exc}")
            index += len(batch)
    log("\nALLE BILDER ERLEDIGT.")
    _notify_done(done_cb)


def _should_retry_oom(exc: RuntimeError, batch_size: int) -> bool:
    """Return ``True`` when *exc* indicates a recoverable GPU OOM."""
    return "gpu out of memory" in str(exc).lower() and batch_size > 1


def _notify_done(done_cb: Callable[[], None] | None) -> None:
    """Invoke *done_cb* when provided."""
    if done_cb is not None:
        done_cb()
