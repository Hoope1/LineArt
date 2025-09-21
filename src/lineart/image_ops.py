"""Image processing utilities used by the pipeline."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .constants import DEFAULT_MAX_LONG, MIN_IMG_SIZE

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


def resize_img(width: int, height: int, *, max_long: int = DEFAULT_MAX_LONG) -> tuple[int, int]:
    """Scale dimensions to multiples of eight, limiting the longest side."""
    if max(width, height) > max_long:
        scale = max_long / max(width, height)
        width, height = int(width * scale), int(height * scale)
    return max(MIN_IMG_SIZE, (width // 8) * 8), max(MIN_IMG_SIZE, (height // 8) * 8)


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert *img* to RGB mode if necessary and detach from the file handle."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img.copy()


def postprocess_lineart(img: Image.Image) -> Image.Image:
    """Binarise *img* for crisp black/white output."""

    def _thr(value: int) -> int:
        return 255 if value > 200 else 0

    return img.convert("L").point(_thr, mode="1").convert("L")


def guided_smooth_if_available(image: Image.Image) -> Image.Image:
    """Apply detail-preserving denoising when OpenCV is available."""
    if cv2 is None:
        return image
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None:
        arr = ximgproc.guidedFilter(guide=arr, src=arr, radius=4, eps=1e-2)
    else:
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=25, sigmaSpace=25)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def rescale_edge(edge: Image.Image, width: int, height: int) -> Image.Image:
    """Resize an edge image back to the original resolution."""
    return edge.resize((width, height), Image.Resampling.NEAREST)
