"""DexiNed detector helpers."""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError
from skimage import exposure, morphology
from skimage.filters import gaussian
from skimage.morphology import binary_closing, remove_small_objects, skeletonize
from skimage.morphology.footprints import square

from ..devices import detect_device
from ..fs import find_model_dirs
from ..image_ops import ensure_rgb, guided_smooth_if_available

logger = logging.getLogger(__name__)

DexiNedDetector = Any  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


@lru_cache(maxsize=1)
def load_dexined(
    device: str | None = None,
    model_id: str = "lllyasviel/Annotators",
    local_dir: Path | None = None,
) -> Any:
    """Return a DexiNed detector loaded on the selected device."""
    try:
        aux_module = importlib.import_module("controlnet_aux")
        Detector = aux_module.DexiNedDetector
    except (ImportError, AttributeError):  # pragma: no cover - optional dependency
        aux_lineart = importlib.import_module("controlnet_aux.lineart")
        Detector = aux_lineart.LineartDetector

        logger.warning(
            "DexiNedDetector not found in controlnet_aux; falling back to LineartDetector",
        )

    target_device = detect_device() if device is None else device
    logger.info("loading DexiNed on %s", target_device)
    try:
        return Detector.from_pretrained(model_id).to(target_device)
    except (RequestsConnectionError, HTTPError, OSError) as exc:
        candidates: list[Path] = []
        if local_dir is not None:
            candidates.append(local_dir)
        candidates.append(Path("models") / "Annotators")
        candidates.extend(find_model_dirs("Annotators"))
        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen or not candidate.exists():
                continue
            seen.add(candidate)
            try:
                return Detector.from_pretrained(candidate).to(target_device)
            except Exception:  # pragma: no cover - best effort
                continue
        msg = (
            "Modell-Download fehlgeschlagen: lllyasviel/Annotators. "
            "Bitte Netzwerk prüfen oder lokalen Pfad nutzen."
        )
        raise RuntimeError(msg) from exc


def get_dexined(
    pil_img: Image.Image,
    *,
    scales: tuple[float, ...] = (1.0, 0.75, 1.25),
    pre_smooth: bool = True,
) -> Image.Image:
    """Run DexiNed on *pil_img* and return a grayscale edge map."""
    import torch

    if cv2 is None:
        logger.warning(
            "OpenCV nicht verfügbar – verwende skimage.gaussian als Fallback",
        )
    detector = load_dexined()
    image = ensure_rgb(pil_img)
    if pre_smooth:
        image = guided_smooth_if_available(image)

    maps: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for scale in scales:
            resized = image.resize(
                (
                    max(64, int(image.width * scale)),
                    max(64, int(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
            edge_map = detector(resized)
            if edge_map.mode != "L":
                edge_map = edge_map.convert("L")
            edge_map = edge_map.resize((image.width, image.height), Image.Resampling.BILINEAR)
            maps.append(np.array(edge_map, dtype=np.float32) / 255.0)

    edge = np.maximum.reduce(maps)
    lo_hi = (
        float(np.percentile(edge, 5)),
        float(np.percentile(edge, 99)),
    )
    edge = exposure.rescale_intensity(edge, in_range=lo_hi)  # pyright: ignore[reportArgumentType]
    if cv2 is not None:
        edge = cv2.GaussianBlur(edge, (0, 0), 0.7)
    else:
        edge = gaussian(edge, sigma=0.7, preserve_range=True)

    threshold = np.clip(np.percentile(edge, 50), 0.25, 0.65)
    mask = edge <= threshold
    mask = binary_closing(mask, square(2))
    mask = remove_small_objects(mask, min_size=16)
    thin = skeletonize(mask)
    thin = morphology.remove_small_holes(thin, area_threshold=16)

    output = np.where(thin, 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="L")
