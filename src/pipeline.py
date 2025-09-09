#!/usr/bin/env python3
"""Core pipeline for DexiNed → SD 1.5 + ControlNet(Lineart).

This module contains all heavy computations so the GUI in ``main.py`` stays
lightweight.
"""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = cast(Any, None)

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage import exposure, morphology
from skimage.filters import gaussian
from skimage.morphology import (
    binary_closing,
    remove_small_objects,
    skeletonize,
)
from skimage.morphology.footprints import square

if TYPE_CHECKING:
    import torch
    from diffusers import StableDiffusionControlNetImg2ImgPipeline

logger = logging.getLogger(__name__)

# Supported image extensions.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Common constants
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
DEFAULT_BATCH_SIZE = 1


class Config(TypedDict):
    """Configuration options for processing."""

    use_sd: bool
    save_svg: bool
    steps: int
    guidance: float
    ctrl: float
    strength: float
    seed: int
    max_long: int
    batch_size: int


def find_model_dirs(name: str, root: Path | None = None) -> list[Path]:
    """Return directories named *name* within *root* and its subfolders.

    Args:
        name: Directory name to search for.
        root: Optional root directory. Defaults to the project root.

    Returns:
        list[Path]: All matching directories.

    Raises:
        None

    """
    base = Path(__file__).resolve().parent.parent if root is None else root
    return [p for p in base.rglob(name) if p.is_dir()]


@lru_cache(maxsize=1)
def detect_device() -> str:
    """Return the best available torch device string.

    The result is cached because hardware availability will not change during
    a single program run.

    Returns:
        str: ``"cuda"``, ``"mps"`` or ``"cpu"`` depending on availability.

    Raises:
        None

    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=3)
def detect_dtype(device: str) -> torch.dtype:
    """Return preferred torch dtype for *device*.

    The result is cached per device type to avoid repeated feature detection.

    Args:
        device: Torch device string.

    Returns:
        torch.dtype: Appropriate dtype for the given device.

    Raises:
        None

    """
    import torch

    if device == "cuda":
        return (
            torch.bfloat16
            if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            else torch.float16
        )
    if device == "mps":
        return torch.float16
    cpu_bf16 = getattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)()
    return torch.bfloat16 if cpu_bf16 else torch.float32


@lru_cache(maxsize=1)
def detect_max_long() -> int:
    """Return recommended ``max_long`` based on available VRAM.

    The calculation is cached because the amount of GPU memory is constant
    during runtime.

    Returns:
        int: ``LOW_VRAM_MAX_LONG`` if GPU memory is below ``LOW_VRAM_BYTES``
        else ``DEFAULT_MAX_LONG``.

    Raises:
        None

    """
    try:
        import torch

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory
            if vram < LOW_VRAM_BYTES:
                return LOW_VRAM_MAX_LONG
    except Exception:  # pragma: no cover - best effort
        pass
    return DEFAULT_MAX_LONG


def list_images(folder: Path) -> list[Path]:
    """Return all image paths in *folder* with a supported extension.

    Args:
        folder: Directory to scan.

    Returns:
        list[Path]: Sorted list of image paths.

    Raises:
        None

    """
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def resize_img(w: int, h: int, max_long: int = DEFAULT_MAX_LONG) -> tuple[int, int]:
    """Scale dimensions to multiples of eight, limiting the longest side.

    Args:
        w: Width in pixels.
        h: Height in pixels.
        max_long: Maximum size of the longest edge.

    Returns:
        tuple[int, int]: Resized width and height.

    Raises:
        None

    """
    if max(w, h) > max_long:
        scale = max_long / max(w, h)
        w, h = int(w * scale), int(h * scale)
    return max(MIN_IMG_SIZE, (w // 8) * 8), max(MIN_IMG_SIZE, (h // 8) * 8)


def ensure_dir(p: Path) -> Path:
    """Create directory *p* if needed and return the path.

    Args:
        p: Directory to create.

    Returns:
        Path: The created directory path.

    Raises:
        None

    """
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert *img* to RGB mode if necessary.

    Args:
        img: Input image.

    Returns:
        Image.Image: RGB image.

    Raises:
        None

    """
    return img.convert("RGB") if img.mode != "RGB" else img


def postprocess_lineart(img: Image.Image) -> Image.Image:
    """Binarise *img* for crisp black/white output.

    Args:
        img: Grayscale or RGB image.

    Returns:
        Image.Image: Thresholded single-channel image.

    Raises:
        None

    """

    def _thr(v: int) -> int:
        return 255 if v > 200 else 0

    return img.convert("L").point(_thr, mode="1").convert("L")


def save_svg_vtracer(png_path: Path, svg_path: Path) -> bool:
    """Convert *png_path* to SVG via ``vtracer`` and save to *svg_path*.

    Args:
        png_path: Input PNG file.
        svg_path: Target SVG path.

    Returns:
        bool: ``True`` if conversion succeeded.

    Raises:
        None

    """
    try:
        subprocess.run(
            [
                "vtracer",
                "--input",
                str(png_path),
                "--output",
                str(svg_path),
                "--mode",
                "polygon",
                "--filter-speckle",
                "8",
                "--hierarchical",
                "true",
            ],
            check=True,
            capture_output=True,
        )
        return True
    except FileNotFoundError as exc:
        logger.error("vtracer CLI nicht gefunden: %s", exc)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external tool
        err = exc.stderr.decode().strip() if exc.stderr else exc
        logger.error("vtracer fehlgeschlagen: %s", err)
    except Exception as exc:  # pragma: no cover - unexpected
        logger.error("Unerwarteter SVG-Exportfehler: %s", exc)
    return False


# ---------- DexiNed (controlnet-aux) ----------


@lru_cache(maxsize=1)
def load_dexined(
    device: str | None = None,
    model_id: str = "lllyasviel/Annotators",
    local_dir: Path | None = None,
) -> Any:
    """Load the DexiNed edge detector on the requested *device*.

    Falls back to :class:`~controlnet_aux.lineart.LineartDetector` if the
    installed ``controlnet_aux`` package lacks ``DexiNedDetector``.

    Args:
        device: Optional torch device string.
        model_id: Hugging Face model identifier.
        local_dir: Optional local model directory used as fallback.

    Returns:
        DexiNedDetector | LineartDetector: Loaded detector instance.

    Raises:
        RuntimeError: If the model cannot be downloaded or loaded locally.

    """
    try:
        from controlnet_aux import DexiNedDetector as Detector
    except ImportError:  # pragma: no cover - optional dependency
        from controlnet_aux.lineart import LineartDetector as Detector

        logger.warning(
            "DexiNedDetector not found in controlnet_aux; "
            "falling back to LineartDetector",
        )

    dev = detect_device() if device is None else device
    logger.info("loading DexiNed on %s", dev)
    try:
        return Detector.from_pretrained(model_id).to(dev)
    except (RequestsConnectionError, HTTPError, OSError) as exc:
        candidates: list[Path] = []
        if local_dir is not None:
            candidates.append(local_dir)
        candidates.append(Path("models") / "Annotators")
        candidates.extend(find_model_dirs("Annotators"))
        seen: set[Path] = set()
        for cand in candidates:
            cand = cand.resolve()
            if cand in seen or not cand.exists():
                continue
            seen.add(cand)
            try:
                return Detector.from_pretrained(cand).to(dev)
            except Exception:  # pragma: no cover - best effort
                continue
        msg = (
            "Modell-Download fehlgeschlagen: lllyasviel/Annotators. "
            "Bitte Netzwerk prüfen oder lokalen Pfad nutzen."
        )
        raise RuntimeError(msg) from exc


def guided_smooth_if_available(pil_img: Image.Image) -> Image.Image:
    """Apply detail-preserving denoising.

    Args:
        pil_img: Input RGB image.

    Returns:
        Image.Image: Smoothed image.

    Raises:
        None

    """
    if cv2 is None:
        return pil_img
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None:
        arr = ximgproc.guidedFilter(guide=arr, src=arr, radius=4, eps=1e-2)
    else:
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=25, sigmaSpace=25)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def get_dexined(
    pil_img: Image.Image,
    scales: tuple[float, ...] = (1.0, 0.75, 1.25),
    pre_smooth: bool = True,
) -> Image.Image:
    """Run DexiNed on *pil_img* and return a grayscale edge map.

    Args:
        pil_img: Input image.
        scales: Rescaling factors for multi-scale inference.
        pre_smooth: Whether to apply denoising before detection.

    Returns:
        Image.Image: Edge map in ``L`` mode.

    Raises:
        None

    """
    import torch

    if cv2 is None:
        logger.warning(
            "OpenCV nicht verfügbar – verwende skimage.gaussian als Fallback"
        )
    det = load_dexined()
    img = ensure_rgb(pil_img)
    if pre_smooth:
        img = guided_smooth_if_available(img)

    maps: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for s in scales:
            img_s = img.resize(
                (max(64, int(img.width * s)), max(64, int(img.height * s))),
                Image.Resampling.LANCZOS,
            )
            e = det(img_s)
            if e.mode != "L":
                e = e.convert("L")
            e = e.resize((img.width, img.height), Image.Resampling.BILINEAR)
            maps.append(np.array(e, dtype=np.float32) / 255.0)

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

    thr = np.clip(np.percentile(edge, 50), 0.25, 0.65)
    mask = edge <= thr
    mask = binary_closing(mask, square(2))
    mask = remove_small_objects(mask, min_size=16)
    thin = skeletonize(mask)
    thin = morphology.remove_small_holes(thin, area_threshold=16)

    out = np.where(thin, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def rescale_edge(edge: Image.Image, w: int, h: int) -> Image.Image:
    """Resize edge image back to original resolution.

    Args:
        edge: Edge map to resize.
        w: Target width.
        h: Target height.

    Returns:
        Image.Image: Resized edge image.

    Raises:
        None

    """
    return edge.resize((w, h), Image.Resampling.NEAREST)


# ---------- SD 1.5 + ControlNet(Lineart) ----------


def _configure_pipeline_memory(
    pipe: StableDiffusionControlNetImg2ImgPipeline, device: str
) -> None:
    """Enable memory optimisations for the diffusion pipeline.

    Args:
        pipe: Diffusion pipeline instance.
        device: Execution device string.

    Returns:
        None

    Raises:
        None

    """
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover - optional accel
        logger.warning("xFormers konnte nicht aktiviert werden: %s", exc)
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    if device == "cuda":
        import torch

        vram = torch.cuda.get_device_properties(0).total_memory
        pipe.enable_model_cpu_offload()
        if vram < SEQUENTIAL_OFFLOAD_VRAM:
            pipe.enable_sequential_cpu_offload()


@lru_cache(maxsize=1)
def load_sd15_lineart(
    model_id: str | Path = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet_id: str | Path = "lllyasviel/control_v11p_sd15_lineart",
    local_model_dir: Path | None = None,
    local_controlnet_dir: Path | None = None,
) -> StableDiffusionControlNetImg2ImgPipeline:
    """Load SD1.5 + ControlNet Lineart with memory-friendly options.

    Args:
        model_id: Hugging Face model identifier for the base pipeline.
        controlnet_id: Hugging Face identifier for the ControlNet weights.
        local_model_dir: Optional local directory for the pipeline.
        local_controlnet_dir: Optional local directory for ControlNet weights.

    Returns:
        StableDiffusionControlNetImg2ImgPipeline: Initialised pipeline.

    Raises:
        RuntimeError: If models fail to download or load locally.

    """
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
    )

    device = detect_device()
    dtype = detect_dtype(device)
    try:
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=dtype,
        )
    except (RequestsConnectionError, HTTPError, OSError) as exc:
        cn_candidates = [
            p
            for p in [
                local_controlnet_dir,
                Path("models") / "control_v11p_sd15_lineart",
            ]
            if p is not None
        ]
        cn_candidates.extend(find_model_dirs("control_v11p_sd15_lineart"))
        sd_candidates = [
            p for p in [local_model_dir, Path("models") / "sd15"] if p is not None
        ]
        sd_candidates.extend(find_model_dirs("sd15"))
        cn_candidates = list(dict.fromkeys([p.resolve() for p in cn_candidates]))
        sd_candidates = list(dict.fromkeys([p.resolve() for p in sd_candidates]))
        pipe = None
        for cn in cn_candidates:
            for sd in sd_candidates:
                if cn.exists() and sd.exists():
                    try:
                        controlnet = ControlNetModel.from_pretrained(
                            cn, torch_dtype=dtype
                        )
                        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                            sd,
                            controlnet=controlnet,
                            safety_checker=None,
                            torch_dtype=dtype,
                        )
                        break
                    except Exception:  # pragma: no cover - best effort
                        continue
            if pipe is not None:
                break
        if pipe is None:
            msg = (
                "Modell-Download fehlgeschlagen: ControlNet oder SD1.5. "
                "Bitte Netzwerk prüfen oder lokalen Pfad nutzen."
            )
            raise RuntimeError(msg) from exc
    pipe.to(device)
    _configure_pipeline_memory(pipe, device)
    return pipe


def sd_refine(
    base_rgb: Image.Image,
    ctrl_L: Image.Image,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    ctrl_scale: float = DEFAULT_CTRL_SCALE,
    strength: float = DEFAULT_STRENGTH,
    seed: int = DEFAULT_SEED,
    max_long: int = DEFAULT_MAX_LONG,
) -> tuple[Image.Image, Image.Image]:
    """Refine edges with SD1.5 + ControlNet.

    Args:
        base_rgb: Original RGB image.
        ctrl_L: Control image (edge map).
        steps: Number of diffusion steps.
        guidance: CFG scale.
        ctrl_scale: ControlNet conditioning scale.
        strength: Img2Img strength.
        seed: Random seed.
        max_long: Longest edge in pixels.

    Returns:
        tuple[Image.Image, Image.Image]: Refined color and BW images.

    Raises:
        RuntimeError: On GPU out-of-memory.

    """
    import torch

    pipe = load_sd15_lineart()
    device = pipe._execution_device
    W, H = base_rgb.size
    if max_long == DEFAULT_MAX_LONG:
        max_long = detect_max_long()
    w, h = resize_img(W, H, max_long=max_long)
    base = base_rgb.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
    ctrl = ctrl_L.resize((w, h), Image.Resampling.NEAREST).convert("RGB")

    gen = torch.Generator(device=device).manual_seed(seed)
    try:
        with (
            torch.inference_mode(),
            torch.autocast(device.type, dtype=detect_dtype(device.type)),
        ):
            result: Any = pipe(
                prompt=(
                    "clean black-and-white line art, uniform outlines, detailed, "
                    "background preserved, white paper look, no shading"
                ),
                negative_prompt=(
                    "color, gradients, blur, watermark, text, messy edges, artifacts"
                ),
                image=base,
                control_image=ctrl,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=ctrl_scale,
                strength=strength,
                generator=gen,
            )
            img = result.images[0]
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - device specific
        msg = "GPU out of memory. Bildgröße oder Batch-Size reduzieren."
        raise RuntimeError(msg) from exc

    bw = postprocess_lineart(img)
    return img, bw


# ---------- Verarbeitung ----------


def process_one(
    path: Path, out_dir: Path, cfg: Config, log: Callable[[str], None]
) -> None:
    """Process a single image and write outputs to *out_dir*.

    Args:
        path: Image file to process.
        out_dir: Output directory.
        cfg: Processing configuration.
        log: Callback for logging.

    Returns:
        None

    Raises:
        None

    """
    t0 = time.perf_counter()
    try:
        with Image.open(path) as img:
            img.verify()
        src = ensure_rgb(Image.open(path))
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

    result_paths = [out_edges]

    if cfg["use_sd"]:
        try:
            refined, bw = sd_refine(
                src,
                edges,
                steps=cfg["steps"],
                guidance=cfg["guidance"],
                ctrl_scale=cfg["ctrl"],
                strength=cfg["strength"],
                seed=cfg["seed"],
                max_long=cfg["max_long"],
            )
        except Exception as exc:
            log(f"FEHLER bei SD-Refinement: {exc}")
            return
        ref_path = out_dir / f"{path.stem}_refined.png"
        bw_path = out_dir / f"{path.stem}_refined_bw.png"
        refined.save(ref_path)
        bw.save(bw_path)
        result_paths += [ref_path, bw_path]

        if cfg["save_svg"]:
            svg_target = out_dir / f"{path.stem}_refined_bw.svg"
            svg_ok = save_svg_vtracer(bw_path, svg_target)
            if svg_ok:
                result_paths.append(svg_target)
            else:
                log("SVG-Export fehlgeschlagen – ist 'vtracer' installiert?")

    dt = time.perf_counter() - t0
    log(
        f"{path.name}  → fertig ({dt:.1f} s)\n   "
        + ", ".join([p.name for p in result_paths])
    )


def process_folder(  # noqa: C901
    inp_dir: Path,
    out_dir: Path,
    cfg: Config,
    log: Callable[[str], None],
    done_cb: Callable[[], None],
    stop_event: threading.Event | None = None,
    progress_cb: Callable[[int, int, Path], None] | None = None,
) -> None:
    """Process all supported images from *inp_dir* into *out_dir*.

    Args:
        inp_dir: Input directory.
        out_dir: Output directory.
        cfg: Processing configuration.
        log: Logger callback.
        done_cb: Callback invoked when finished.
        stop_event: Optional stop flag.
        progress_cb: Optional progress callback.

    Returns:
        None

    Raises:
        None

    """
    import torch

    try:
        ensure_dir(out_dir)
        imgs = list_images(inp_dir)
        if not imgs:
            log("Keine Eingabebilder gefunden.")
            done_cb()
            return
        total = len(imgs)
        if shutil.disk_usage(out_dir).free < MIN_DISK_SPACE:
            log("FEHLER: Zu wenig Speicherplatz im Ausgabeverzeichnis")
            done_cb()
            return

        batch_size = max(1, cfg.get("batch_size", 1))
        idx = 0
        while idx < total:
            batch = imgs[idx : idx + batch_size]
            try:
                for p in batch:
                    if stop_event is not None and stop_event.is_set():
                        log("Verarbeitung abgebrochen.")
                        done_cb()
                        return
                    process_one(p, out_dir, cfg, log)
                    idx += 1
                    if progress_cb:
                        progress_cb(idx, total, p)
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "gpu out of memory" in str(exc).lower() and batch_size > 1:
                    log("GPU OOM – reduziere Batch-Size auf 1 und versuche erneut …")
                    batch_size = 1
                    if torch.cuda.is_available():  # pragma: no cover
                        torch.cuda.empty_cache()
                    continue
                log(f"FEHLER: {exc}")
                idx += len(batch)
        log("\nALLE BILDER ERLEDIGT.")
        done_cb()
    finally:
        cleanup_models()


def prefetch_models(log: Callable[[str], None]) -> None:
    """Download all required models ahead of time.

    Args:
        log: Logger callback.

    Returns:
        None

    Raises:
        None

    """
    log("Lade Modelle vom Hub … (einmalig)")
    _ = load_dexined(device="cpu")
    _ = load_sd15_lineart()
    log("Modelle vorhanden.\n")


def cleanup_models() -> None:
    """Release loaded models and free GPU memory.

    Returns:
        None

    Raises:
        None

    """
    import gc

    import torch

    load_dexined.cache_clear()
    load_sd15_lineart.cache_clear()
    gc.collect()
    if torch.cuda.is_available():  # pragma: no cover - hardware specific
        torch.cuda.empty_cache()
