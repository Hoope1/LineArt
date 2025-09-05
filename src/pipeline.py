#!/usr/bin/env python3
"""Core pipeline for DexiNed → SD 1.5 + ControlNet(Lineart).

This module contains all heavy computations so the GUI in ``main.py`` stays
lightweight.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import exposure, morphology
from skimage.morphology import binary_closing, remove_small_objects, skeletonize
from skimage.morphology.footprints import square

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    """Return all image paths in *folder* with a supported extension."""
    return [p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS]


def resize_img(w: int, h: int, max_long: int = 896) -> tuple[int, int]:
    """Scale dimensions to multiples of eight, limiting the longest side."""
    if max(w, h) > max_long:
        scale = max_long / max(w, h)
        w, h = int(w * scale), int(h * scale)
    return max(64, (w // 8) * 8), max(64, (h // 8) * 8)


def ensure_dir(p: Path) -> Path:
    """Create directory *p* if needed and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert *img* to RGB mode if necessary."""
    return img.convert("RGB") if img.mode != "RGB" else img


def postprocess_lineart(img: Image.Image) -> Image.Image:
    """Binarise *img* for crisp black/white output."""
    return img.convert("L").point(lambda v: 255 if v > 200 else 0, mode="1").convert("L")


def save_svg_vtracer(png_path: Path, svg_path: Path) -> bool:
    """Convert *png_path* to SVG via ``vtracer`` and save to *svg_path*."""
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
    except Exception:
        return False


# ---------- DexiNed (controlnet-aux) ----------


@lru_cache(maxsize=1)
def load_dexined(device: str = "cuda"):
    """Load the DexiNed edge detector on the requested *device*."""
    from controlnet_aux import DexiNedDetector  # type: ignore[attr-defined]

    return DexiNedDetector.from_pretrained("lllyasviel/Annotators").to(device)


def guided_smooth_if_available(pil_img: Image.Image) -> Image.Image:
    """Apply detail-preserving denoising."""
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None:
        arr = ximgproc.guidedFilter(guide=arr, src=arr, radius=4, eps=1e-2)  # type: ignore[call-arg]
    else:
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=25, sigmaSpace=25)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def get_dexined(
    pil_img: Image.Image,
    scales: tuple[float, ...] = (1.0, 0.75, 1.25),
    pre_smooth: bool = True,
) -> Image.Image:
    """Run DexiNed on *pil_img* and return a grayscale edge map."""
    det = load_dexined(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    img = ensure_rgb(pil_img)
    if pre_smooth:
        img = guided_smooth_if_available(img)

    maps: list[np.ndarray] = []
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
    lo, hi = np.percentile(edge, 5), np.percentile(edge, 99)
    edge = exposure.rescale_intensity(edge, in_range=(lo, hi))
    edge = cv2.GaussianBlur(edge, (0, 0), 0.7)

    thr = np.clip(np.percentile(edge, 50), 0.25, 0.65)
    mask = edge <= thr
    mask = binary_closing(mask, square(2))
    mask = remove_small_objects(mask, min_size=16)
    thin = skeletonize(mask)
    thin = morphology.remove_small_holes(thin, area_threshold=16)

    out = np.where(thin, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def rescale_edge(edge: Image.Image, w: int, h: int) -> Image.Image:
    """Resize edge image back to original resolution."""
    return edge.resize((w, h), Image.Resampling.NEAREST)


# ---------- SD 1.5 + ControlNet(Lineart) ----------


@lru_cache(maxsize=1)
def load_sd15_lineart():
    """Load SD1.5 + ControlNet Lineart with memory-friendly options."""
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    return pipe


def sd_refine(
    base_rgb: Image.Image,
    ctrl_L: Image.Image,
    steps: int = 32,
    guidance: float = 6.0,
    ctrl_scale: float = 1.0,
    strength: float = 0.70,
    seed: int = 42,
    max_long: int = 896,
) -> tuple[Image.Image, Image.Image]:
    """Refine edges with SD1.5 + ControlNet and return color and BW images."""
    import torch

    pipe = load_sd15_lineart()
    W, H = base_rgb.size
    w, h = resize_img(W, H, max_long=max_long)
    base = base_rgb.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
    ctrl = ctrl_L.resize((w, h), Image.Resampling.NEAREST).convert("RGB")

    gen = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    img = pipe(
        prompt=(
            "clean black-and-white line art, uniform outlines, detailed, "
            "background preserved, white paper look, no shading"
        ),
        negative_prompt="color, gradients, blur, watermark, text, messy edges, artifacts",
        image=base,
        control_image=ctrl,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=ctrl_scale,
        strength=strength,
        generator=gen,
    ).images[0]

    bw = postprocess_lineart(img)
    return img, bw


# ---------- Verarbeitung ----------


def process_one(path: Path, out_dir: Path, cfg, log: Callable[[str], None]) -> None:
    """Process a single image and write outputs to *out_dir*."""
    t0 = time.perf_counter()
    src = ensure_rgb(Image.open(path))

    edges = get_dexined(src)
    ensure_dir(out_dir)
    prep_dir = ensure_dir(out_dir / "preprocessed")
    out_edges = prep_dir / f"{path.stem}_dexi.png"
    edges.save(out_edges)

    result_paths = [out_edges]

    if cfg["use_sd"]:
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
        ref_path = out_dir / f"{path.stem}_refined.png"
        bw_path = out_dir / f"{path.stem}_refined_bw.png"
        refined.save(ref_path)
        bw.save(bw_path)
        result_paths += [ref_path, bw_path]

        if cfg["save_svg"]:
            svg_ok = save_svg_vtracer(bw_path, out_dir / f"{path.stem}_refined_bw.svg")
            if svg_ok:
                result_paths.append(out_dir / f"{path.stem}_refined_bw.svg")

    dt = time.perf_counter() - t0
    log(f"{path.name}  → fertig ({dt:.1f} s)\n   " + ", ".join([p.name for p in result_paths]))


def process_folder(
    inp_dir: Path,
    out_dir: Path,
    cfg,
    log: Callable[[str], None],
    done_cb: Callable[[], None],
) -> None:
    """Process all supported images from *inp_dir* into *out_dir*."""
    imgs = list_images(inp_dir)
    if not imgs:
        log("Keine Eingabebilder gefunden.")
        done_cb()
        return
    for p in imgs:
        process_one(p, out_dir, cfg, log)
    log("\nALLE BILDER ERLEDIGT.")
    done_cb()


def prefetch_models(log: Callable[[str], None]) -> None:
    """Download all required models ahead of time."""
    log("Lade Modelle vom Hub … (einmalig)")
    _ = load_dexined(device="cpu")
    _ = load_sd15_lineart()
    log("Modelle vorhanden.\n")
