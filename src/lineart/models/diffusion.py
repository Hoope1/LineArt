"""Stable Diffusion + ControlNet helpers."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from ..config import PipelineConfig
from ..constants import DEFAULT_MAX_LONG, SEQUENTIAL_OFFLOAD_VRAM
from ..devices import detect_device, detect_dtype, detect_max_long
from ..fs import find_model_dirs
from ..image_ops import postprocess_lineart, resize_img

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from diffusers import StableDiffusionControlNetImg2ImgPipeline
    from torch import dtype as TorchDType


@lru_cache(maxsize=1)
def load_sd15_lineart(
    model_id: str | Path = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet_id: str | Path = "lllyasviel/control_v11p_sd15_lineart",
    local_model_dir: Path | None = None,
    local_controlnet_dir: Path | None = None,
) -> StableDiffusionControlNetImg2ImgPipeline:
    """Load SD1.5 + ControlNet Lineart with memory-friendly options."""
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
    )
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import HTTPError

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
            for p in (
                local_controlnet_dir,
                Path("models") / "control_v11p_sd15_lineart",
            )
            if p is not None
        ]
        cn_candidates.extend(find_model_dirs("control_v11p_sd15_lineart"))
        sd_candidates = [p for p in (local_model_dir, Path("models") / "sd15") if p is not None]
        sd_candidates.extend(find_model_dirs("sd15"))
        cn_candidates = list(dict.fromkeys([p.resolve() for p in cn_candidates]))
        sd_candidates = list(dict.fromkeys([p.resolve() for p in sd_candidates]))
        pipe = None
        for cn in cn_candidates:
            for sd in sd_candidates:
                if cn.exists() and sd.exists():
                    try:
                        controlnet = ControlNetModel.from_pretrained(cn, torch_dtype=dtype)
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


def _configure_pipeline_memory(pipe: StableDiffusionControlNetImg2ImgPipeline, device: str) -> None:
    """Enable memory optimisations for the diffusion pipeline."""
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


def _autocast_context(device: torch.device, dtype: TorchDType) -> AbstractContextManager[Any]:
    """Return a safe autocast context for the given *device* and *dtype*."""
    import torch

    device_type = device.type
    if device_type == "cuda":
        return torch.autocast(device_type, dtype=dtype)
    if device_type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type, dtype=dtype)
    return nullcontext()


def sd_refine(
    base_rgb: Image.Image,
    ctrl_L: Image.Image,
    config: PipelineConfig,
) -> tuple[Image.Image, Image.Image]:
    """Refine edges with SD1.5 + ControlNet."""
    import torch

    pipe = load_sd15_lineart()
    device = pipe._execution_device
    device_type = device.type
    width, height = base_rgb.size
    max_long = config.max_long
    if max_long == DEFAULT_MAX_LONG:
        max_long = detect_max_long()
    width_resized, height_resized = resize_img(width, height, max_long=max_long)
    base = base_rgb.resize((width_resized, height_resized), Image.Resampling.LANCZOS).convert("RGB")
    ctrl = ctrl_L.resize((width_resized, height_resized), Image.Resampling.NEAREST).convert("RGB")

    gen_device = f"{device_type}:{device.index}" if device.index is not None else device_type
    generator = torch.Generator(device=gen_device).manual_seed(config.seed)
    autocast_ctx = _autocast_context(device, detect_dtype(device_type))
    try:
        with (
            torch.inference_mode(),
            autocast_ctx,
        ):
            result: Any = pipe(
                prompt=(
                    "clean black-and-white line art, uniform outlines, detailed, "
                    "background preserved, white paper look, no shading"
                ),
                negative_prompt=("color, gradients, blur, watermark, text, messy edges, artifacts"),
                image=base,
                control_image=ctrl,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance,
                controlnet_conditioning_scale=config.ctrl,
                strength=config.strength,
                generator=generator,
            )
            img = result.images[0]
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - device specific
        msg = "GPU out of memory. Bildgröße oder Batch-Size reduzieren."
        raise RuntimeError(msg) from exc

    bw = postprocess_lineart(img)
    return img, bw
