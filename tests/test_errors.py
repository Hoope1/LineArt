"""Tests for pipeline error handling."""

from __future__ import annotations

import diffusers
import pytest
import torch
from PIL import Image

import lineart.pipeline as pipeline


class DummyPipe:  # noqa: D101
    """Mock pipeline raising OOM."""

    _execution_device = torch.device("cpu")

    def __call__(self, *_args, **_kwargs):  # noqa: D401
        """Raise OOM to simulate GPU exhaustion."""
        raise torch.cuda.OutOfMemoryError("OOM")


class DummyControlNet:  # noqa: D101
    """Mock ControlNet that fails on load."""

    @classmethod
    def from_pretrained(_cls, *_args, **_kwargs):  # noqa: D401
        """Raise error to simulate download failure."""
        raise ConnectionError("net")


class DummySDPipeline:  # noqa: D101
    """Mock SD pipeline that fails on load."""

    @classmethod
    def from_pretrained(_cls, *_args, **_kwargs):  # noqa: D401
        """Raise error to simulate download failure."""
        raise ConnectionError("net")


def test_load_sd15_lineart_download_error(monkeypatch) -> None:
    """load_sd15_lineart wraps download failures."""
    monkeypatch.setattr(diffusers, "ControlNetModel", DummyControlNet)
    monkeypatch.setattr(
        diffusers,
        "StableDiffusionControlNetImg2ImgPipeline",
        DummySDPipeline,
    )
    with pytest.raises(RuntimeError):
        pipeline.load_sd15_lineart()


def test_sd_refine_oom(monkeypatch) -> None:
    """sd_refine raises friendly OOM errors."""
    monkeypatch.setattr(pipeline, "load_sd15_lineart", DummyPipe)
    monkeypatch.setattr(
        "lineart.models.diffusion.load_sd15_lineart",
        lambda *_args, **_kwargs: DummyPipe(),
    )
    img = Image.new("RGB", (64, 64))
    with pytest.raises(RuntimeError):
        pipeline.sd_refine(img, img, pipeline.PipelineConfig())
