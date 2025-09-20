# noqa: D100

from contextlib import nullcontext

import torch
from PIL import Image

from src import pipeline
from src.pipeline import detect_dtype, ensure_rgb, list_images, resize_img


def test_resize_img_limits_and_multiple_of_eight() -> None:
    """Resizing keeps dimensions within limit and multiples of eight."""
    w, h = resize_img(2000, 1000, max_long=1000)
    assert max(w, h) <= 1000
    assert w % 8 == h % 8 == 0


def test_ensure_rgb_converts_mode() -> None:
    """ensure_rgb converts non-RGB images to RGB."""
    img = Image.new("L", (10, 10))
    rgb = ensure_rgb(img)
    assert rgb.mode == "RGB"


def test_ensure_rgb_returns_copy_for_rgb() -> None:
    """ensure_rgb returns a copy so the source file handle is released."""
    img = Image.new("RGB", (10, 10))
    rgb = ensure_rgb(img)
    assert rgb is not img


def test_list_images_sorted(tmp_path) -> None:
    """list_images returns files in deterministic order."""
    (tmp_path / "b.png").touch()
    (tmp_path / "a.png").touch()
    files = list_images(tmp_path)
    assert [f.name for f in files] == ["a.png", "b.png"]


def test_list_images_skips_directories(tmp_path) -> None:
    """list_images ignores directories even if they have an image extension."""
    (tmp_path / "a.png").mkdir()
    (tmp_path / "b.png").touch()
    files = list_images(tmp_path)
    assert [f.name for f in files] == ["b.png"]


def test_detect_dtype_cpu(monkeypatch) -> None:
    """CPU dtype falls back to float32 without CUDA check."""
    import torch

    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda: (_ for _ in ()).throw(RuntimeError)
    )
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", bool)
    assert detect_dtype("cpu") is torch.float32


def test_autocast_context_skips_unsupported_dtype(monkeypatch) -> None:
    """_autocast_context should not call torch.autocast for unsupported combos."""
    called = False

    def fake_autocast(*_args, **_kwargs):  # noqa: D401
        """Record the call and return a dummy context manager."""
        nonlocal called
        called = True
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    ctx = pipeline._autocast_context(torch.device("cpu"), torch.float32)
    assert not called
    assert isinstance(ctx, type(nullcontext()))


def test_autocast_context_cpu_bfloat16(monkeypatch) -> None:
    """Supported CPU dtype should call torch.autocast."""
    called = False

    def fake_autocast(*_args, **_kwargs):  # noqa: D401
        """Record invocation."""
        nonlocal called
        called = True
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    ctx = pipeline._autocast_context(torch.device("cpu"), torch.bfloat16)
    assert called
    with ctx:
        pass
