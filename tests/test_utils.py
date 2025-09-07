# noqa: D100

from PIL import Image

from src.pipeline import detect_dtype, ensure_rgb, list_images, resize_img


def test_resize_img_limits_and_multiple_of_eight() -> None:
    """Resizing keeps dimensions within limit and multiples of eight."""
    w, h = resize_img(2000, 1000, max_long=1000)
    assert max(w, h) <= 1000
    assert w % 8 == 0 and h % 8 == 0


def test_ensure_rgb_converts_mode() -> None:
    """ensure_rgb converts non-RGB images to RGB."""
    img = Image.new("L", (10, 10))
    rgb = ensure_rgb(img)
    assert rgb.mode == "RGB"


def test_list_images_sorted(tmp_path) -> None:
    """list_images returns files in deterministic order."""
    (tmp_path / "b.png").touch()
    (tmp_path / "a.png").touch()
    files = list_images(tmp_path)
    assert [f.name for f in files] == ["a.png", "b.png"]


def test_detect_dtype_cpu(monkeypatch) -> None:
    """CPU dtype falls back to float32 without CUDA check."""
    import torch

    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda: (_ for _ in ()).throw(RuntimeError)
    )
    monkeypatch.setattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)
    assert detect_dtype("cpu") is torch.float32
