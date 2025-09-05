# noqa: D100

from PIL import Image

from src.pipeline import ensure_rgb, resize_img


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
