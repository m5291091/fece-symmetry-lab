from io import BytesIO

import pytest
from PIL import Image

from face_symmetry_lab.application.image_input import (
    ImageInputError,
    decode_uploaded_image,
)


def _image_bytes(
    *,
    image_format: str = "PNG",
    size: tuple[int, int] = (16, 12),
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=(32, 64, 96)).save(buffer, format=image_format)
    return buffer.getvalue()


def test_decode_uploaded_image_returns_rgb_pixels_without_metadata() -> None:
    decoded = decode_uploaded_image(_image_bytes())

    assert decoded.format == "PNG"
    assert decoded.width == 16
    assert decoded.height == 12
    assert decoded.rgb.mode == "RGB"
    assert decoded.rgb.getexif() == {}


def test_decode_uploaded_image_rejects_unsupported_format() -> None:
    with pytest.raises(ImageInputError, match="JPEG または PNG"):
        decode_uploaded_image(_image_bytes(image_format="BMP"))


def test_decode_uploaded_image_rejects_oversized_payload() -> None:
    with pytest.raises(ImageInputError, match="ファイルサイズ"):
        decode_uploaded_image(_image_bytes(), max_bytes=10)


def test_decode_uploaded_image_rejects_excessive_pixel_count() -> None:
    with pytest.raises(ImageInputError, match="画素数"):
        decode_uploaded_image(_image_bytes(size=(20, 20)), max_pixels=399)


def test_decode_uploaded_image_rejects_animated_image() -> None:
    buffer = BytesIO()
    first = Image.new("RGB", (8, 8), color="white")
    second = Image.new("RGB", (8, 8), color="black")
    first.save(buffer, format="PNG", save_all=True, append_images=[second])

    with pytest.raises(ImageInputError, match="アニメーション"):
        decode_uploaded_image(buffer.getvalue())


def test_decode_uploaded_image_rejects_corrupt_data() -> None:
    with pytest.raises(ImageInputError, match="読み込めません"):
        decode_uploaded_image(b"not-an-image")
