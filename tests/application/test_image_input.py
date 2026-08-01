from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from face_symmetry_lab.application.image_input import (
    ImageInputError,
    decode_uploaded_image,
    extract_face_crop,
)
from face_symmetry_lab.domain.landmarks import build_landmark_frame


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


def test_face_crop_uses_the_landmark_bbox_with_padding() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    frame = build_landmark_frame(
        frame_id=1,
        face_count=1,
        points=((0.25, 0.20), (0.75, 0.80)),
        width=200,
        height=100,
    )

    crop = extract_face_crop(image, frame, padding_ratio=0.10)

    assert crop.shape == (72, 120, 3)
    assert crop is not image


def test_face_crop_is_clamped_to_the_image() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    frame = build_landmark_frame(
        frame_id=2,
        face_count=1,
        points=((0.0, 0.0), (1.0, 1.0)),
        width=30,
        height=20,
    )

    crop = extract_face_crop(image, frame, padding_ratio=0.25)

    assert crop.shape == image.shape


def test_face_crop_rejects_frames_without_one_face() -> None:
    with pytest.raises(ImageInputError, match="単一顔"):
        extract_face_crop(
            np.zeros((20, 30, 3), dtype=np.uint8),
            build_landmark_frame(frame_id=3, face_count=0),
        )
