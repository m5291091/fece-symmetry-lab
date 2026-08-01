from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from face_symmetry_lab.domain.landmarks import LandmarkFrame

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_PIXELS = 12_000_000
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG"})


class ImageInputError(ValueError):
    """A safe, user-facing image validation failure."""


@dataclass(frozen=True, slots=True)
class DecodedImage:
    rgb: Image.Image
    format: str
    width: int
    height: int


def decode_uploaded_image(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> DecodedImage:
    """Validate, orient and detach an uploaded JPEG/PNG from its metadata."""
    if not payload:
        raise ImageInputError("画像を読み込めません。空のファイルです。")
    if len(payload) > max_bytes:
        raise ImageInputError("ファイルサイズが上限を超えています。")

    try:
        with Image.open(BytesIO(payload)) as source:
            image_format = source.format
            if image_format not in SUPPORTED_FORMATS:
                raise ImageInputError("JPEG または PNG を選択してください。")
            if getattr(source, "is_animated", False) or getattr(source, "n_frames", 1) != 1:
                raise ImageInputError("アニメーション画像は使用できません。")

            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ImageInputError("画像の画素数が上限を超えています。")

            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgb = oriented.convert("RGB").copy()
            rgb.info.clear()
            rgb.getexif().clear()
    except ImageInputError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ImageInputError("画像を読み込めません。破損していないか確認してください。") from error

    return DecodedImage(
        rgb=rgb,
        format=image_format,
        width=rgb.width,
        height=rgb.height,
    )


def extract_face_crop(
    rgb_image: NDArray[np.uint8],
    frame: LandmarkFrame,
    *,
    padding_ratio: float = 0.15,
) -> NDArray[np.uint8]:
    """Return a clamped, detached face bbox crop for local face models."""
    image = np.asarray(rgb_image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ImageInputError("顔 crop には uint8 の RGB 画像が必要です。")
    if not frame.has_single_face or not frame.points:
        raise ImageInputError("顔モデル推論には単一顔のlandmarkが必要です。")
    if not 0.0 <= padding_ratio <= 1.0:
        raise ImageInputError("顔 crop の余白率が不正です。")

    height, width = image.shape[:2]
    xs = [point[0] * width for point in frame.points]
    ys = [point[1] * height for point in frame.points]
    face_width = max(xs) - min(xs)
    face_height = max(ys) - min(ys)
    if face_width <= 0 or face_height <= 0:
        raise ImageInputError("顔 crop の範囲を決定できません。")
    padding_x = face_width * padding_ratio
    padding_y = face_height * padding_ratio
    left = max(0, int(np.floor(min(xs) - padding_x)))
    right = min(width, int(np.ceil(max(xs) + padding_x)))
    top = max(0, int(np.floor(min(ys) - padding_y)))
    bottom = min(height, int(np.ceil(max(ys) + padding_y)))
    if left >= right or top >= bottom:
        raise ImageInputError("顔 crop の範囲を決定できません。")
    return np.ascontiguousarray(image[top:bottom, left:right]).copy()
