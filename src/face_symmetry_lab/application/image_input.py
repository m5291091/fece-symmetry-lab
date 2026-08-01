from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

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
