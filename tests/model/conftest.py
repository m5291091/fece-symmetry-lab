"""実モデル smoke test のための合成 fixture。

§21.4 は再配布可能な合成 fixture を優先する。以下の顔はプログラムで描画するので、
実在の人物の画像がこの repository に保存されることはなく、入力はどの機械でも
同一である。
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

FACE_IMAGE_SIZE = 512


def _draw_synthetic_face() -> Image.Image:
    size = FACE_IMAGE_SIZE
    image = Image.new("RGB", (size, size), (235, 230, 225))
    canvas = ImageDraw.Draw(image)

    canvas.ellipse([140, 90, 372, 400], fill=(226, 190, 165))
    canvas.rectangle([225, 370, 287, 430], fill=(214, 178, 153))
    canvas.ellipse([120, 415, 392, 560], fill=(90, 100, 130))
    canvas.chord([135, 70, 377, 300], 180, 360, fill=(60, 45, 40))

    for center_x in (211, 301):
        canvas.ellipse([center_x - 26, 208, center_x + 26, 240], fill=(252, 252, 250))
        canvas.ellipse([center_x - 12, 210, center_x + 12, 236], fill=(70, 50, 35))
        canvas.ellipse([center_x - 5, 217, center_x + 5, 229], fill=(15, 12, 10))
        canvas.arc([center_x - 28, 195, center_x + 28, 250], 180, 360, fill=(80, 60, 50), width=4)

    canvas.arc([183, 178, 239, 214], 190, 350, fill=(60, 45, 40), width=7)
    canvas.arc([273, 178, 329, 214], 190, 350, fill=(60, 45, 40), width=7)

    canvas.polygon([(256, 235), (240, 300), (272, 300)], fill=(214, 175, 150))
    canvas.ellipse([238, 292, 250, 304], fill=(190, 150, 128))
    canvas.ellipse([262, 292, 274, 304], fill=(190, 150, 128))
    canvas.chord([216, 305, 296, 355], 0, 180, fill=(170, 95, 90))

    return image.filter(ImageFilter.GaussianBlur(1.2))


@pytest.fixture(scope="session")
def synthetic_face_rgb() -> NDArray[np.uint8]:
    """Face Landmarker が確実に検出できる、描画された単一顔の RGB 画像。"""
    return np.asarray(_draw_synthetic_face(), dtype=np.uint8)


@pytest.fixture(scope="session")
def synthetic_two_face_rgb(synthetic_face_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """MULTI_FACE の経路を通すための、横に並べた 2 つの描画顔。"""
    face = Image.fromarray(synthetic_face_rgb).resize((360, 360))
    canvas = Image.new("RGB", (820, 420), (235, 230, 225))
    canvas.paste(face, (20, 30))
    canvas.paste(face, (440, 30))
    return np.asarray(canvas, dtype=np.uint8)
