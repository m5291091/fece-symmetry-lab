"""FR-042: レポートへ埋め込む図の符号化（`adapters/imaging/image_encoding`）。

守るのは 2 点である。

1. **原寸を埋めない。** 12MP を base64 で埋めると 1 ファイルが 20MB を超え、
   共有も閲覧も現実的でなくなる（§10.4）。長辺 1,280px 以下へ縮小する。
2. **縮小は表示専用。** 呼び出し側の配列を書き換えない。測定値はこの経路を
   一切通らない。
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from face_symmetry_lab.adapters.imaging.image_encoding import (
    MAX_LONG_EDGE_PX,
    ImageEncodingError,
    downscale_for_report,
    encode_data_uri,
)


def _rgb(height: int, width: int, value: int = 120) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_an_image_within_the_limit_is_left_alone() -> None:
    """罠: 常に resize する実装は、小さな図を無意味に再標本化して滲ませる。"""
    source = _rgb(480, 640)

    assert downscale_for_report(source) is source


def test_a_large_image_is_reduced_to_the_documented_long_edge() -> None:
    reduced = downscale_for_report(_rgb(3000, 4000))

    assert max(reduced.shape[:2]) == MAX_LONG_EDGE_PX


def test_the_aspect_ratio_survives_the_reduction() -> None:
    reduced = downscale_for_report(_rgb(3000, 4000))
    height, width = reduced.shape[:2]

    assert width / height == pytest.approx(4000 / 3000, rel=1e-2)


def test_a_tall_image_is_reduced_by_its_own_long_edge() -> None:
    """罠: 幅だけを見る実装は、縦長の写真を縮小せずに通す。"""
    reduced = downscale_for_report(_rgb(4000, 3000))

    assert reduced.shape[0] == MAX_LONG_EDGE_PX
    assert reduced.shape[1] < MAX_LONG_EDGE_PX


def test_the_caller_array_is_never_modified() -> None:
    source = _rgb(2000, 2000)
    before = source.copy()

    downscale_for_report(source)

    assert np.array_equal(source, before)


def test_the_data_uri_is_self_contained_and_decodable() -> None:
    """AC-REPORT-02: 外部リソースを参照しない。図は同じファイルの中にある。"""
    uri = encode_data_uri(_rgb(64, 64))

    assert uri.startswith("data:image/jpeg;base64,")
    payload = base64.b64decode(uri.split(",", 1)[1], validate=True)
    # JPEG の SOI マーカー。実際に画像として符号化されている証拠。
    assert payload[:2] == b"\xff\xd8"


def test_encoding_reduces_a_large_image_before_embedding() -> None:
    """罠: 縮小を通さずに符号化する実装は、原寸を base64 で埋め込む。"""
    big = encode_data_uri(_rgb(3000, 4000, value=90))
    small = encode_data_uri(_rgb(1200, 1600, value=90))

    # 同じ内容・同じ品質なら、符号化後の長さは画素数に従う。原寸のまま埋めて
    # いれば、こちらが数倍になる。
    assert len(big) < len(small) * 2


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_a_malformed_array_is_refused(bad: np.ndarray) -> None:
    with pytest.raises(ImageEncodingError):
        encode_data_uri(bad)


def test_a_malformed_array_is_refused_before_resizing_too() -> None:
    with pytest.raises(ImageEncodingError):
        downscale_for_report(np.zeros((10, 10), dtype=np.uint8))
