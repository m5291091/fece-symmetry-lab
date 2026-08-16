"""FR-042: レポートへ埋め込む図を `data:` URI へ符号化する。

**この module は測定を一切知らない。** 受け取るのは表示用に作られた RGB 配列
（ランドマーク重畳プレビュー・対称化オーバーレイ・対称化画像）だけで、ここを
通った値が指標へ戻ることはない。縮小は表示専用である（§10.4）。

`data:` URI にするのは、レポートが外部リソースを参照しないためである
（AC-REPORT-02）。開いた先で通信が起きるレポートは、ローカル完結という前提を
ファイル単体では守れていない。
"""

from __future__ import annotations

import base64
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

#: §10.4: 埋め込む前に長辺をここまで落とす。原寸（最大 12MP）を base64 で埋めると
#: 1 ファイルが 20MB を超え、共有も閲覧も現実的でなくなる。
MAX_LONG_EDGE_PX: Final[int] = 1280

#: JPEG の品質。図はランドマークの点と線が読めれば足りるので、可逆である必要は
#: ない。PNG にすると同じ写真で数倍の大きさになる。
JPEG_QUALITY: Final[int] = 85


class ImageEncodingError(ValueError):
    """レポートへ埋め込む画像が不正である。"""


def _validate(rgb_image: NDArray[np.uint8]) -> None:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ImageEncodingError("画像は (H, W, 3) の RGB 配列である必要があります。")
    if rgb_image.dtype != np.uint8:
        raise ImageEncodingError("画像は uint8 である必要があります。")
    if rgb_image.shape[0] < 1 or rgb_image.shape[1] < 1:
        raise ImageEncodingError("画像に画素がありません。")


def downscale_for_report(rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """長辺が ``MAX_LONG_EDGE_PX`` を超えるときだけ、比率を保って縮小する。

    **拡大はしない。** 上限以下の画像はそのまま（同一オブジェクトを）返す。
    常に `resize` を通すと、小さな図が無意味に再標本化されて滲む。

    呼び出し側の配列は書き換えない（`cv2.resize` は新しい配列を返す）。
    """
    _validate(rgb_image)
    height, width = rgb_image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= MAX_LONG_EDGE_PX:
        return rgb_image
    scale = MAX_LONG_EDGE_PX / long_edge
    # 少なくとも 1px は残す。極端に細長い図で 0 になると `cv2.resize` が落ちる。
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(rgb_image, target, interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def encode_data_uri(rgb_image: NDArray[np.uint8]) -> str:
    """縮小してから JPEG へ符号化し、`data:` URI として返す。

    **縮小を必ず通す。** ここを飛ばして符号化する実装は、原寸を base64 で埋め込む。
    """
    reduced = downscale_for_report(rgb_image)
    # cv2 は BGR を前提にする。RGB のまま渡すと赤と青が入れ替わった図が出る。
    bgr = cv2.cvtColor(reduced, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not success:
        raise ImageEncodingError("画像を符号化できませんでした。")
    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
