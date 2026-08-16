"""FR-042: アプリが持っている画像を、レポートへ埋め込める図へ変換する。

`ui.report`（HTML の組み立て）と `adapters.imaging.image_encoding`（符号化）の
あいだを繋ぐだけの薄い層である。分けてあるのは、`ui.report` を Streamlit からも
cv2 からも自由に保つためで、あちらは文字列を作るだけの関数として検証できる。

**符号化の失敗で、レポートそのものを落とさない。** 図は補助であり、数値と来歴は
図が無くても成立する。1 枚の符号化に失敗したら、その 1 枚を落として残りを返す
（不変条件 12 の失敗分離と同じ扱い）。
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.adapters.imaging.image_encoding import (
    ImageEncodingError,
    encode_data_uri,
)
from face_symmetry_lab.application.symmetrization import SymmetrizationView
from face_symmetry_lab.ui.report import ReportFigure

LANDMARK_CAPTION: Final[str] = "検出したランドマークを重ねたプレビュー"

OVERLAY_CAPTION: Final[str] = "左右対称にそろえた位置との比較"

#: §5.3 / 不変条件 8: 対称化点は「そろえた位置」であって理想でも正解でもない。
#: 画面（`ui.symmetrization_panel`）と同じ立場をレポートでも明示する。
OVERLAY_NOTE: Final[str] = (
    "実測した点と、左右対称にそろえた位置の点を重ねた図です。"
    "どちらが正しい・望ましいという意味はありません。"
)

WARPED_CAPTION: Final[str] = "左右対称にそろえた画像（幾何変形）"

#: 不変条件 8: 対称化画像は加工画像である。**この但し書きは図と同じファイルの中に
#: 無ければならない**——画像だけが切り出されて共有されると、無加工の写真として
#: 流通しうる。
WARPED_NOTE: Final[str] = (
    "画素を幾何変形して作った加工画像です。無加工の写真として提示・共有しないでください。"
)


def _figure(
    rgb_image: NDArray[np.uint8] | None,
    *,
    caption: str,
    note: str = "",
) -> ReportFigure | None:
    """1 枚を符号化する。失敗と欠測は `None`（呼び出し側が落とす）。"""
    if rgb_image is None:
        return None
    try:
        data_uri = encode_data_uri(np.asarray(rgb_image))
    except (ImageEncodingError, ValueError):
        return None
    return ReportFigure(caption=caption, data_uri=data_uri, note=note)


def build_report_figures(
    *,
    include_figures: bool,
    preview: NDArray[np.uint8] | None = None,
    view: SymmetrizationView | None = None,
) -> tuple[ReportFigure, ...]:
    """レポートへ入れる図を組み立てる。

    ``include_figures`` が偽なら**必ず空を返す**（§15.1 の設定、AC-REPORT-03）。
    ここが唯一の分岐点であり、呼び出し側が個別に判断しない。

    ``preview`` はランドマーク重畳、``view`` は FR-034 の対称化ビュー。後者は
    `warped`（FR-035）を持っているときだけ 2 枚目を出す。
    """
    if not include_figures:
        return ()
    candidates = [
        _figure(preview, caption=LANDMARK_CAPTION),
        _figure(None if view is None else view.overlay, caption=OVERLAY_CAPTION, note=OVERLAY_NOTE),
        _figure(None if view is None else view.warped, caption=WARPED_CAPTION, note=WARPED_NOTE),
    ]
    return tuple(figure for figure in candidates if figure is not None)
