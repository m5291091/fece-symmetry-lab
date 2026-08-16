"""FR-042: レポートへ入れる図の組み立て（`ui.report_figures`）。

守るのは 3 点である。

1. **`include_figures` が偽なら必ず空。** 分岐はここ 1 か所で、呼び出し側が
   個別に判断しない（AC-REPORT-03）。
2. **対称化画像には「加工画像」の但し書きが付く。** 図だけが切り出されて共有
   されると、無加工の写真として流通しうる（不変条件 8）。
3. **符号化の失敗でレポートが落ちない。** 図は補助であり、数値と来歴は図が
   無くても成立する（不変条件 12）。
"""

from __future__ import annotations

import numpy as np

from face_symmetry_lab.ui.report_figures import (
    LANDMARK_CAPTION,
    OVERLAY_CAPTION,
    WARPED_CAPTION,
    WARPED_NOTE,
    build_report_figures,
)


class _View:
    """`SymmetrizationView` のうち、図の組み立てが読む属性だけを持つ stub。"""

    def __init__(self, *, warped: np.ndarray | None) -> None:
        self.overlay = np.full((40, 40, 3), 100, dtype=np.uint8)
        self.warped = warped


def _preview() -> np.ndarray:
    return np.full((32, 48, 3), 200, dtype=np.uint8)


def test_nothing_is_built_when_figures_are_switched_off() -> None:
    """罠: 呼び出し側それぞれで判断する実装は、片方だけ OFF を無視する。"""
    figures = build_report_figures(
        include_figures=False,
        preview=_preview(),
        view=_View(warped=np.zeros((8, 8, 3), dtype=np.uint8)),
    )

    assert figures == ()


def test_the_landmark_preview_becomes_a_figure() -> None:
    figures = build_report_figures(include_figures=True, preview=_preview())

    assert [figure.caption for figure in figures] == [LANDMARK_CAPTION]
    assert figures[0].data_uri.startswith("data:image/jpeg;base64,")


def test_the_symmetrisation_overlay_and_warp_are_separate_figures() -> None:
    figures = build_report_figures(
        include_figures=True,
        preview=_preview(),
        view=_View(warped=np.full((40, 40, 3), 50, dtype=np.uint8)),
    )

    assert [figure.caption for figure in figures] == [
        LANDMARK_CAPTION,
        OVERLAY_CAPTION,
        WARPED_CAPTION,
    ]


def test_an_absent_warp_leaves_only_the_overlay() -> None:
    """FR-035 は既定 OFF。生成していない画像の枠を作らない。"""
    figures = build_report_figures(include_figures=True, view=_View(warped=None))

    assert [figure.caption for figure in figures] == [OVERLAY_CAPTION]


def test_the_warped_image_is_declared_as_processed() -> None:
    """不変条件 8: 但し書きは図と同じファイルの中に無ければならない。"""
    figures = build_report_figures(
        include_figures=True,
        view=_View(warped=np.full((40, 40, 3), 50, dtype=np.uint8)),
    )

    warped = next(figure for figure in figures if figure.caption == WARPED_CAPTION)
    assert warped.note == WARPED_NOTE
    assert "加工画像" in warped.note


def test_a_session_without_anything_to_show_yields_no_figures() -> None:
    assert build_report_figures(include_figures=True) == ()


def test_an_unencodable_image_is_dropped_instead_of_raising() -> None:
    """不変条件 12: 1 枚の失敗が、数値と来歴まで道連れにしない。"""
    broken = _View(warped=np.zeros((4, 4), dtype=np.uint8))  # (H, W) は符号化できない

    figures = build_report_figures(include_figures=True, preview=_preview(), view=broken)

    assert [figure.caption for figure in figures] == [LANDMARK_CAPTION, OVERLAY_CAPTION]
