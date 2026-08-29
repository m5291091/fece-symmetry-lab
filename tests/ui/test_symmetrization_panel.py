"""Camera と Image が共有する対称化パネル（FR-034 / FR-035）。

**この module が存在する理由そのものがテスト対象である。** 以前は同じ
オーバーレイ・同じ 2 つのメトリクス・同じ 3 つの注記が `ui.camera` と
`ui.app` へ二重に書かれていた。§5.3/§10.7 が最も強く規定する文言（対称化した
写真は「補正済みの写真」と最も取り違えられやすい出力）なので、片方だけが古い
枠付けのまま残る事故を構造で防ぐ。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.ui.symmetrization_panel import (
    DEVIATION_UNIT_CAPTION,
    SYMMETRIZED_IMAGE_CAPTION,
    SYMMETRIZED_IMAGE_LIMITS,
    SYMMETRIZED_IMAGE_NOTICE,
    WARP_CHECKBOX_HELP,
    WARP_CHECKBOX_LABEL,
)


def _warp_view(*, warped, reason=None):
    """描画側が読む項目だけを持つ `SymmetrizationView` のスタブ。"""
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _View:
        warped: Any
        warp_unavailable_reason: Any

    return _View(warped=warped, warp_unavailable_reason=reason)


def _full_view(*, warped=None, reason=None):
    """`render_symmetrization_view` が読む項目まで揃えたスタブ。"""
    from dataclasses import dataclass
    from typing import Any

    import numpy as np

    @dataclass
    class _View:
        overlay: Any
        mean_deviation: Any
        max_deviation: Any
        symmetrized_point_count: int
        warped: Any
        warp_unavailable_reason: Any

    return _View(
        overlay=np.zeros((8, 8, 3), dtype=np.uint8),
        mean_deviation=0.0123,
        max_deviation=0.0456,
        symmetrized_point_count=126,
        warped=warped,
        warp_unavailable_reason=reason,
    )


# --- 共有パネルの並びと文言 ---------------------------------------------------


def test_the_shared_view_shows_the_overlay_both_deviations_and_the_unit_note() -> None:
    """§8.6.2: オーバーレイ・平均/最大のズレ・点数・単位注記を 1 か所で出す。

    罠: Camera と Image がこれを別々に書いていた頃は、片方だけ注記を落とす
    変更が通ってしまった。
    """
    at = AppTest.from_string(
        "from tests.ui.test_symmetrization_panel import _full_view\n"
        "from face_symmetry_lab.ui.symmetrization_panel import render_symmetrization_view\n"
        "render_symmetrization_view(_full_view(), warp_requested=False)\n"
    )
    at.run()

    assert not at.exception
    assert at.image
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["平均のズレ"] == "1.23%"
    assert metrics["最大のズレ"] == "4.56%"
    captions = " ".join(element.value for element in at.caption)
    assert "126 点" in captions
    assert DEVIATION_UNIT_CAPTION in captions


def test_the_deviation_note_explains_the_unit_without_a_value_judgement() -> None:
    """§5.3: ズレは §8.5 のペア距離の半分であり、対称度とは別の量である。"""
    assert "両眼の間隔を 100%" in DEVIATION_UNIT_CAPTION
    for forbidden in ("良い", "悪い", "理想", "正解", "補正後", "美化"):
        assert forbidden not in DEVIATION_UNIT_CAPTION


def test_the_warp_checkbox_is_off_by_default_and_says_the_output_is_manipulated() -> None:
    """§8.6.4: 全画素を作り直すため既定 OFF の明示操作にする。

    ラベルと help は Camera / Image で同じものを使う（同じ操作が同じ言葉で
    説明されるべきである）。
    """
    at = AppTest.from_string(
        "from face_symmetry_lab.ui.symmetrization_panel import render_warp_checkbox\n"
        "render_warp_checkbox(key='probe')\n"
    )
    at.run()

    assert not at.exception
    assert at.checkbox[0].value is False
    assert at.checkbox[0].label == WARP_CHECKBOX_LABEL
    assert "加工画像" in WARP_CHECKBOX_HELP


# --- FR-035 対称化画像の枠付け ----------------------------------------------


def test_the_warped_image_is_not_rendered_unless_it_was_requested() -> None:
    """§8.6.4 は任意指定である。要求されていない変形については何も述べない。"""
    at = AppTest.from_string(
        "import numpy as np, streamlit as st\n"
        "from face_symmetry_lab.ui.symmetrization_panel import render_symmetrized_image\n"
        "from tests.ui.test_symmetrization_panel import _warp_view\n"
        "render_symmetrized_image(_warp_view(warped=None), requested=False)\n"
    )
    at.run()

    assert not at.exception
    assert not at.info
    assert not at.image


def test_a_refused_warp_explains_itself_instead_of_rendering_nothing() -> None:
    """AC-SYMM-02: 理由を画面に出したうえで fail-soft する。"""
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.symmetrization_panel import render_symmetrized_image\n"
        "from tests.ui.test_symmetrization_panel import _warp_view\n"
        "render_symmetrized_image(\n"
        "    _warp_view(warped=None, reason='変形量が大きすぎるため画像を生成しません。'),\n"
        "    requested=True,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    assert any("変形量" in element.value for element in at.info)


def test_a_refused_warp_keeps_the_point_overlay_alive() -> None:
    """AC-SYMM-02: 画像化の失敗はフェーズ 1（点の可視化）を無効にしない。"""
    at = AppTest.from_string(
        "from tests.ui.test_symmetrization_panel import _full_view\n"
        "from face_symmetry_lab.ui.symmetrization_panel import render_symmetrization_view\n"
        "render_symmetrization_view(\n"
        "    _full_view(warped=None, reason='変位ガードを超えました。'),\n"
        "    warp_requested=True,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    assert at.image  # 点のオーバーレイは残る
    assert any("変位ガード" in element.value for element in at.info)


def test_the_warped_image_is_labelled_as_a_manipulated_image() -> None:
    """§5.3/§10.7: 対称化した写真が、無加工の写真として通ってはならない。"""
    at = AppTest.from_string(
        "import numpy as np\n"
        "from face_symmetry_lab.ui.symmetrization_panel import render_symmetrized_image\n"
        "from tests.ui.test_symmetrization_panel import _warp_view\n"
        "render_symmetrized_image(\n"
        "    _warp_view(warped=np.zeros((8, 8, 3), dtype=np.uint8)),\n"
        "    requested=True,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    assert at.image
    rendered = " ".join(element.value for element in at.caption)
    assert "加工画像" in rendered
    assert "実際の顔写真ではなく" in rendered


def test_the_warped_image_wording_avoids_a_value_judgement() -> None:
    """§10.7: 規範でも、補正でも、before/after でもない。

    関数のソースではなく利用者に見える文字列に対して検証する。禁止する枠付けを
    禁じるために名指しする docstring が、このチェックを落とさないようにする
    ためである。
    """
    shown = SYMMETRIZED_IMAGE_CAPTION + SYMMETRIZED_IMAGE_NOTICE + SYMMETRIZED_IMAGE_LIMITS

    for forbidden in ("理想", "正解", "補正後", "美化", "before", "after"):
        assert forbidden not in shown
    # そして、その画像が実際に何であるかを述べ続けなければならない。
    assert "加工画像" in shown


def test_image_reaches_the_shared_panel_instead_of_writing_its_own() -> None:
    """重複の再発を構造で止める。

    2026-08-29（項目 33）: 対称化と一発取り込みは Camera から削除され、Image
    だけが持つ機能になった（FR-034）。**Camera がこの module を呼ばないこと**
    まで固定する——罠: どちらかの module が `st.image(view.overlay, ...)` や
    「平均のズレ」を自前で書き直すと、この module を通らない 2 つ目の表示規則が
    復活する。
    """
    import inspect

    from face_symmetry_lab.ui import camera, image_section

    image_source = inspect.getsource(image_section)
    assert "render_symmetrization_view" in image_source
    assert "平均のズレ" not in image_source
    assert "最大のズレ" not in image_source
    assert "view.overlay" not in image_source

    assert "render_symmetrization_view" not in inspect.getsource(camera)
