"""`ui.app`: Analyze ページの骨格（同意 → 入力方法の選択 → 委譲）。

入力モードそれぞれの中身は `test_camera.py`（Camera）と
`test_image_section.py`（Image）が受け持つ。
"""

import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.ui.app import (
    ACTIVE_INPUT_MODE_KEY,
    CONSENT_PRIVACY_KEY,
    CONSENT_RESEARCH_KEY,
    CONSENT_RIGHTS_KEY,
    INPUT_MODE_CAMERA,
    INPUT_MODE_IMAGE,
    INPUT_MODE_STATE_KEY,
)
from face_symmetry_lab.ui.model_setup import (
    DEMOGRAPHIC_ONNX_PATH_ENV_VAR,
    configured_demographic_path,
)

STREAMLIT_CONFIG_PATH = Path(__file__).parents[2] / ".streamlit" / "config.toml"


def test_configured_model_path_defaults_to_repository_models(monkeypatch) -> None:
    monkeypatch.delenv(DEMOGRAPHIC_ONNX_PATH_ENV_VAR, raising=False)

    onnx_path = configured_demographic_path()

    assert onnx_path.name == "age-gender-recognition-retail-0013.onnx"
    assert onnx_path.parent.name == "models"


def test_configured_model_path_uses_explicit_local_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "age.onnx"
    monkeypatch.setenv(DEMOGRAPHIC_ONNX_PATH_ENV_VAR, str(onnx_path))

    assert configured_demographic_path() == onnx_path


def test_streamlit_uses_the_dark_product_theme() -> None:
    config = tomllib.loads(STREAMLIT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["theme"]["base"] == "dark"
    assert config["theme"]["primaryColor"] == "#38BDF8"
    assert config["theme"]["backgroundColor"] == "#07111F"
    assert config["theme"]["secondaryBackgroundColor"] == "#111C2E"


# --- 不変条件 3: 同意ゲート ---------------------------------------------------


def _consent_gate_app():
    return AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.app import _render_consent_gate\n"
        "st.session_state['consented'] = _render_consent_gate()\n"
    )


def test_the_consent_gate_starts_closed_and_lists_all_three_items() -> None:
    """不変条件 3: 画像 upload と camera permission の前に明示同意を取る。"""
    at = _consent_gate_app()
    at.run()

    assert not at.exception
    assert at.session_state["consented"] is False
    assert len(at.checkbox) == 3


def test_the_consent_gate_opens_only_when_all_three_are_checked() -> None:
    """罠: `any` で判定する実装は、1 つ目だけでカメラを開けてしまう。"""
    at = _consent_gate_app()
    at.session_state[CONSENT_PRIVACY_KEY] = True
    at.session_state[CONSENT_RESEARCH_KEY] = True
    at.run()
    assert at.session_state["consented"] is False

    at.session_state[CONSENT_RIGHTS_KEY] = True
    at.run()

    assert not at.exception
    assert at.session_state["consented"] is True


def test_the_consent_gate_shows_how_many_items_are_left() -> None:
    """§4.2: 3 項目のうち何が残っているかを、開く前に読めるようにする。

    罠: 進捗を出さない実装（「3 項目を確認すると…」だけ）は、2 つ確認した
    利用者に何が残っているかを伝えられない。
    """
    at = _consent_gate_app()
    at.session_state[CONSENT_PRIVACY_KEY] = True
    at.session_state[CONSENT_RESEARCH_KEY] = True
    at.run()

    progress = " ".join(element.value for element in at.caption)

    assert "1" in progress, "残り 1 項目であることを示すこと"
    assert "3" in progress, "全体が 3 項目であることを示すこと"


# --- FR-004 入力方法の選択 --------------------------------------------------


def _input_selector_app():
    return AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.app import _render_input_mode_selector\n"
        "st.session_state['selected'] = _render_input_mode_selector()\n"
    )


def test_the_input_selector_describes_both_modes_before_one_is_chosen() -> None:
    """FR-004: 各入力が何をするかは、選ぶ前に読める。"""
    at = _input_selector_app()
    at.run()

    assert not at.exception
    rendered = " ".join(element.value for element in list(at.markdown) + list(at.caption))
    assert "Web カメラ" in rendered
    assert "画像ファイル" in rendered
    assert "JPEG / PNG" in rendered


def test_the_input_selector_defaults_to_camera_and_marks_the_active_mode() -> None:
    at = _input_selector_app()
    at.run()

    assert at.session_state["selected"] == INPUT_MODE_CAMERA


def test_deselecting_keeps_the_previous_input_active() -> None:
    """§15.1: 常にどれかの入力が有効である。選択解除がページを空にしてはならない。"""
    at = _input_selector_app()
    at.session_state[INPUT_MODE_STATE_KEY] = INPUT_MODE_IMAGE
    at.run()
    assert at.session_state["selected"] == INPUT_MODE_IMAGE

    # `st.segmented_control` は選択中の項目を押すと解除できる。
    at.session_state[INPUT_MODE_STATE_KEY] = None
    at.run()

    assert not at.exception
    assert at.session_state["selected"] == INPUT_MODE_IMAGE


def test_switching_the_input_releases_the_previous_runtime() -> None:
    """FR-004: モード変更の hook が、もう一方の入力の状態を破棄する。"""
    import inspect

    from face_symmetry_lab.ui import app

    source = inspect.getsource(app._switch_input_mode)

    assert "release_session_pipeline" in source
    assert "release_image_landmarker" in source


def test_each_unselected_mode_card_can_actually_be_chosen() -> None:
    """説明カードは「押せそうで押せない面」であってはならない。

    選択の実体は `st.segmented_control`（AC-A11Y-01 のためのネイティブ widget）
    のままだが、その下に並ぶ枠付きカードは実ブラウザで押せる面に見える。
    未選択のカードには、そのモードへ切り替える操作を必ず持たせる。
    """
    at = _input_selector_app()
    at.run()
    assert at.session_state["selected"] == INPUT_MODE_CAMERA

    switches = [element for element in at.button if INPUT_MODE_IMAGE in element.key]
    assert len(switches) == 1, "未選択のモードにだけ切り替え操作を出すこと"

    switches[0].click().run()

    assert not at.exception
    assert at.session_state["selected"] == INPUT_MODE_IMAGE
    assert at.session_state[ACTIVE_INPUT_MODE_KEY] == INPUT_MODE_IMAGE
    # 選択済みのカードは操作を持たない（「選択中」バッジだけ）。
    assert [element for element in at.button if INPUT_MODE_IMAGE in element.key] == []


def test_choosing_a_mode_from_a_card_releases_the_previous_runtime() -> None:
    """FR-004: カード経由の切り替えも `st.segmented_control` と同じ後始末をする。

    罠: `on_click` で session state だけ書き換える実装は、直前の入力の landmark
    runtime を掴んだままにする（§10.4）。
    """
    import inspect

    from face_symmetry_lab.ui import app

    source = inspect.getsource(app._select_input_mode)

    assert "_switch_input_mode" in source


# --- §5.1 フッター: 実態と食い違う説明を置かない ------------------------------
#
# `_render_about_footer` は `st.page_link` で `st.Page` を参照するため、
# `st.navigation` を通らない `AppTest` では検証できない（Streamlit 1.59.1 は
# 未登録ページの検証で AttributeError を投げる）。文言は定数として、構成は
# source として確かめる。


def test_the_footer_does_not_claim_scores_sit_beside_the_video() -> None:
    """項目26（H3, 2026-08-12）/ AC-LAYOUT-01 の回帰。

    映像の隣の「ライブ解析」パネルは削除済みで、映像・画像に隣接して数値を
    出すのは HUD と吹き出しだけである。**罠: 旧仕様の「スコアは映像の隣に
    表示します」を残した実装は、画面に存在しない区画を案内する。**
    """
    from face_symmetry_lab.ui.app import ABOUT_FOOTER_SUMMARY

    assert "映像の隣" not in ABOUT_FOOTER_SUMMARY


def test_the_footer_names_where_the_numbers_actually_appear() -> None:
    """§5.1: 数値は映像上の HUD と画像上の吹き出しに出る。"""
    from face_symmetry_lab.ui.app import ABOUT_FOOTER_SUMMARY

    assert "HUD" in ABOUT_FOOTER_SUMMARY
    assert "吹き出し" in ABOUT_FOOTER_SUMMARY


def test_the_footer_links_to_both_secondary_pages() -> None:
    """§4.1: Analyze から About・参照統計のどちらへも主画面から到達できる。"""
    import inspect

    from face_symmetry_lab.ui import app

    source = inspect.getsource(app._render_about_footer)

    assert source.count("st.page_link(") == 2
    assert "ABOUT_PAGE" in source
    assert "REFERENCE_PAGE" in source


def test_the_footer_does_not_dress_static_facts_as_metrics() -> None:
    """§5.2: 測定値でないものを `st.metric` の大きな数字で見せない。

    フッターの「処理境界 / 端末内のみ」等は静的な事実であり、対称度の結果より
    視覚的に重くなってはならない。
    """
    import inspect

    from face_symmetry_lab.ui import app

    source = inspect.getsource(app._render_about_footer)

    assert ".metric(" not in source
    assert app.FOOTER_FACTS, "事実そのものは残す（metric を使わないだけである）"
