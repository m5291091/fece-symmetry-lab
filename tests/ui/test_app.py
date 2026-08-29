"""`ui.app`: Analyze ページの骨格（同意 → 入力方法の選択 → 委譲）。

入力モードそれぞれの中身は `test_camera.py`（Camera）と
`test_image_section.py`（Image）が受け持つ。
"""

import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.ui.app import (
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


def test_the_consent_gate_carries_nothing_but_its_three_items() -> None:
    """§0 項目 30（H7, 2026-08-29）: 進行状況の caption を削除する。

    expander ラベルと 3 つの checkbox の文言は §5.3.1「同意 3 項目はこの分量
    規律の対象外」により 1 文字も変えない。削るのは残り項目数の caption だけ
    である。

    罠: caption を 1 本だけ残す実装を塞ぐ。
    """
    import inspect

    from face_symmetry_lab.ui import app

    at = _consent_gate_app()
    at.run()

    assert not at.exception
    assert len(at.checkbox) == 3
    assert list(at.caption) == []
    assert "st.caption(" not in inspect.getsource(app._render_consent_gate)


# --- FR-004 入力方法の選択 --------------------------------------------------


def _input_selector_app():
    return AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.app import _render_input_mode_selector\n"
        "st.session_state['selected'] = _render_input_mode_selector()\n"
    )


def test_the_input_selector_moves_its_descriptions_into_the_tooltip() -> None:
    """FR-004（2026-08-29 改訂）: 選択肢のラベルは出るが、説明は `help` へ移す。

    §5.3.1 は常時表示の説明カードを撤去し、選ぶ前に読める場所を `help` の
    ツールチップへ変えた。**ラベルの確認は `at.segmented_control[0].options`
    （`format_func` 適用後の実際の表示値を `AppTest` が保持している）で行う**
    ——`st.segmented_control` は Material アイコン記法を `content_icon` へ
    分離するため、`options` はアイコンを含まないラベル文字列そのものになる
    （実機で確認済み）。

    罠: 説明ごと消してしまう実装は、3 番目の assert で落ちる。FR-004 は
    「選ぶ前に読み取れる形」を要求しており、消すことは許されていない。
    """
    at = _input_selector_app()
    at.run()

    assert not at.exception
    labels = at.segmented_control[0].options
    assert "Webカメラ" in labels
    assert "画像アップロード" in labels

    rendered = " ".join(
        element.value for element in list(at.markdown) + list(at.caption) + list(at.info)
    )
    assert "JPEG / PNG" not in rendered

    from face_symmetry_lab.ui import app

    assert "JPEG / PNG" in app._input_mode_help()


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
    """FR-004: モード変更の hook が、もう一方の入力の状態を破棄する。

    カード経由の切り替え（`_select_input_mode`）は 2026-08-29 の改訂で説明
    カードごと削除されたため、runtime 解放の確認はこの経路だけが単独で担う。
    """
    import inspect

    from face_symmetry_lab.ui import app

    source = inspect.getsource(app._switch_input_mode)

    assert "release_session_pipeline" in source
    assert "release_image_landmarker" in source


def test_the_input_selector_leaves_no_card_no_badge_and_no_extra_button() -> None:
    """FR-004（2026-08-29 改訂）: 説明カード・バッジ・切り替えボタンを持たない。

    選択の実体は `st.segmented_control`（AC-A11Y-01 のためのネイティブ widget）
    のままだが、以前その下に並んでいた枠付きカード・「選択中」バッジ・切り替え
    ボタンはすべて撤去する。

    罠: カードだけ消してボタンを残す実装、あるいは `_select_input_mode` を
    呼ばれないまま残す実装は、ここで落ちる。
    """
    import inspect

    from face_symmetry_lab.ui import app

    at = _input_selector_app()
    at.run()

    assert not at.exception
    assert list(at.button) == []

    source = inspect.getsource(app._render_input_mode_selector)
    assert "st.container(border=True)" not in source
    assert ".badge(" not in source
    assert "st.columns(" not in source

    assert not hasattr(app, "_select_input_mode")


# --- §5.1（2026-08-29）: Analyze ページに「Face Symmetry Lab について」を置かない ---


def test_the_analyze_page_carries_no_about_footer() -> None:
    """§5.1 / 項目 31（H8, 2026-08-29）: ページ最下部の説明区画を削除した。

    算出方法・制約・禁止用途は About ページが唯一の正本として持ち（§5.3.1）、
    About と参照統計へはサイドバーの multipage nav から到達する。

    罠: 関数と定数を残したまま呼び出しだけ消す実装は、次に触る人が「まだ使って
    いる」と読んで復活させる。`hasattr` まで見るのはそのためである。
    """
    import inspect

    from face_symmetry_lab.ui import app

    for removed in (
        "_render_about_footer",
        "FOOTER_FACTS",
        "ABOUT_FOOTER_SUMMARY",
        "ABOUT_FOOTER_NOTICE",
    ):
        assert not hasattr(app, removed), removed

    source = inspect.getsource(app.render_app)
    assert "st.page_link(" not in source
    assert "Face Symmetry Lab について" not in source

    # §4.1: アプリ名とプライバシー境界の 1 行は残す（削減の対象ではない）。
    assert "st.title(" in source
    assert "LOCAL PROCESSING" in source


def test_the_navigation_still_lists_all_three_pages() -> None:
    """§4.1: フッターの導線を消しても、multipage nav の 3 ページは残す。

    罠: フッターを消すついでに参照統計ページ・About ページごとナビから外す実装を
    塞ぐ。
    """
    import inspect

    from face_symmetry_lab.ui import app

    assert len(app.build_pages()) == 3

    # `StreamlitPage.title` / `.icon` はスクリプト実行文脈の外では読めないため
    # （`_title` が遅延設定される）、中身は source で確かめる。
    builder = inspect.getsource(app.build_pages)
    for target in (
        "render_app",
        "render_reference_page",
        "render_about_page",
        '"参照統計"',
        '"使い方・算出方法"',
    ):
        assert target in builder, target
    assert "st.navigation(build_pages())" in inspect.getsource(app)
