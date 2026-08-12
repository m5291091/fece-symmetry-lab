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
