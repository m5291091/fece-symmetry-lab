import tomllib
from pathlib import Path

from face_symmetry_lab.ui.app import (
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


def test_the_image_symmetrization_panel_reports_an_unusable_frame() -> None:
    """FR-034: 折り返せないフレームは、落ちるのではなく理由を説明する。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(
        "import numpy as np\n"
        "import streamlit as st\n"
        "from face_symmetry_lab.application.analyzer import AnalysisOutcome\n"
        "from face_symmetry_lab.application.still_image import (\n"
        "    StillImageAnalysis,\n"
        "    StillImageLandmarks,\n"
        ")\n"
        "from face_symmetry_lab.domain.landmarks import build_landmark_frame\n"
        "from face_symmetry_lab.domain.models import AnalysisState, Status\n"
        "from face_symmetry_lab.ui.app import _render_image_symmetrization\n"
        "rgb = np.zeros((20, 20, 3), dtype=np.uint8)\n"
        "frame = build_landmark_frame(frame_id=1, face_count=0, width=20, height=20)\n"
        "_render_image_symmetrization(\n"
        "    rgb,\n"
        "    StillImageAnalysis(\n"
        "        landmarks=StillImageLandmarks(frame=frame, preview=rgb),\n"
        "        outcome=AnalysisOutcome(\n"
        "            frame_id=1,\n"
        "            state=AnalysisState(status=Status.NO_FACE),\n"
        "            symmetry=None,\n"
        "            overall_distance_smooth=None,\n"
        "            pose=None,\n"
        "            face_area_ratio=None,\n"
        "            metric_version='metric/test',\n"
        "            landmark_map_version='map/test',\n"
        "        ),\n"
        "    ),\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    assert at.info  # 黙って空白にせず、理由を説明する


def test_the_image_symmetrization_panel_avoids_value_laden_wording() -> None:
    """§5.3/§10.7: 対称化した顔は構成物であって、規範ではない。"""
    import inspect

    from face_symmetry_lab.ui import app

    source = inspect.getsource(app._render_image_symmetrization)

    for forbidden in ("理想", "正解", "補正後", "あるべき", "完璧", "欠点"):
        assert forbidden not in source


# --- FR-004 入力方法の選択 --------------------------------------------------


def _input_selector_app():
    from streamlit.testing.v1 import AppTest

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
