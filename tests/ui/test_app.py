import tomllib
from datetime import UTC, datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

import face_symmetry_lab.ui.app as app_module
from face_symmetry_lab.domain.demographics import build_demographic_estimate
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


def _fake_demographic_estimate(model_version: str = "test-demographic/1.0.0"):
    return build_demographic_estimate(
        estimated_age_years=30.0,
        gender_probabilities=(0.3, 0.7),
        model_version=model_version,
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 11, tzinfo=UTC),
    )


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


# --- H2: 画像アップロードの年齢・性別自動実行 --------------------------------

_DEMOGRAPHIC_TRIGGER_SCRIPT = (
    "import numpy as np\n"
    "from face_symmetry_lab.ui.app import (\n"
    "    _maybe_estimate_image_demographics,\n"
    "    _render_image_demographic_detail,\n"
    ")\n"
    "age_input = np.zeros((62, 62, 3), dtype=np.uint8)\n"
    "_maybe_estimate_image_demographics(\n"
    "    age_input, input_fingerprint='fp-1', recorder=None,\n"
    ")\n"
    "_render_image_demographic_detail(age_input)\n"
)


def test_single_face_image_shows_demographics_without_a_button(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """依頼 1・AC-AGE-03: 単一顔が検出できたら、ボタン操作なしに参考年齢・
    参考性別が描画される。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        app_module, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    class _FakeEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path
            self._model_version = model_version

        def estimate(self, rgb_image):
            del rgb_image
            return _fake_demographic_estimate(self._model_version)

    monkeypatch.setattr(app_module, "OpenCvDemographicEstimator", _FakeEstimator)

    at = AppTest.from_string(_DEMOGRAPHIC_TRIGGER_SCRIPT)
    at.run()

    assert not at.exception
    assert not at.button  # ボタンが存在しない
    labels = [metric.label for metric in at.metric]
    assert "参考年齢" in labels
    assert "外見由来の参考性別" in labels


def test_the_estimator_runs_once_per_image_even_after_reruns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """依頼 1・AC-AGE-03: 同じ画像に対して estimator は 1 度しか呼ばれない
    （罠 (a): fingerprint によるキャッシュを外すと、再実行のたびに増える）。
    異なる画像（fingerprint）に対しては新たに 1 度だけ呼ばれる。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        app_module, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    calls = {"n": 0}

    class _CountingEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path
            self._model_version = model_version

        def estimate(self, rgb_image):
            del rgb_image
            calls["n"] += 1
            return _fake_demographic_estimate(self._model_version)

    monkeypatch.setattr(app_module, "OpenCvDemographicEstimator", _CountingEstimator)

    at = AppTest.from_string(
        "import numpy as np\n"
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.app import _maybe_estimate_image_demographics\n"
        "age_input = np.zeros((62, 62, 3), dtype=np.uint8)\n"
        "fingerprint = st.session_state.get('test_fingerprint', 'fp-1')\n"
        "_maybe_estimate_image_demographics(\n"
        "    age_input, input_fingerprint=fingerprint, recorder=None,\n"
        ")\n"
    )
    at.run()
    assert calls["n"] == 1

    # 同じ画像のまま再実行（他ウィジェットの操作を模す）しても増えない。
    at.run()
    assert not at.exception
    assert calls["n"] == 1

    # 新しい画像（別 fingerprint）は新たに 1 度だけ呼ばれる。
    at.session_state["test_fingerprint"] = "fp-2"
    at.run()
    assert not at.exception
    assert calls["n"] == 2


def test_a_failed_estimate_does_not_stop_the_symmetry_panel_from_rendering(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """依頼 1・不変条件 12・FR-024: estimator が例外を投げても、左右差側の
    パネルは描画され続け、年齢・性別区画だけに理由が示される。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        app_module, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    class _FailingEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path, model_version

        def estimate(self, rgb_image):
            del rgb_image
            raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "OpenCvDemographicEstimator", _FailingEstimator)

    at = AppTest.from_string(
        "import numpy as np\n"
        "from face_symmetry_lab.ui.app import (\n"
        "    _maybe_estimate_image_demographics,\n"
        "    _render_image_demographic_detail,\n"
        ")\n"
        "from face_symmetry_lab.ui.scores import render_headline_metrics\n"
        "age_input = np.zeros((62, 62, 3), dtype=np.uint8)\n"
        "_maybe_estimate_image_demographics(\n"
        "    age_input, input_fingerprint='fp-1', recorder=None,\n"
        ")\n"
        # 左右差側は年齢・性別の後に描画される。ここで例外を伝播させないことが
        # 不変条件 12 の核心である。
        "render_headline_metrics(overall=0.02, equal_weight=0.05, estimate=None)\n"
        "_render_image_demographic_detail(age_input)\n"
    )
    at.run()

    assert not at.exception
    assert at.metric[0].label == "全体の対称度"
    assert at.metric[0].value == "98.00%"
    assert any("推定できませんでした" in element.value for element in at.error)


def test_no_face_detected_skips_inference_and_explains_why(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """依頼 1・AC-AGE-03: 顔が検出できない（`age_input is None`）画像では
    estimator を一切呼ばず、理由を説明する。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        app_module, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    calls = {"n": 0}

    class _CountingEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path, model_version

        def estimate(self, rgb_image):
            del rgb_image
            calls["n"] += 1
            return _fake_demographic_estimate()

    monkeypatch.setattr(app_module, "OpenCvDemographicEstimator", _CountingEstimator)

    at = AppTest.from_string(
        "from face_symmetry_lab.ui.app import (\n"
        "    _maybe_estimate_image_demographics,\n"
        "    _render_image_demographic_detail,\n"
        ")\n"
        "_maybe_estimate_image_demographics(\n"
        "    None, input_fingerprint='fp-1', recorder=None,\n"
        ")\n"
        "_render_image_demographic_detail(None)\n"
    )
    at.run()

    assert not at.exception
    assert calls["n"] == 0
    assert any("単一顔" in element.value for element in at.info)


def test_the_demographic_detail_panel_no_longer_mentions_a_button() -> None:
    """依頼 1: ボタンを廃止したので、案内文がボタン操作を前提にしてはならない。"""
    import inspect

    source = inspect.getsource(app_module._render_image_demographic_detail)

    assert "ボタン" not in source


def test_a_missing_demographic_model_explains_itself_without_calling_the_estimator(
    monkeypatch,
) -> None:
    """止まって報告すべきことの確認: モデル未配置でも自動実行の表示は壊れず、
    estimator を呼ばずに理由を説明する（FR-024）。"""
    monkeypatch.setattr(app_module, "resolve_ready_demographic", lambda: None)

    calls = {"n": 0}

    class _UnreachableEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path, model_version

        def estimate(self, rgb_image):
            del rgb_image
            calls["n"] += 1
            return _fake_demographic_estimate()

    monkeypatch.setattr(app_module, "OpenCvDemographicEstimator", _UnreachableEstimator)

    at = AppTest.from_string(_DEMOGRAPHIC_TRIGGER_SCRIPT)
    at.run()

    assert not at.exception
    assert calls["n"] == 0
    assert any("未配置" in element.value for element in at.info)
