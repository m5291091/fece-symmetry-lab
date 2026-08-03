import tomllib
from pathlib import Path

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
    """FR-034: a frame that cannot be folded explains itself, never crashes."""
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
    assert at.info  # explained, not silently blank


def test_the_image_symmetrization_panel_avoids_value_laden_wording() -> None:
    """§5.3/§10.7: a symmetrized face is a construction, not an ideal."""
    import inspect

    from face_symmetry_lab.ui import app

    source = inspect.getsource(app._render_image_symmetrization)

    for forbidden in ("理想", "正解", "補正後", "あるべき", "完璧", "欠点"):
        assert forbidden not in source
