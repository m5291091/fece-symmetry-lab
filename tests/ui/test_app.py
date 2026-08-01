import tomllib
from pathlib import Path

from face_symmetry_lab.ui.app import DEFAULT_AGE_MODEL, _configured_model_path

STREAMLIT_CONFIG_PATH = Path(__file__).parents[2] / ".streamlit" / "config.toml"


def test_configured_model_path_defaults_to_repository_model(monkeypatch) -> None:
    monkeypatch.delenv("FACE_SYMMETRY_LAB_AGE_MODEL_PATH", raising=False)

    assert _configured_model_path() == DEFAULT_AGE_MODEL


def test_configured_model_path_uses_explicit_local_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "age.onnx"
    monkeypatch.setenv("FACE_SYMMETRY_LAB_AGE_MODEL_PATH", str(model_path))

    assert _configured_model_path() == model_path


def test_streamlit_uses_the_dark_product_theme() -> None:
    config = tomllib.loads(STREAMLIT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["theme"]["base"] == "dark"
    assert config["theme"]["primaryColor"] == "#38BDF8"
    assert config["theme"]["backgroundColor"] == "#07111F"
    assert config["theme"]["secondaryBackgroundColor"] == "#111C2E"
