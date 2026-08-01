import tomllib
from pathlib import Path

from face_symmetry_lab.ui.app import (
    DEFAULT_DEMOGRAPHIC_BIN,
    DEFAULT_DEMOGRAPHIC_XML,
    _configured_model_paths,
)

STREAMLIT_CONFIG_PATH = Path(__file__).parents[2] / ".streamlit" / "config.toml"


def test_configured_model_paths_default_to_repository_models(monkeypatch) -> None:
    monkeypatch.delenv("FACE_SYMMETRY_LAB_DEMOGRAPHIC_XML_PATH", raising=False)
    monkeypatch.delenv("FACE_SYMMETRY_LAB_DEMOGRAPHIC_BIN_PATH", raising=False)

    assert _configured_model_paths() == (DEFAULT_DEMOGRAPHIC_XML, DEFAULT_DEMOGRAPHIC_BIN)


def test_configured_model_paths_use_explicit_local_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "age.xml"
    bin_path = tmp_path / "age.bin"
    monkeypatch.setenv("FACE_SYMMETRY_LAB_DEMOGRAPHIC_XML_PATH", str(xml_path))
    monkeypatch.setenv("FACE_SYMMETRY_LAB_DEMOGRAPHIC_BIN_PATH", str(bin_path))

    assert _configured_model_paths() == (xml_path, bin_path)


def test_streamlit_uses_the_dark_product_theme() -> None:
    config = tomllib.loads(STREAMLIT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["theme"]["base"] == "dark"
    assert config["theme"]["primaryColor"] == "#38BDF8"
    assert config["theme"]["backgroundColor"] == "#07111F"
    assert config["theme"]["secondaryBackgroundColor"] == "#111C2E"
