import tomllib
from pathlib import Path

from face_symmetry_lab.ui.model_setup import (
    DEMOGRAPHIC_BIN_PATH_ENV_VAR,
    DEMOGRAPHIC_XML_PATH_ENV_VAR,
    configured_demographic_paths,
)

STREAMLIT_CONFIG_PATH = Path(__file__).parents[2] / ".streamlit" / "config.toml"


def test_configured_model_paths_default_to_repository_models(monkeypatch) -> None:
    monkeypatch.delenv(DEMOGRAPHIC_XML_PATH_ENV_VAR, raising=False)
    monkeypatch.delenv(DEMOGRAPHIC_BIN_PATH_ENV_VAR, raising=False)

    xml_path, bin_path = configured_demographic_paths()

    assert xml_path.name == "age-gender-recognition-retail-0013.xml"
    assert bin_path.name == "age-gender-recognition-retail-0013.bin"
    assert xml_path.parent.name == bin_path.parent.name == "models"


def test_configured_model_paths_use_explicit_local_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "age.xml"
    bin_path = tmp_path / "age.bin"
    monkeypatch.setenv(DEMOGRAPHIC_XML_PATH_ENV_VAR, str(xml_path))
    monkeypatch.setenv(DEMOGRAPHIC_BIN_PATH_ENV_VAR, str(bin_path))

    assert configured_demographic_paths() == (xml_path, bin_path)


def test_streamlit_uses_the_dark_product_theme() -> None:
    config = tomllib.loads(STREAMLIT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["theme"]["base"] == "dark"
    assert config["theme"]["primaryColor"] == "#38BDF8"
    assert config["theme"]["backgroundColor"] == "#07111F"
    assert config["theme"]["secondaryBackgroundColor"] == "#111C2E"
