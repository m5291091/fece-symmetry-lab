from hashlib import sha256
from pathlib import Path

import face_symmetry_lab.ui.model_setup as model_setup
from face_symmetry_lab.adapters.io.model_store import InstallStatus, ModelAsset
from face_symmetry_lab.ui.model_setup import (
    DEMOGRAPHIC_BIN_PATH_ENV_VAR,
    DEMOGRAPHIC_XML_PATH_ENV_VAR,
    LANDMARKER_PATH_ENV_VAR,
    configured_demographic_paths,
    configured_landmarker_path,
    demographic_model_status,
    landmark_asset,
    landmarker_status,
)


def test_landmark_asset_comes_from_the_pinned_manifest() -> None:
    asset = landmark_asset()

    assert asset.name == "face_landmarker"
    assert asset.filename.endswith(".task")
    assert len(asset.sha256) == 64
    assert asset.url.startswith("https://")


def test_default_path_is_the_models_directory_entry_from_the_manifest(monkeypatch) -> None:
    monkeypatch.delenv(LANDMARKER_PATH_ENV_VAR, raising=False)

    path = configured_landmarker_path()

    assert path.name == landmark_asset().filename
    assert path.parent.name == "models"


def test_path_can_be_overridden_with_a_local_file(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom_landmarker.task"
    monkeypatch.setenv(LANDMARKER_PATH_ENV_VAR, str(override))

    assert configured_landmarker_path() == override


def test_absent_model_reports_missing(tmp_path: Path) -> None:
    assert landmarker_status(tmp_path / "absent.task") is InstallStatus.MISSING


def test_a_file_that_is_not_the_pinned_asset_reports_mismatch(tmp_path: Path) -> None:
    impostor = tmp_path / "face_landmarker.task"
    impostor.write_bytes(b"not the pinned model")

    assert landmarker_status(impostor) is InstallStatus.MISMATCH


def test_default_demographic_paths_come_from_the_pinned_manifest(monkeypatch) -> None:
    monkeypatch.delenv(DEMOGRAPHIC_XML_PATH_ENV_VAR, raising=False)
    monkeypatch.delenv(DEMOGRAPHIC_BIN_PATH_ENV_VAR, raising=False)

    xml_path, bin_path = configured_demographic_paths()

    assert xml_path.name == "age-gender-recognition-retail-0013.xml"
    assert bin_path.name == "age-gender-recognition-retail-0013.bin"
    assert xml_path.parent == bin_path.parent
    assert xml_path.parent.name == "models"


def test_demographic_model_requires_both_verified_ir_files(tmp_path: Path) -> None:
    xml_path = tmp_path / "age-gender-recognition-retail-0013.xml"
    bin_path = tmp_path / "age-gender-recognition-retail-0013.bin"

    assert demographic_model_status(xml_path, bin_path) is InstallStatus.MISSING

    xml_path.write_bytes(b"not the pinned XML")
    assert demographic_model_status(xml_path, bin_path) is InstallStatus.MISMATCH

    bin_path.write_bytes(b"not the pinned BIN")
    assert demographic_model_status(xml_path, bin_path) is InstallStatus.MISMATCH


def test_demographic_model_reports_ready_for_a_verified_pair(
    monkeypatch,
    tmp_path: Path,
) -> None:
    xml_bytes = b"pinned XML"
    bin_bytes = b"pinned BIN"
    xml_asset = ModelAsset(
        name="demographic_age_gender_xml",
        version="test/FP16",
        filename="age.xml",
        url="https://storage.openvinotoolkit.org/age.xml",
        size_bytes=len(xml_bytes),
        sha256=sha256(xml_bytes).hexdigest(),
        license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        input_contract="test XML",
    )
    bin_asset = ModelAsset(
        name="demographic_age_gender_bin",
        version="test/FP16",
        filename="age.bin",
        url="https://storage.openvinotoolkit.org/age.bin",
        size_bytes=len(bin_bytes),
        sha256=sha256(bin_bytes).hexdigest(),
        license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        input_contract="test BIN",
    )
    monkeypatch.setattr(model_setup, "demographic_assets", lambda: (xml_asset, bin_asset))
    xml_path = tmp_path / xml_asset.filename
    bin_path = tmp_path / bin_asset.filename
    xml_path.write_bytes(xml_bytes)
    bin_path.write_bytes(bin_bytes)

    assert demographic_model_status(xml_path, bin_path) is InstallStatus.READY
