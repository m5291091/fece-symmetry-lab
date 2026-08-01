from pathlib import Path

from face_symmetry_lab.adapters.io.model_store import InstallStatus
from face_symmetry_lab.ui.model_setup import (
    LANDMARKER_PATH_ENV_VAR,
    configured_landmarker_path,
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
