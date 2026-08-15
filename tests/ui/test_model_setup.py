from hashlib import sha256
from pathlib import Path

import pytest

import face_symmetry_lab.ui.model_setup as model_setup
from face_symmetry_lab.adapters.io.model_store import DerivedModelAsset, InstallStatus
from face_symmetry_lab.application.age_gate import AgeGateService
from face_symmetry_lab.ui.model_setup import (
    DEMOGRAPHIC_ONNX_PATH_ENV_VAR,
    LANDMARKER_PATH_ENV_VAR,
    SEGMENTER_PATH_ENV_VAR,
    build_age_gate_service,
    configured_demographic_path,
    configured_landmarker_path,
    configured_segmenter_path,
    demographic_model_status,
    landmark_asset,
    landmarker_status,
    segmenter_asset,
    segmenter_status,
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


def test_default_demographic_path_comes_from_the_pinned_manifest(monkeypatch) -> None:
    monkeypatch.delenv(DEMOGRAPHIC_ONNX_PATH_ENV_VAR, raising=False)

    onnx_path = configured_demographic_path()

    assert onnx_path.name == "age-gender-recognition-retail-0013.onnx"
    assert onnx_path.parent.name == "models"


def test_demographic_model_requires_the_verified_derived_onnx(tmp_path: Path) -> None:
    onnx_path = tmp_path / "age-gender-recognition-retail-0013.onnx"

    assert demographic_model_status(onnx_path) is InstallStatus.MISSING

    onnx_path.write_bytes(b"not the pinned ONNX")
    assert demographic_model_status(onnx_path) is InstallStatus.MISMATCH


def test_segmenter_asset_comes_from_the_pinned_manifest() -> None:
    asset = segmenter_asset()

    assert asset.name == "selfie_segmenter"
    assert asset.filename.endswith(".tflite")
    assert len(asset.sha256) == 64
    assert asset.url.startswith("https://")


def test_default_segmenter_path_is_the_models_directory_entry(monkeypatch) -> None:
    monkeypatch.delenv(SEGMENTER_PATH_ENV_VAR, raising=False)

    path = configured_segmenter_path()

    assert path.name == segmenter_asset().filename
    assert path.parent.name == "models"


def test_segmenter_path_can_be_overridden_with_a_local_file(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom_segmenter.tflite"
    monkeypatch.setenv(SEGMENTER_PATH_ENV_VAR, str(override))

    assert configured_segmenter_path() == override


def test_absent_segmenter_reports_missing(tmp_path: Path) -> None:
    assert segmenter_status(tmp_path / "absent.tflite") is InstallStatus.MISSING


def test_a_file_that_is_not_the_pinned_segmenter_reports_mismatch(tmp_path: Path) -> None:
    impostor = tmp_path / "selfie_segmenter.tflite"
    impostor.write_bytes(b"not the pinned model")

    assert segmenter_status(impostor) is InstallStatus.MISMATCH


def test_demographic_model_reports_ready_for_a_verified_onnx(
    monkeypatch,
    tmp_path: Path,
) -> None:
    onnx_bytes = b"pinned ONNX"
    onnx_asset = DerivedModelAsset(
        name="demographic_age_gender_onnx",
        version="test/ONNX",
        filename="age.onnx",
        size_bytes=len(onnx_bytes),
        sha256=sha256(onnx_bytes).hexdigest(),
        license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        input_contract="test ONNX",
        source_assets=("demographic_age_gender_xml", "demographic_age_gender_bin"),
        converter="scripts/convert_demographic_model.py",
        converter_version="test-v1",
        tools={"onnx": "1.22.0"},
    )
    monkeypatch.setattr(model_setup, "demographic_asset", lambda: onnx_asset)
    onnx_path = tmp_path / onnx_asset.filename
    onnx_path.write_bytes(onnx_bytes)

    assert demographic_model_status(onnx_path) is InstallStatus.READY


@pytest.mark.model
def test_build_age_gate_service_loads_the_real_bundled_model() -> None:
    """FR-038: bundled資産（§22.5）はリポジトリ同梱なので、既定で classifier を持つ。"""
    service = build_age_gate_service()

    assert isinstance(service, AgeGateService)
    assert service._classifier is not None


def test_build_age_gate_service_falls_back_to_no_classifier_when_the_spec_is_missing(
    monkeypatch,
) -> None:
    """AC-AGEGATE-03: spec が読めなくても例外を投げず、判定不能側の service を返す。"""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("spec が見つかりません")

    monkeypatch.setattr(model_setup, "load_age_gate_spec", _raise)

    service = build_age_gate_service()

    assert isinstance(service, AgeGateService)
    assert service._classifier is None
