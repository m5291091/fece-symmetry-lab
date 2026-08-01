import json
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import pytest

from face_symmetry_lab.adapters.io.model_store import (
    DEFAULT_MANIFEST_PATH,
    DerivedModelAsset,
    InstallStatus,
    ModelAsset,
    ModelStore,
    ModelStoreError,
    find_asset,
    find_derived_asset,
    load_derived_assets,
    load_manifest,
)

PAYLOAD = b"face-landmarker-bytes"
PAYLOAD_SHA256 = sha256(PAYLOAD).hexdigest()


def _asset(**overrides: object) -> dict[str, object]:
    asset: dict[str, object] = {
        "name": "face_landmarker",
        "version": "float16/1",
        "filename": "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker.task",
        "size_bytes": len(PAYLOAD),
        "sha256": PAYLOAD_SHA256,
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "input_contract": "RGB uint8 image",
    }
    asset.update(overrides)
    return asset


def _manifest_file(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "model-manifest.json"
    path.write_text(
        json.dumps({"models": [_asset(**overrides)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _downloader(payload: bytes = PAYLOAD):
    def download(url: str) -> Iterator[bytes]:
        yield payload

    return download


def test_repository_manifest_is_loadable_and_pins_the_landmark_model() -> None:
    assets = load_manifest(DEFAULT_MANIFEST_PATH)

    landmarker = find_asset(assets, "face_landmarker")
    assert landmarker.url.startswith("https://")
    assert len(landmarker.sha256) == 64
    assert landmarker.size_bytes > 0
    assert landmarker.license


def test_repository_manifest_pins_the_official_openvino_demographic_pair() -> None:
    assets = load_manifest(DEFAULT_MANIFEST_PATH)

    xml = find_asset(assets, "demographic_age_gender_xml")
    weights = find_asset(assets, "demographic_age_gender_bin")

    assert xml.version == weights.version == "open-model-zoo/2023.0/FP16"
    assert xml.filename == "age-gender-recognition-retail-0013.xml"
    assert weights.filename == "age-gender-recognition-retail-0013.bin"
    assert xml.size_bytes == 62_543
    assert weights.size_bytes == 4_276_038
    assert xml.sha256 == "2de3182a773f31d2c2ed91197ee069feae340a505add157a4a92067d333cf067"
    assert weights.sha256 == "7c6e3da30e369fd870f6d208fc6b08f7cc15a137c1c9c1030b911a5fad31da13"
    assert xml.url.endswith("/FP16/age-gender-recognition-retail-0013.xml")
    assert weights.url.endswith("/FP16/age-gender-recognition-retail-0013.bin")
    assert xml.license == weights.license == "Apache-2.0"


def test_repository_manifest_pins_the_locally_derived_demographic_onnx() -> None:
    derived_assets = load_derived_assets(DEFAULT_MANIFEST_PATH)

    onnx = find_derived_asset(derived_assets, "demographic_age_gender_onnx")

    assert isinstance(onnx, DerivedModelAsset)
    assert onnx.version == "open-model-zoo/2023.0/FP16-onnx-v1"
    assert onnx.filename == "age-gender-recognition-retail-0013.onnx"
    assert onnx.size_bytes > 4_000_000
    assert len(onnx.sha256) == 64
    assert onnx.source_assets == (
        "demographic_age_gender_xml",
        "demographic_age_gender_bin",
    )
    assert onnx.converter == "scripts/convert_demographic_model.py"
    assert onnx.converter_version == "ir11-to-onnx-opset13-v1"
    assert onnx.tools == {
        "ml-dtypes": "0.5.4",
        "numpy": "2.5.1",
        "onnx": "1.22.0",
        "protobuf": "7.35.1",
    }
    assert onnx.license == "Apache-2.0"


def test_manifest_entries_are_parsed(tmp_path: Path) -> None:
    assets = load_manifest(_manifest_file(tmp_path))

    assert len(assets) == 1
    assert isinstance(assets[0], ModelAsset)
    assert assets[0].name == "face_landmarker"


def test_unknown_asset_name_is_reported(tmp_path: Path) -> None:
    assets = load_manifest(_manifest_file(tmp_path))

    with pytest.raises(ModelStoreError, match="manifest"):
        find_asset(assets, "missing_model")


def test_plain_http_urls_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelStoreError, match="HTTPS"):
        load_manifest(_manifest_file(tmp_path, url="http://storage.googleapis.com/a.task"))


def test_urls_outside_the_allowed_hosts_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelStoreError, match="ホスト"):
        load_manifest(_manifest_file(tmp_path, url="https://example.com/a.task"))


@pytest.mark.parametrize("filename", ["../escape.task", "nested/model.task", ""])
def test_filenames_that_escape_the_models_directory_are_refused(
    tmp_path: Path,
    filename: str,
) -> None:
    with pytest.raises(ModelStoreError, match="ファイル名"):
        load_manifest(_manifest_file(tmp_path, filename=filename))


def test_malformed_manifest_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "model-manifest.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ModelStoreError, match="manifest"):
        load_manifest(path)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelStoreError, match="manifest"):
        load_manifest(tmp_path / "absent.json")


def test_install_writes_the_verified_asset(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    store = ModelStore(models_dir=tmp_path / "models", downloader=_downloader())

    installed = store.install(asset)

    assert installed.read_bytes() == PAYLOAD
    assert store.status(asset) is InstallStatus.READY


def test_missing_asset_is_reported_before_any_download(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    store = ModelStore(models_dir=tmp_path / "models", downloader=_downloader())

    assert store.status(asset) is InstallStatus.MISSING


def test_a_single_changed_byte_is_not_installed(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    tampered = bytearray(PAYLOAD)
    tampered[0] ^= 0x01
    store = ModelStore(models_dir=tmp_path / "models", downloader=_downloader(bytes(tampered)))

    with pytest.raises(ModelStoreError, match="sha256"):
        store.install(asset)

    assert store.status(asset) is InstallStatus.MISSING


def test_a_failed_install_keeps_the_previously_verified_file(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    models_dir = tmp_path / "models"
    ModelStore(models_dir=models_dir, downloader=_downloader()).install(asset)

    broken = ModelStore(models_dir=models_dir, downloader=_downloader(b"corrupted"))
    with pytest.raises(ModelStoreError):
        broken.install(asset)

    assert broken.path_for(asset).read_bytes() == PAYLOAD
    assert broken.status(asset) is InstallStatus.READY


def test_partial_downloads_do_not_remain_in_the_models_directory(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    models_dir = tmp_path / "models"

    def failing(url: str) -> Iterator[bytes]:
        yield PAYLOAD[:4]
        raise OSError("connection reset")

    store = ModelStore(models_dir=models_dir, downloader=failing)
    with pytest.raises(ModelStoreError, match="取得"):
        store.install(asset)

    assert list(models_dir.glob("*")) == []


def test_oversized_downloads_are_aborted(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    store = ModelStore(
        models_dir=tmp_path / "models",
        downloader=_downloader(PAYLOAD + b"extra bytes"),
    )

    with pytest.raises(ModelStoreError, match="サイズ"):
        store.install(asset)


def test_truncated_downloads_are_rejected(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    store = ModelStore(models_dir=tmp_path / "models", downloader=_downloader(PAYLOAD[:5]))

    with pytest.raises(ModelStoreError, match="サイズ"):
        store.install(asset)


def test_a_tampered_installed_file_is_detected_without_reinstalling(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    models_dir = tmp_path / "models"
    store = ModelStore(models_dir=models_dir, downloader=_downloader())
    store.install(asset)

    store.path_for(asset).write_bytes(b"replaced")

    assert store.status(asset) is InstallStatus.MISMATCH


def test_store_never_downloads_while_only_checking_status(tmp_path: Path) -> None:
    asset = load_manifest(_manifest_file(tmp_path))[0]
    calls: list[str] = []

    def counting(url: str) -> Iterator[bytes]:
        calls.append(url)
        yield PAYLOAD

    store = ModelStore(models_dir=tmp_path / "models", downloader=counting)
    store.status(asset)

    assert calls == []
