from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from face_symmetry_lab.adapters.io.model_store import (
    DEFAULT_MODELS_DIR,
    InstallStatus,
    asset_status,
)
from face_symmetry_lab.ui.model_setup import demographic_asset, demographic_assets

APP_ROOT = Path(__file__).parents[2]
CONVERTER = APP_ROOT / "scripts" / "convert_demographic_model.py"


@pytest.mark.model
def test_conversion_reproduces_the_manifest_pinned_onnx(tmp_path: Path) -> None:
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("開発時変換packageのonnxが現在の環境にありません。")
    xml_asset, bin_asset = demographic_assets()
    xml_path = DEFAULT_MODELS_DIR / xml_asset.filename
    bin_path = DEFAULT_MODELS_DIR / bin_asset.filename
    if (
        asset_status(xml_path, xml_asset) is not InstallStatus.READY
        or asset_status(bin_path, bin_asset) is not InstallStatus.READY
    ):
        pytest.skip("manifest検証済みの公式OpenVINO IRが未配置です。")

    output_path = tmp_path / demographic_asset().filename
    completed = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--xml",
            str(xml_path),
            "--bin",
            str(bin_path),
            "--output",
            str(output_path),
        ],
        cwd=APP_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert asset_status(output_path, demographic_asset()) is InstallStatus.READY
