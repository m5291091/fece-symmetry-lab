from __future__ import annotations

import importlib.util
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
SCRIPTS_DIR = APP_ROOT / "scripts"


@pytest.mark.model
def test_openvino_reference_and_shipped_onnx_stay_within_the_od07_gate() -> None:
    """OD-07: fixed-input parity between the OpenVINO IR and the derived ONNX.

    ``openvino`` never joins the app's dependency group (ADR-0003), so this
    test only runs when a development environment explicitly installs it
    alongside the app (see ``scripts/verify_openvino_parity.py``); everywhere
    else it skips. This mirrors ``test_demographic_conversion.py``.
    """
    if importlib.util.find_spec("openvino") is None:
        pytest.skip(
            "OpenVINO参照runtimeが現在の環境にありません。"
            "別の開発環境で `uv run scripts/verify_openvino_parity.py` を実行してください。"
        )

    xml_asset, bin_asset = demographic_assets()
    onnx_asset = demographic_asset()
    xml_path = DEFAULT_MODELS_DIR / xml_asset.filename
    bin_path = DEFAULT_MODELS_DIR / bin_asset.filename
    onnx_path = DEFAULT_MODELS_DIR / onnx_asset.filename
    if (
        asset_status(xml_path, xml_asset) is not InstallStatus.READY
        or asset_status(bin_path, bin_asset) is not InstallStatus.READY
        or asset_status(onnx_path, onnx_asset) is not InstallStatus.READY
    ):
        pytest.skip("manifest検証済みの公式OpenVINO IRまたは派生ONNXが未配置です。")

    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import verify_openvino_parity as parity
    finally:
        sys.path.remove(str(SCRIPTS_DIR))

    results = parity.evaluate(
        xml_path=xml_path,
        bin_path=bin_path,
        onnx_path=onnx_path,
        manifest_path=APP_ROOT / "configs" / "model-manifest.json",
    )
    summary = parity.summarize(
        results,
        age_gate_years=parity.DEFAULT_AGE_GATE_YEARS,
        gender_gate_prob=parity.DEFAULT_GENDER_GATE_PROB,
    )

    assert summary["age_gate_passed"], (
        f"年齢差が gate を超えています: {summary['max_age_diff_years']:.6f} "
        f"> {parity.DEFAULT_AGE_GATE_YEARS}"
    )
    assert summary["gender_gate_passed"], (
        f"性別確率差が gate を超えています: {summary['max_gender_prob_diff']:.6e} "
        f"> {parity.DEFAULT_GENDER_GATE_PROB}"
    )
