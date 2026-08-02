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

#: The shipped ONNX (cv2.dnn) must track a second independent ONNX runtime
#: (onnxruntime) far more tightly than either tracks OpenVINO. This bounds
#: "far more tightly" generously (an order of magnitude) so the test is a
#: robust guard against a real conversion regression, not a tight numeric
#: gate like OD-07's OpenVINO parity target.
MAX_CV_ORT_AGE_DIFF_YEARS = 0.01
MAX_CV_ORT_GENDER_DIFF = 1e-4


@pytest.mark.model
def test_shipped_onnx_agrees_with_a_second_independent_onnx_runtime() -> None:
    """OD-07 due diligence: rule out a conversion bug as the parity source.

    If ``cv2.dnn`` (the app's runtime) and ``onnxruntime`` (an independent
    ONNX engine) executing the exact same graph disagree by roughly as much
    as either disagrees with OpenVINO, that would point at a bug in
    ``scripts/convert_demographic_model.py`` rather than an OpenVINO-CPU
    execution trait. See ``scripts/verify_onnx_runtime_agreement.py``.
    """
    if (
        importlib.util.find_spec("openvino") is None
        or importlib.util.find_spec("onnxruntime") is None
    ):
        pytest.skip(
            "OpenVINOまたはonnxruntimeが現在の環境にありません。"
            "別の開発環境で `uv run scripts/verify_onnx_runtime_agreement.py` を実行してください。"
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
        import verify_onnx_runtime_agreement as agreement
    finally:
        sys.path.remove(str(SCRIPTS_DIR))

    results = agreement.evaluate(
        xml_path=xml_path,
        bin_path=bin_path,
        onnx_path=onnx_path,
        manifest_path=APP_ROOT / "configs" / "model-manifest.json",
    )
    summary = agreement.summarize(results)

    assert summary["max_cv_ort_age_diff_years"] <= MAX_CV_ORT_AGE_DIFF_YEARS, (
        f"cv2.dnn と onnxruntime の年齢差が大きすぎます: "
        f"{summary['max_cv_ort_age_diff_years']:.6f} > {MAX_CV_ORT_AGE_DIFF_YEARS}"
    )
    assert summary["max_cv_ort_gender_diff"] <= MAX_CV_ORT_GENDER_DIFF, (
        f"cv2.dnn と onnxruntime の性別確率差が大きすぎます: "
        f"{summary['max_cv_ort_gender_diff']:.6e} > {MAX_CV_ORT_GENDER_DIFF}"
    )
