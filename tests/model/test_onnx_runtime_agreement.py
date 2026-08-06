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

#: 同梱 ONNX（cv2.dnn）は、独立した 2 つ目の ONNX runtime（onnxruntime）と、
#: どちらが OpenVINO と一致する度合いよりもはるかに強く一致していなければ
#: ならない。ここでは「はるかに強く」を桁 1 つ分と大きく見積もる。これにより
#: このテストは、OD-07 の OpenVINO parity 目標のような厳密な数値ゲートではなく、
#: 実際の変換回帰に対する頑健なガードになる。
MAX_CV_ORT_AGE_DIFF_YEARS = 0.01
MAX_CV_ORT_GENDER_DIFF = 1e-4


@pytest.mark.model
def test_shipped_onnx_agrees_with_a_second_independent_onnx_runtime() -> None:
    """OD-07 の due diligence: parity の原因が変換の bug でないことを確かめる。

    まったく同じグラフを実行する ``cv2.dnn``（アプリの runtime）と
    ``onnxruntime``（独立した ONNX エンジン）が、どちらかが OpenVINO と食い違う
    のとおおむね同じだけ互いに食い違うなら、それは OpenVINO CPU の実行特性では
    なく ``scripts/convert_demographic_model.py`` の bug を示す。
    ``scripts/verify_onnx_runtime_agreement.py`` を参照。
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
