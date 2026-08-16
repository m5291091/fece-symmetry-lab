"""UI の各セクションが共有する、manifest 検証済みモデルの準備状態。

固定された landmark asset と年齢・性別 asset を解決し、その検証状態を報告し、
準備方法を利用者へ示す。取得は一切行わない。準備は
`scripts/bootstrap_models.py` の明示的な実行である（FR-002, §10.6）。
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from face_symmetry_lab.adapters.inference.onnx_age_gate import (
    OnnxAgeGateClassifier,
    load_age_gate_spec,
)
from face_symmetry_lab.adapters.io.model_store import (
    DEFAULT_MODELS_DIR,
    DerivedModelAsset,
    InstallStatus,
    ModelAsset,
    ModelStoreError,
    asset_status,
    find_asset,
    find_derived_asset,
    load_derived_assets,
    load_manifest,
)
from face_symmetry_lab.application.age_gate import AgeGateService

#: spec を読めない場合に `AgeGateService` へ渡す既定値。`classifier=None` の
#: ときは `evaluate()` がこれらへ触れる前に `UNDETERMINED` へ倒すため、
#: 実際の判定へは影響しない（AC-AGEGATE-03）。
_FALLBACK_AGE_GATE_THRESHOLD = 0.20
_FALLBACK_AGE_GATE_SPEC_VERSION = "age-gate/1.0.0"

LANDMARKER_PATH_ENV_VAR = "FACE_SYMMETRY_LAB_LANDMARKER_PATH"
DEMOGRAPHIC_ONNX_PATH_ENV_VAR = "FACE_SYMMETRY_LAB_DEMOGRAPHIC_ONNX_PATH"
SEGMENTER_PATH_ENV_VAR = "FACE_SYMMETRY_LAB_SEGMENTER_PATH"
LANDMARK_ASSET_NAME = "face_landmarker"
DEMOGRAPHIC_XML_ASSET_NAME = "demographic_age_gender_xml"
DEMOGRAPHIC_BIN_ASSET_NAME = "demographic_age_gender_bin"
DEMOGRAPHIC_ONNX_ASSET_NAME = "demographic_age_gender_onnx"
SEGMENTER_ASSET_NAME = "selfie_segmenter"
BOOTSTRAP_COMMAND = "uv run python scripts/bootstrap_models.py --accept-manifest"
CONVERT_DEMOGRAPHIC_COMMAND = "uv run --script scripts/convert_demographic_model.py"


def landmark_asset() -> ModelAsset:
    """manifest から、固定された Face Landmarker の項目を返す。"""
    return find_asset(load_manifest(), LANDMARK_ASSET_NAME)


def segmenter_asset() -> ModelAsset:
    """manifest から、固定された Image Segmenter の項目を返す。"""
    return find_asset(load_manifest(), SEGMENTER_ASSET_NAME)


def demographic_assets() -> tuple[ModelAsset, ModelAsset]:
    """対になるメタデータを検証したうえで、固定された XML と BIN の項目を返す。"""
    assets = load_manifest()
    xml_asset = find_asset(assets, DEMOGRAPHIC_XML_ASSET_NAME)
    bin_asset = find_asset(assets, DEMOGRAPHIC_BIN_ASSET_NAME)
    if xml_asset.version != bin_asset.version:
        raise ModelStoreError("年齢・性別モデルのXML/BINバージョンが一致しません。")
    if xml_asset.license != bin_asset.license:
        raise ModelStoreError("年齢・性別モデルのXML/BINライセンスが一致しません。")
    return xml_asset, bin_asset


def demographic_asset() -> DerivedModelAsset:
    """元 asset の由来を確認したうえで、派生 ONNX の項目を返す。"""
    xml_asset, bin_asset = demographic_assets()
    onnx_asset = find_derived_asset(load_derived_assets(), DEMOGRAPHIC_ONNX_ASSET_NAME)
    if onnx_asset.source_assets != (xml_asset.name, bin_asset.name):
        raise ModelStoreError("年齢・性別ONNXの元asset指定がmanifestと一致しません。")
    if onnx_asset.license != xml_asset.license or onnx_asset.license != bin_asset.license:
        raise ModelStoreError("年齢・性別ONNXと元assetのライセンスが一致しません。")
    return onnx_asset


def configured_landmarker_path() -> Path:
    """ローカル asset の path を解決する。環境変数が models ディレクトリに優先する。"""
    configured = os.environ.get(LANDMARKER_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MODELS_DIR / landmark_asset().filename


def configured_demographic_path() -> Path:
    """派生 ONNX の path を解決する。ローカル指定でも hash 一致は必須である。"""
    asset = demographic_asset()
    configured = os.environ.get(DEMOGRAPHIC_ONNX_PATH_ENV_VAR)
    return Path(configured).expanduser() if configured else DEFAULT_MODELS_DIR / asset.filename


def configured_segmenter_path() -> Path:
    """ローカル asset の path を解決する。環境変数が models ディレクトリに優先する。"""
    configured = os.environ.get(SEGMENTER_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MODELS_DIR / segmenter_asset().filename


def landmarker_status(path: Path) -> InstallStatus:
    """``path`` にあるファイルを、固定された sha256 と照合する。"""
    return asset_status(path, landmark_asset())


def demographic_model_status(onnx_path: Path) -> InstallStatus:
    """推論の前に、派生 ONNX が固定 manifest と一致することを要求する。"""
    return asset_status(onnx_path, demographic_asset())


def segmenter_status(path: Path) -> InstallStatus:
    """``path`` にあるファイルを、固定された sha256 と照合する。"""
    return asset_status(path, segmenter_asset())


def _render_missing(path: Path, asset: ModelAsset) -> None:
    st.warning("ランドマークモデルが未配置です。アプリは自動取得しません。")
    st.caption(
        f"取得元 {asset.url} / {asset.size_bytes:,} bytes / "
        f"ライセンス {asset.license} / sha256 {asset.sha256}"
    )
    st.code(BOOTSTRAP_COMMAND, language="bash")
    st.caption(
        f"別の場所に置く場合は環境変数 `{LANDMARKER_PATH_ENV_VAR}` で `{path}` を指定します。"
    )


def resolve_ready_landmarker() -> Path | None:
    """モデルの状態を描画し、検証できた場合にのみ path を返す。"""
    try:
        asset = landmark_asset()
    except ModelStoreError as error:
        st.error(str(error))
        return None

    path = configured_landmarker_path()
    status = landmarker_status(path)

    if status is InstallStatus.MISSING:
        _render_missing(path, asset)
        return None
    if status is InstallStatus.MISMATCH:
        st.error(
            "配置済みモデルの sha256 が manifest と一致しません。"
            "改変または破損の可能性があるため読み込みません。"
        )
        st.code(BOOTSTRAP_COMMAND, language="bash")
        return None

    st.caption(f"model: {asset.version} · sha256 {asset.sha256[:12]}… · 検証済み")
    return path


def resolve_ready_demographic() -> tuple[Path, str] | None:
    """状態を描画し、検証済みの派生 ONNX の path だけを返す。"""
    try:
        xml_asset, bin_asset = demographic_assets()
        onnx_asset = demographic_asset()
        onnx_path = configured_demographic_path()
        status = demographic_model_status(onnx_path)
    except ModelStoreError as error:
        st.error(str(error))
        return None

    if status is InstallStatus.MISSING:
        st.warning("年齢・性別モデルの派生ONNXが未配置です。アプリは自動生成しません。")
        for asset in (xml_asset, bin_asset):
            st.caption(
                f"取得元 {asset.url} / {asset.size_bytes:,} bytes / "
                f"ライセンス {asset.license} / sha256 {asset.sha256}"
            )
        st.code(BOOTSTRAP_COMMAND, language="bash")
        st.code(CONVERT_DEMOGRAPHIC_COMMAND, language="bash")
        return None
    if status is InstallStatus.MISMATCH:
        st.error(
            "年齢・性別モデルのONNXがmanifestと一致しません。"
            "改変または破損の可能性があるため読み込みません。"
        )
        st.code(CONVERT_DEMOGRAPHIC_COMMAND, language="bash")
        return None

    st.caption(
        f"model: {onnx_asset.version} · ONNX {onnx_asset.sha256[:12]}… · "
        f"source XML/BIN {xml_asset.sha256[:8]}…/{bin_asset.sha256[:8]}… · 検証済み"
    )
    return onnx_path, onnx_asset.version


def build_age_gate_service() -> AgeGateService:
    """FR-038: spec・ONNXのいずれが欠けても例外を投げず、`classifier=None`のサービスへ倒す。

    `age-gate/1.0.0` は取得元URLを持たないbundled資産（§22.5）で、
    `resolve_ready_*` のような取得操作の案内は該当しない。欠落時は
    `AgeGateService` が常に `UNDETERMINED`（安全側の既定）を返すため、
    ここでは利用者向けの警告を出さず静かにフォールバックする
    （AC-AGEGATE-03、ゲートの成否は左右差分析を止めない）。
    """
    try:
        spec = load_age_gate_spec()
    except ValueError:
        return AgeGateService(
            None,
            threshold=_FALLBACK_AGE_GATE_THRESHOLD,
            spec_version=_FALLBACK_AGE_GATE_SPEC_VERSION,
        )
    try:
        classifier = OnnxAgeGateClassifier(spec=spec)
    except ValueError:
        classifier = None
    return AgeGateService(
        classifier,
        threshold=spec.threshold,
        spec_version=spec.gate_version,
        # §9.2: 記録へ残す来歴。classifier の生成に失敗しても spec は読めている
        # ので、どの版で判定を試みたかは残せる。
        spec_revision=spec.spec_revision,
        model_sha256=spec.onnx_sha256,
    )
