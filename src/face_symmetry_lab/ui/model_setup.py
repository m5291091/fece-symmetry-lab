"""Manifest-verified model readiness shared by the UI sections.

Resolves pinned landmark and demographic assets, reports their verification
status, and tells the user how to prepare them. It never downloads anything:
preparation is an explicit `scripts/bootstrap_models.py` run (FR-002, §10.6).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

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
    """Return the pinned Face Landmarker entry from the manifest."""
    return find_asset(load_manifest(), LANDMARK_ASSET_NAME)


def segmenter_asset() -> ModelAsset:
    """Return the pinned Image Segmenter entry from the manifest."""
    return find_asset(load_manifest(), SEGMENTER_ASSET_NAME)


def demographic_assets() -> tuple[ModelAsset, ModelAsset]:
    """Return the pinned XML and BIN entries after validating pair metadata."""
    assets = load_manifest()
    xml_asset = find_asset(assets, DEMOGRAPHIC_XML_ASSET_NAME)
    bin_asset = find_asset(assets, DEMOGRAPHIC_BIN_ASSET_NAME)
    if xml_asset.version != bin_asset.version:
        raise ModelStoreError("年齢・性別モデルのXML/BINバージョンが一致しません。")
    if xml_asset.license != bin_asset.license:
        raise ModelStoreError("年齢・性別モデルのXML/BINライセンスが一致しません。")
    return xml_asset, bin_asset


def demographic_asset() -> DerivedModelAsset:
    """Return the derived ONNX entry after checking its source provenance."""
    xml_asset, bin_asset = demographic_assets()
    onnx_asset = find_derived_asset(load_derived_assets(), DEMOGRAPHIC_ONNX_ASSET_NAME)
    if onnx_asset.source_assets != (xml_asset.name, bin_asset.name):
        raise ModelStoreError("年齢・性別ONNXの元asset指定がmanifestと一致しません。")
    if onnx_asset.license != xml_asset.license or onnx_asset.license != bin_asset.license:
        raise ModelStoreError("年齢・性別ONNXと元assetのライセンスが一致しません。")
    return onnx_asset


def configured_landmarker_path() -> Path:
    """Resolve the local asset path. The env var wins over the models directory."""
    configured = os.environ.get(LANDMARKER_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MODELS_DIR / landmark_asset().filename


def configured_demographic_path() -> Path:
    """Resolve the derived ONNX path; local overrides still require its hash."""
    asset = demographic_asset()
    configured = os.environ.get(DEMOGRAPHIC_ONNX_PATH_ENV_VAR)
    return Path(configured).expanduser() if configured else DEFAULT_MODELS_DIR / asset.filename


def configured_segmenter_path() -> Path:
    """Resolve the local asset path. The env var wins over the models directory."""
    configured = os.environ.get(SEGMENTER_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MODELS_DIR / segmenter_asset().filename


def landmarker_status(path: Path) -> InstallStatus:
    """Compare the file at ``path`` against the pinned sha256."""
    return asset_status(path, landmark_asset())


def demographic_model_status(onnx_path: Path) -> InstallStatus:
    """Require the derived ONNX to match the pinned manifest before inference."""
    return asset_status(onnx_path, demographic_asset())


def segmenter_status(path: Path) -> InstallStatus:
    """Compare the file at ``path`` against the pinned sha256."""
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
    """Render the model status and return the path only when it is verified."""
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


def resolve_ready_segmenter() -> Path | None:
    """Render the model status and return the path only when it is verified.

    Unlike the landmark/demographic models, this asset is optional: the
    background feature stays OFF and unusable until it is verified, but
    nothing else in the app depends on it (§14.3b).
    """
    try:
        asset = segmenter_asset()
    except ModelStoreError as error:
        st.error(str(error))
        return None

    path = configured_segmenter_path()
    status = segmenter_status(path)

    if status is InstallStatus.MISSING:
        st.warning("背景合成モデルが未配置です。アプリは自動取得しません。")
        st.caption(
            f"取得元 {asset.url} / {asset.size_bytes:,} bytes / "
            f"ライセンス {asset.license} / sha256 {asset.sha256}"
        )
        st.code(BOOTSTRAP_COMMAND, language="bash")
        st.caption(
            f"別の場所に置く場合は環境変数 `{SEGMENTER_PATH_ENV_VAR}` で `{path}` を指定します。"
        )
        return None
    if status is InstallStatus.MISMATCH:
        st.error(
            "配置済みの背景合成モデルの sha256 が manifest と一致しません。"
            "改変または破損の可能性があるため読み込みません。"
        )
        st.code(BOOTSTRAP_COMMAND, language="bash")
        return None

    st.caption(f"model: {asset.version} · sha256 {asset.sha256[:12]}… · 検証済み")
    return path


def resolve_ready_demographic() -> tuple[Path, str] | None:
    """Render status and return only a verified derived ONNX path."""
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
