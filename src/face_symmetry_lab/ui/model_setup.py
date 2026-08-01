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
    InstallStatus,
    ModelAsset,
    ModelStoreError,
    asset_status,
    find_asset,
    load_manifest,
)

LANDMARKER_PATH_ENV_VAR = "FACE_SYMMETRY_LAB_LANDMARKER_PATH"
DEMOGRAPHIC_XML_PATH_ENV_VAR = "FACE_SYMMETRY_LAB_DEMOGRAPHIC_XML_PATH"
DEMOGRAPHIC_BIN_PATH_ENV_VAR = "FACE_SYMMETRY_LAB_DEMOGRAPHIC_BIN_PATH"
LANDMARK_ASSET_NAME = "face_landmarker"
DEMOGRAPHIC_XML_ASSET_NAME = "demographic_age_gender_xml"
DEMOGRAPHIC_BIN_ASSET_NAME = "demographic_age_gender_bin"
BOOTSTRAP_COMMAND = "uv run python scripts/bootstrap_models.py --accept-manifest"


def landmark_asset() -> ModelAsset:
    """Return the pinned Face Landmarker entry from the manifest."""
    return find_asset(load_manifest(), LANDMARK_ASSET_NAME)


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


def configured_landmarker_path() -> Path:
    """Resolve the local asset path. The env var wins over the models directory."""
    configured = os.environ.get(LANDMARKER_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MODELS_DIR / landmark_asset().filename


def configured_demographic_paths() -> tuple[Path, Path]:
    """Resolve demographic IR paths; explicit local overrides still require hashes."""
    xml_asset, bin_asset = demographic_assets()
    configured_xml = os.environ.get(DEMOGRAPHIC_XML_PATH_ENV_VAR)
    configured_bin = os.environ.get(DEMOGRAPHIC_BIN_PATH_ENV_VAR)
    xml_path = (
        Path(configured_xml).expanduser()
        if configured_xml
        else DEFAULT_MODELS_DIR / xml_asset.filename
    )
    bin_path = (
        Path(configured_bin).expanduser()
        if configured_bin
        else DEFAULT_MODELS_DIR / bin_asset.filename
    )
    return xml_path, bin_path


def landmarker_status(path: Path) -> InstallStatus:
    """Compare the file at ``path`` against the pinned sha256."""
    return asset_status(path, landmark_asset())


def demographic_model_status(xml_path: Path, bin_path: Path) -> InstallStatus:
    """Require both IR files to match the pinned manifest before inference."""
    xml_asset, bin_asset = demographic_assets()
    statuses = (asset_status(xml_path, xml_asset), asset_status(bin_path, bin_asset))
    if InstallStatus.MISMATCH in statuses:
        return InstallStatus.MISMATCH
    if InstallStatus.MISSING in statuses:
        return InstallStatus.MISSING
    return InstallStatus.READY


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


def resolve_ready_demographic() -> tuple[Path, Path, str] | None:
    """Render the demographic pair status and return only verified IR paths."""
    try:
        xml_asset, bin_asset = demographic_assets()
        xml_path, bin_path = configured_demographic_paths()
        status = demographic_model_status(xml_path, bin_path)
    except ModelStoreError as error:
        st.error(str(error))
        return None

    if status is InstallStatus.MISSING:
        st.warning("年齢・性別モデルが未配置です。アプリは自動取得しません。")
        for asset in (xml_asset, bin_asset):
            st.caption(
                f"取得元 {asset.url} / {asset.size_bytes:,} bytes / "
                f"ライセンス {asset.license} / sha256 {asset.sha256}"
            )
        st.code(BOOTSTRAP_COMMAND, language="bash")
        return None
    if status is InstallStatus.MISMATCH:
        st.error(
            "年齢・性別モデルのXMLまたはBINがmanifestと一致しません。"
            "改変または破損の可能性があるため読み込みません。"
        )
        st.code(BOOTSTRAP_COMMAND, language="bash")
        return None

    st.caption(
        f"model: {xml_asset.version} · XML {xml_asset.sha256[:12]}… · "
        f"BIN {bin_asset.sha256[:12]}… · 検証済み"
    )
    return xml_path, bin_path, xml_asset.version
