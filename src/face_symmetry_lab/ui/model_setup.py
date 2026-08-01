"""Model readiness shared by the camera and still-image sections.

Resolves the pinned landmark asset, reports its verification status, and tells
the user how to prepare it. It never downloads anything: preparation is an
explicit `scripts/bootstrap_models.py` run (FR-002, §10.6).
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
    file_sha256,
    find_asset,
    load_manifest,
)

LANDMARKER_PATH_ENV_VAR = "FACE_SYMMETRY_LAB_LANDMARKER_PATH"
LANDMARK_ASSET_NAME = "face_landmarker"
BOOTSTRAP_COMMAND = "uv run python scripts/bootstrap_models.py --accept-manifest"


def landmark_asset() -> ModelAsset:
    """Return the pinned Face Landmarker entry from the manifest."""
    return find_asset(load_manifest(), LANDMARK_ASSET_NAME)


def configured_landmarker_path() -> Path:
    """Resolve the local asset path. The env var wins over the models directory."""
    configured = os.environ.get(LANDMARKER_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MODELS_DIR / landmark_asset().filename


def landmarker_status(path: Path) -> InstallStatus:
    """Compare the file at ``path`` against the pinned sha256."""
    if not path.is_file():
        return InstallStatus.MISSING
    asset = landmark_asset()
    if path.stat().st_size != asset.size_bytes:
        return InstallStatus.MISMATCH
    if file_sha256(path) != asset.sha256:
        return InstallStatus.MISMATCH
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
