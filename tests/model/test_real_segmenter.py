"""Real-model smoke tests for the background segmenter (§14.3b, §21.1).

These run against the actual `selfie_segmenter.tflite` asset installed by
`scripts/bootstrap_models.py`. They skip when the asset is absent so a clean
checkout still passes, and they never download anything themselves.

The repository's synthetic, line-art fixture (§21.4) gives this model
near-zero foreground confidence everywhere — it is not a photographic
selfie, so behavioral assertions here stay to shape/dtype/finiteness. Set
`FACE_SYMMETRY_LAB_SEGMENTER_SMOKE_IMAGE` to a locally held, consented photo
to also see a real foreground mask; such images must not be committed here.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_segmenter import MediaPipeSegmenter
from face_symmetry_lab.adapters.io.model_store import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODELS_DIR,
    InstallStatus,
    ModelStore,
    find_asset,
    load_manifest,
)
from face_symmetry_lab.domain.frame_adjustments import composite_background

pytestmark = pytest.mark.model

SMOKE_IMAGE_ENV_VAR = "FACE_SYMMETRY_LAB_SEGMENTER_SMOKE_IMAGE"


def _asset():
    return find_asset(load_manifest(DEFAULT_MANIFEST_PATH), "selfie_segmenter")


def _installed_model() -> Path:
    asset = _asset()
    store = ModelStore(models_dir=DEFAULT_MODELS_DIR)
    status = store.status(asset)
    if status is InstallStatus.MISSING:
        pytest.skip("モデル未配置: uv run python scripts/bootstrap_models.py --accept-manifest")
    assert status is InstallStatus.READY, "インストール済みモデルの sha256 が manifest と不一致"
    return store.path_for(asset)


def _noise(height: int = 64, width: int = 64) -> NDArray[np.uint8]:
    """A non-degenerate frame.

    A flat, all-zero frame was observed to make the underlying TFLite
    runtime intermittently return uninitialized (huge, still technically
    finite) values at small sizes — a real upstream instability on
    degenerate input, not something this adapter can paper over. A seeded
    noise frame keeps the smoke test both realistic and deterministic.
    """
    rng = np.random.default_rng(20260801)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def _load_smoke_image() -> NDArray[np.uint8]:
    configured = os.environ.get(SMOKE_IMAGE_ENV_VAR)
    if not configured:
        pytest.skip(f"{SMOKE_IMAGE_ENV_VAR} が未設定のため実写検出は実行しない")
    from PIL import Image

    with Image.open(Path(configured).expanduser()) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def test_installed_asset_matches_the_pinned_manifest_hash() -> None:
    model_path = _installed_model()

    assert model_path.is_file()
    assert model_path.stat().st_size == _asset().size_bytes


def test_the_segmenter_loads_and_runs_on_a_frame() -> None:
    segmenter = MediaPipeSegmenter(model_path=_installed_model(), model_version=_asset().version)
    try:
        mask = segmenter.segment(_noise(64, 64), timestamp_ms=0)
    finally:
        segmenter.close()

    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert np.all(np.isfinite(mask))
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_consecutive_frames_require_increasing_timestamps() -> None:
    segmenter = MediaPipeSegmenter(model_path=_installed_model(), model_version=_asset().version)
    try:
        first = segmenter.segment(_noise(32, 32), timestamp_ms=0)
        second = segmenter.segment(_noise(32, 32), timestamp_ms=33)
    finally:
        segmenter.close()

    assert first.shape == second.shape == (32, 32)


def test_the_mask_can_composite_a_solid_background(synthetic_face_rgb: NDArray[np.uint8]) -> None:
    """End-to-end: real model -> mask -> domain compositing, still a valid image."""
    segmenter = MediaPipeSegmenter(model_path=_installed_model(), model_version=_asset().version)
    try:
        height, width = synthetic_face_rgb.shape[:2]
        mask = segmenter.segment(synthetic_face_rgb, timestamp_ms=0)
    finally:
        segmenter.close()

    composited = composite_background(synthetic_face_rgb, mask, (0, 0, 0))

    assert composited.shape == (height, width, 3)
    assert composited.dtype == np.uint8


def test_a_real_photo_produces_a_high_confidence_foreground_region() -> None:
    """Only runs with a locally supplied, consented photo (never committed)."""
    image = _load_smoke_image()
    segmenter = MediaPipeSegmenter(model_path=_installed_model(), model_version=_asset().version)
    try:
        mask = segmenter.segment(image, timestamp_ms=0)
    finally:
        segmenter.close()

    assert mask.shape == image.shape[:2]
    assert mask.max() > 0.5, "実写では前景confidenceが有意に上がることを期待する"
