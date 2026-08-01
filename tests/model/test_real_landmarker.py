"""Real-model smoke tests (§21.1 "Model smoke").

These run against the actual `face_landmarker.task` asset installed by
`scripts/bootstrap_models.py`. They skip when the asset is absent so that a
clean checkout still passes, and they never download anything themselves.

Set `FACE_SYMMETRY_LAB_SMOKE_IMAGE` to a locally held, consented face photo to
also exercise a real detection. Such images are face-derived data and must not
be committed to this repository (§21.4).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    MediaPipeFaceLandmarker,
    MediaPipeImageLandmarker,
)
from face_symmetry_lab.adapters.io.model_store import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODELS_DIR,
    InstallStatus,
    ModelStore,
    find_asset,
    load_manifest,
)
from face_symmetry_lab.application.still_image import detect_still_image_landmarks
from face_symmetry_lab.domain.landmarks import FACE_LANDMARK_COUNT, TrackingStatus
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import Status

pytestmark = pytest.mark.model

SMOKE_IMAGE_ENV_VAR = "FACE_SYMMETRY_LAB_SMOKE_IMAGE"


def _asset():
    return find_asset(load_manifest(DEFAULT_MANIFEST_PATH), "face_landmarker")


def _installed_model() -> Path:
    asset = _asset()
    store = ModelStore(models_dir=DEFAULT_MODELS_DIR)
    status = store.status(asset)
    if status is InstallStatus.MISSING:
        pytest.skip("モデル未配置: uv run python scripts/bootstrap_models.py --accept-manifest")
    assert status is InstallStatus.READY, "インストール済みモデルの sha256 が manifest と不一致"
    return store.path_for(asset)


def _blank(height: int = 480, width: int = 640) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


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


def test_video_landmarker_loads_and_runs_on_a_blank_frame() -> None:
    landmarker = MediaPipeFaceLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        first = landmarker.detect_video(_blank(), frame_id=1, timestamp_ms=0)
        second = landmarker.detect_video(_blank(), frame_id=2, timestamp_ms=33)
    finally:
        landmarker.close()

    assert first.status is TrackingStatus.NO_FACE
    assert second.status is TrackingStatus.NO_FACE


def test_image_landmarker_loads_and_runs_on_a_blank_image() -> None:
    landmarker = MediaPipeImageLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        frame = landmarker.detect(_blank())
    finally:
        landmarker.close()

    assert frame.status is TrackingStatus.NO_FACE
    assert frame.points == ()


def test_synthetic_face_is_detected_with_the_full_landmark_set(
    synthetic_face_rgb: NDArray[np.uint8],
) -> None:
    landmarker = MediaPipeImageLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        result = detect_still_image_landmarks(landmarker, synthetic_face_rgb)
    finally:
        landmarker.close()

    assert result.frame.status is TrackingStatus.SINGLE_FACE
    assert result.landmark_count == FACE_LANDMARK_COUNT
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in result.frame.points)
    assert not np.array_equal(result.preview, synthetic_face_rgb)


def test_synthetic_two_face_image_is_refused_without_picking_one(
    synthetic_two_face_rgb: NDArray[np.uint8],
) -> None:
    landmarker = MediaPipeImageLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        result = detect_still_image_landmarks(landmarker, synthetic_two_face_rgb)
    finally:
        landmarker.close()

    assert result.frame.face_count >= 2
    assert result.frame.status is TrackingStatus.MULTI_FACE
    assert result.landmark_count == 0
    assert np.array_equal(result.preview, synthetic_two_face_rgb)


def test_video_mode_detects_the_synthetic_face_across_frames(
    synthetic_face_rgb: NDArray[np.uint8],
) -> None:
    landmarker = MediaPipeFaceLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        frames = [
            landmarker.detect_video(synthetic_face_rgb, frame_id=index, timestamp_ms=index * 33)
            for index in range(1, 4)
        ]
    finally:
        landmarker.close()

    assert [frame.status for frame in frames] == [TrackingStatus.SINGLE_FACE] * 3
    assert all(len(frame.points) == FACE_LANDMARK_COUNT for frame in frames)


def test_real_photo_produces_a_full_landmark_set() -> None:
    image = _load_smoke_image()
    landmarker = MediaPipeImageLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        result = detect_still_image_landmarks(landmarker, image)
    finally:
        landmarker.close()

    assert result.frame.status is TrackingStatus.SINGLE_FACE
    assert result.landmark_count == FACE_LANDMARK_COUNT
    assert result.preview.shape == image.shape
    assert not np.array_equal(result.preview, image)


def test_the_synthetic_face_produces_a_symmetry_score(
    synthetic_face_rgb: NDArray[np.uint8],
) -> None:
    """End-to-end: real model -> pose -> quality gate -> score."""
    from face_symmetry_lab.adapters.io.metric_config import load_metric_config
    from face_symmetry_lab.application.still_image import analyze_still_image

    landmarker = MediaPipeImageLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        analysis = analyze_still_image(landmarker, synthetic_face_rgb, config=load_metric_config())
    finally:
        landmarker.close()

    outcome = analysis.outcome
    assert outcome.state.status is Status.OK, outcome.state.reason_codes
    assert outcome.symmetry is not None
    assert outcome.overall_raw is not None
    assert 0.0 <= outcome.overall_raw <= 100.0
    for name in PART_NAMES:
        value = outcome.symmetry.part(name)
        assert value is not None and 0.0 <= value <= 100.0


def test_a_drawn_symmetric_face_scores_higher_than_a_skewed_one(
    synthetic_face_rgb: NDArray[np.uint8],
) -> None:
    """A deliberately asymmetric render must not score above the symmetric one."""
    from PIL import Image

    from face_symmetry_lab.adapters.io.metric_config import load_metric_config
    from face_symmetry_lab.application.still_image import analyze_still_image

    skewed = Image.fromarray(synthetic_face_rgb).transform(
        (512, 512),
        Image.Transform.AFFINE,
        (1.0, 0.10, -25.0, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BILINEAR,
    )
    config = load_metric_config()
    landmarker = MediaPipeImageLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        balanced = analyze_still_image(landmarker, synthetic_face_rgb, config=config)
        distorted = analyze_still_image(
            landmarker,
            np.asarray(skewed, dtype=np.uint8),
            config=config,
        )
    finally:
        landmarker.close()

    assert balanced.outcome.overall_raw is not None
    if distorted.outcome.overall_raw is not None:
        assert distorted.outcome.overall_raw <= balanced.outcome.overall_raw


def test_the_head_pose_of_the_frontal_synthetic_face_is_near_zero(
    synthetic_face_rgb: NDArray[np.uint8],
) -> None:
    landmarker = MediaPipeImageLandmarker(
        model_path=_installed_model(),
        model_version=_asset().version,
    )
    try:
        frame = landmarker.detect(synthetic_face_rgb)
    finally:
        landmarker.close()

    assert frame.pose is not None
    assert abs(frame.pose.yaw_deg) < 15.0
    assert abs(frame.pose.roll_deg) < 15.0
