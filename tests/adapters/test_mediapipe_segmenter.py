from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_segmenter import (
    MediaPipeSegmenter,
    SegmentationError,
    SegmentationFrameOrderError,
)


class _FakeMask:
    def __init__(self, array: NDArray[np.float32]) -> None:
        self._array = array

    def numpy_view(self) -> NDArray[np.float32]:
        return self._array


class FakeDetector:
    def __init__(self, masks: list[NDArray[np.float32]] | None = None) -> None:
        self.masks = masks
        self.timestamps: list[int] = []
        self.closed = False
        self.error: Exception | None = None

    def segment_for_video(self, image: Any, timestamp_ms: int) -> SimpleNamespace:
        self.timestamps.append(timestamp_ms)
        if self.error is not None:
            raise self.error
        confidence_masks = [_FakeMask(mask) for mask in (self.masks or [])]
        return SimpleNamespace(confidence_masks=confidence_masks)

    def close(self) -> None:
        self.closed = True


def _model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "selfie_segmenter.tflite"
    model_path.write_bytes(b"test-segmenter")
    return model_path


def _segmenter(tmp_path: Path, detector: FakeDetector) -> MediaPipeSegmenter:
    return MediaPipeSegmenter(
        model_path=_model_file(tmp_path),
        model_version="selfie-segmenter/1.0.0",
        detector_factory=lambda _: detector,
        image_factory=lambda rgb: rgb,
    )


def _rgb(height: int = 32, width: int = 32) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _mask(height: int = 32, width: int = 32, value: float = 0.7) -> NDArray[np.float32]:
    return np.full((height, width, 1), value, dtype=np.float32)


def test_a_single_mask_is_returned_squeezed_to_two_dimensions(tmp_path: Path) -> None:
    detector = FakeDetector([_mask(32, 32, 0.7)])
    segmenter = _segmenter(tmp_path, detector)

    mask = segmenter.segment(_rgb(32, 32), timestamp_ms=0)

    assert mask.shape == (32, 32)
    assert np.all(mask == pytest.approx(0.7))
    # A construction-time warm-up call (timestamp -1) precedes real frames.
    assert detector.timestamps == [-1, 0]


def test_timestamps_must_increase_monotonically(tmp_path: Path) -> None:
    detector = FakeDetector([_mask()])
    segmenter = _segmenter(tmp_path, detector)
    segmenter.segment(_rgb(), timestamp_ms=100)

    with pytest.raises(SegmentationFrameOrderError):
        segmenter.segment(_rgb(), timestamp_ms=100)


def test_out_of_order_frames_do_not_reach_the_detector(tmp_path: Path) -> None:
    detector = FakeDetector([_mask()])
    segmenter = _segmenter(tmp_path, detector)
    segmenter.segment(_rgb(), timestamp_ms=100)

    with pytest.raises(SegmentationFrameOrderError):
        segmenter.segment(_rgb(), timestamp_ms=40)

    assert detector.timestamps == [-1, 100]


def test_a_missing_mask_result_is_a_recoverable_error(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, FakeDetector([]))

    with pytest.raises(SegmentationError):
        segmenter.segment(_rgb(), timestamp_ms=0)


def test_a_mismatched_mask_shape_is_a_recoverable_error(tmp_path: Path) -> None:
    detector = FakeDetector([_mask(height=8, width=8)])
    segmenter = _segmenter(tmp_path, detector)

    with pytest.raises(SegmentationError):
        segmenter.segment(_rgb(32, 32), timestamp_ms=0)


def test_a_non_finite_mask_is_a_recoverable_error(tmp_path: Path) -> None:
    mask = _mask(8, 8)
    mask[0, 0, 0] = float("nan")
    detector = FakeDetector([mask])
    segmenter = _segmenter(tmp_path, detector)

    with pytest.raises(SegmentationError):
        segmenter.segment(_rgb(8, 8), timestamp_ms=0)


def test_detector_failures_become_recoverable_adapter_errors(tmp_path: Path) -> None:
    detector = FakeDetector([_mask()])
    segmenter = _segmenter(tmp_path, detector)
    detector.error = RuntimeError("graph failure")

    with pytest.raises(SegmentationError, match="セグメンテーション"):
        segmenter.segment(_rgb(), timestamp_ms=0)


def test_a_failed_warm_up_is_a_recoverable_error(tmp_path: Path) -> None:
    detector = FakeDetector([_mask()])
    detector.error = RuntimeError("graph failure")

    with pytest.raises(SegmentationError, match="モデル"):
        _segmenter(tmp_path, detector)


def test_non_rgb_input_is_rejected(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, FakeDetector([_mask(8, 8)]))

    with pytest.raises(SegmentationError, match="RGB"):
        segmenter.segment(np.zeros((8, 8), dtype=np.uint8), timestamp_ms=0)


def test_model_identity_is_tracked_by_version_and_hash(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, FakeDetector([]))

    assert segmenter.model_version == "selfie-segmenter/1.0.0"
    assert len(segmenter.model_sha256) == 64


def test_missing_model_file_is_reported_before_any_segmentation(tmp_path: Path) -> None:
    with pytest.raises(SegmentationError, match="モデル"):
        MediaPipeSegmenter(
            model_path=tmp_path / "missing.tflite",
            model_version="selfie-segmenter/1.0.0",
            detector_factory=lambda _: FakeDetector([]),
        )


def test_close_releases_the_detector(tmp_path: Path) -> None:
    detector = FakeDetector([])
    segmenter = _segmenter(tmp_path, detector)

    segmenter.close()

    assert detector.closed is True


def test_segmentation_after_close_is_refused(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, FakeDetector([]))
    segmenter.close()

    with pytest.raises(SegmentationError, match="停止"):
        segmenter.segment(_rgb(), timestamp_ms=0)
