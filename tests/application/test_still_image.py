import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import LandmarkDetectionError
from face_symmetry_lab.application.still_image import (
    analyze_still_image,
    detect_still_image_landmarks,
)
from face_symmetry_lab.domain.landmarks import (
    LandmarkFrame,
    TrackingStatus,
    build_landmark_frame,
)
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.models import Status
from face_symmetry_lab.domain.pose import FacePose


class FakeImageLandmarker:
    def __init__(self, face_count: int = 1) -> None:
        self.face_count = face_count
        self.error: Exception | None = None
        self.calls = 0

    def detect(self, rgb_image: NDArray[np.uint8], *, frame_id: int = 0) -> LandmarkFrame:
        self.calls += 1
        if self.error is not None:
            raise self.error
        points = tuple((0.4 + 0.1 * i, 0.5) for i in range(3)) if self.face_count == 1 else ()
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=self.face_count,
            points=points,
            width=rgb_image.shape[1],
            height=rgb_image.shape[0],
        )

    def close(self) -> None:
        return None


def _rgb(height: int = 40, width: int = 40) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_single_face_images_get_an_annotated_preview() -> None:
    result = detect_still_image_landmarks(FakeImageLandmarker(face_count=1), _rgb())

    assert result.frame.status is TrackingStatus.SINGLE_FACE
    assert result.landmark_count == 3
    assert result.preview.sum() > 0
    assert result.preview.shape == (40, 40, 3)


def test_multi_face_images_are_reported_without_an_overlay() -> None:
    result = detect_still_image_landmarks(FakeImageLandmarker(face_count=2), _rgb())

    assert result.frame.status is TrackingStatus.MULTI_FACE
    assert result.landmark_count == 0
    assert result.preview.sum() == 0


def test_no_face_images_are_reported_without_an_overlay() -> None:
    result = detect_still_image_landmarks(FakeImageLandmarker(face_count=0), _rgb())

    assert result.frame.status is TrackingStatus.NO_FACE
    assert result.preview.sum() == 0


def test_the_source_image_is_not_mutated() -> None:
    source = _rgb()

    detect_still_image_landmarks(FakeImageLandmarker(face_count=1), source)

    assert source.sum() == 0


def test_detection_failures_are_surfaced_to_the_caller() -> None:
    landmarker = FakeImageLandmarker()
    landmarker.error = LandmarkDetectionError("model failure")

    with pytest.raises(LandmarkDetectionError):
        detect_still_image_landmarks(landmarker, _rgb())


def test_still_images_are_stateless_and_reuse_frame_id_zero() -> None:
    landmarker = FakeImageLandmarker()

    first = detect_still_image_landmarks(landmarker, _rgb())
    second = detect_still_image_landmarks(landmarker, _rgb())

    assert first.frame.frame_id == 0
    assert second.frame.frame_id == 0
    assert landmarker.calls == 2


def test_still_image_midline_can_be_turned_on_and_off() -> None:
    source = _rgb()
    landmarker = FakeImageLandmarker()

    shown = detect_still_image_landmarks(
        landmarker,
        source,
        show_midline=True,
        midline_indices=(0, 1, 2),
    )
    hidden = detect_still_image_landmarks(
        landmarker,
        source,
        show_midline=False,
        midline_indices=(0, 1, 2),
    )

    assert not np.array_equal(shown.preview, hidden.preview)


# --------------------------------------------------------------------------
# §8.7: 静止画は stateless — 3D の平滑値は常に欠測
# --------------------------------------------------------------------------

#: 640x480 フレーム内にある、顔面積ゲート（既定 0.02〜0.55）を満たす正規化点。
_BASE_POINTS = (
    (0.40, 0.40),  # 0 左目 外側
    (0.45, 0.40),  # 1 左目 内側
    (0.55, 0.40),  # 2 右目 内側
    (0.60, 0.40),  # 3 右目 外側
    (0.50, 0.30),  # 4 正中
    (0.50, 0.50),  # 5 正中
    (0.50, 0.62),  # 6 正中
    (0.35, 0.55),  # 7 輪郭
    (0.65, 0.55),  # 8
    (0.47, 0.50),  # 9 鼻
    (0.53, 0.50),  # 10
    (0.44, 0.58),  # 11 口
    (0.56, 0.58),  # 12
)
SYMMETRIC_POINTS = tuple((0.5 + (x - 0.5) * 1.3, 0.5 + (y - 0.5) * 1.3) for x, y in _BASE_POINTS)


def _depths(points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    """指標が z を読んでいれば必ず値が動く、非対称で非自明な奥行き。"""
    return tuple(0.05 * (x - 0.5) - 0.02 * y for x, y in points)


def _still_metric_config(metric_version: str):
    return build_metric_config(
        metric_version=metric_version,
        landmark_map_version="test/1.0.0",
        landmark_count=len(SYMMETRIC_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8),), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
            "nose": PartMap(pairs=((9, 10),), min_pairs=1),
            "mouth": PartMap(pairs=((11, 12),), min_pairs=1),
        },
    )


class DepthAwareLandmarker:
    """奥行き付きの単一顔フレームを 1 枚返す（`ImageLandmarkProvider` 契約）。"""

    def detect(self, rgb_image: NDArray[np.uint8], *, frame_id: int = 0) -> LandmarkFrame:
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=1,
            points=SYMMETRIC_POINTS,
            depths=_depths(SYMMETRIC_POINTS),
            pose=FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
            width=640,
            height=480,
        )

    def close(self) -> None:
        return None


def test_a_still_image_never_reports_a_smoothed_three_dimensional_distance() -> None:
    """§8.7: 静止画は stateless。

    `analyze_still_image` は毎回新しい `FrameAnalyzer`（=新しい EMA）を使うため、
    §8.5.1 の重ね合わせが成立すれば `overall_distance_3d_smooth` はその EMA への
    *最初の* 更新となり、何もしなければ生値と等しい非 null 値を返してしまう
    （罠）。`AnalysisOutcome.overall_distance_3d_smooth` は明示的に欠測へ
    落とさなければならない。
    """
    analysis = analyze_still_image(
        DepthAwareLandmarker(),
        np.zeros((480, 640, 3), dtype=np.uint8),
        config=_still_metric_config("symmetry-2d/test"),
        config_3d=_still_metric_config("symmetry-3d/test"),
    )

    assert analysis.outcome.state.status is Status.OK
    assert analysis.outcome.symmetry_3d is not None
    assert analysis.outcome.symmetry_3d.overall is not None
    assert analysis.outcome.symmetry_3d.overall.distance > 0.0  # z が非対称なので 0 ではない
    assert analysis.outcome.overall_distance_3d_smooth is None


def test_a_still_image_never_reports_smoothed_part_distances_either() -> None:
    """§8.7 改訂（H1）: 部位別 (`d_3d,p`) の平滑値も、全体値と同じ理由で
    stateless な静止画では欠測へ落とす。`FrameAnalyzer` を毎回新しく作るため、
    何もしなければ部位別 EMA への最初の更新(=生値と等しい非 null 値)を返して
    しまう(全体値と同じ罠)。UI は §8.6.3 のとおり部位別の生値へフォールバック
    する。
    """
    analysis = analyze_still_image(
        DepthAwareLandmarker(),
        np.zeros((480, 640, 3), dtype=np.uint8),
        config=_still_metric_config("symmetry-2d/test"),
        config_3d=_still_metric_config("symmetry-3d/test"),
    )

    assert analysis.outcome.symmetry_3d is not None
    assert analysis.outcome.symmetry_3d.distance("mouth") is not None
    assert analysis.outcome.part_distances_3d_smooth is None
