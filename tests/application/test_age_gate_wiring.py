"""FR-038: `application.age_gate` の配線ヘルパ。

`pose_gate_is_open` / `evaluate_age_gate_for_frame` が対象である。
`AgeGateService` 自体の contract は `tests/application/test_age_gate.py` が持つ
（`domain.age_gate.build_features` は実 478 点の landmark を要求するため、ここでは
経路の組み立てだけを、`.evaluate` の呼び出しを記録するだけの fake service で見る）。
ここで見るのは、実際の `FrameAnalyzer` の出力（`LandmarkFrame` / `AnalysisOutcome`）
から `AgeGateService.evaluate` の引数を組み立てる薄い配線が、単一顔・姿勢ゲート・
対称度・奥行きの各前提を正しく短絡すること、および service を無駄に呼ばないことである。
"""

from __future__ import annotations

import numpy as np
import pytest

from face_symmetry_lab.application.age_gate import (
    AgeGateOutcome,
    AgeGateStatus,
    evaluate_age_gate_for_frame,
    pose_gate_is_open,
)
from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.symmetry_3d import AxisSplit, Symmetry3DResult

_THRESHOLDS = QualityThresholds()

#: §8.2 ゲート内側の値そのものを使い、境界の内外を明確に分ける。
_OPEN_POSE = FacePose(yaw_deg=5.0, pitch_deg=-5.0, roll_deg=10.0)
_CLOSED_POSE = FacePose(yaw_deg=40.0, pitch_deg=0.0, roll_deg=0.0)

#: 8 点・4 部位に 1 ペアずつの、disjoint な最小 `MetricConfig` 用フレーム。
_POINTS = (
    (0.40, 0.40),  # 0 左目 外側
    (0.45, 0.40),  # 1 左目 内側
    (0.55, 0.40),  # 2 右目 内側
    (0.60, 0.40),  # 3 右目 外側
    (0.35, 0.55),  # 4 輪郭
    (0.65, 0.55),  # 5 輪郭
    (0.47, 0.50),  # 6 鼻
    (0.53, 0.50),  # 7 鼻
)
_DEPTHS = tuple(0.01 * index for index in range(len(_POINTS)))


class _ServiceStub:
    """`AgeGateService` の代わりに `.evaluate` の呼び出しだけを記録する fake。

    `evaluate_age_gate_for_frame` が特徴量計算 (`domain.age_gate.build_features`)
    を経由せずに、正しい引数で `service.evaluate` を呼べているかだけを見る。
    """

    def __init__(self, outcome: AgeGateOutcome) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, object]] = []

    def evaluate(self, normalized_landmarks, **kwargs: object) -> AgeGateOutcome:
        self.calls.append({"normalized_landmarks": normalized_landmarks, **kwargs})
        return self._outcome


_IN_RANGE_OUTCOME = AgeGateOutcome(status=AgeGateStatus.IN_RANGE, decision=None, reason=None)


def _config():
    return build_metric_config(
        metric_version="symmetry-3d/test",
        landmark_map_version="test/1.0.0",
        landmark_count=len(_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(0, 1, 2),
        parts={
            "contour": PartMap(pairs=((4, 5),), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3),), min_pairs=1),
            "nose": PartMap(pairs=((6, 7),), min_pairs=1),
            "mouth": PartMap(pairs=((1, 2),), min_pairs=1),
        },
    )


def _complete_symmetry(value: float = 0.02) -> Symmetry3DResult:
    split = AxisSplit(distance=value, plane=value, depth=0.0)
    return Symmetry3DResult(
        splits={"contour": split, "eyes": split, "nose": split, "mouth": split},
        overall=split,
        metric_version="symmetry-3d/test",
        landmark_map_version="test/1.0.0",
    )


def _frame(*, face_count: int = 1, pose: FacePose | None = _OPEN_POSE, with_depth: bool = True):
    return build_landmark_frame(
        frame_id=1,
        face_count=face_count,
        points=_POINTS if face_count == 1 else (),
        depths=(_DEPTHS if with_depth else None) if face_count == 1 else None,
        pose=pose,
        width=100,
        height=100,
    )


_UNSET = object()


def _outcome(
    *, pose: FacePose | None = _OPEN_POSE, symmetry_3d: object = _UNSET
) -> AnalysisOutcome:
    return AnalysisOutcome(
        frame_id=1,
        state=None,  # type: ignore[arg-type]  # このヘルパは state を読まない。
        symmetry=None,
        overall_distance_smooth=None,
        pose=pose,
        face_area_ratio=None,
        metric_version="symmetry-2d/test",
        landmark_map_version="test/1.0.0",
        symmetry_3d=_complete_symmetry() if symmetry_3d is _UNSET else symmetry_3d,  # type: ignore[arg-type]
    )


# --- pose_gate_is_open -------------------------------------------------------


def test_pose_gate_is_open_for_a_pose_within_the_frozen_gate() -> None:
    assert pose_gate_is_open(_OPEN_POSE, _THRESHOLDS) is True


def test_pose_gate_is_closed_without_a_pose() -> None:
    assert pose_gate_is_open(None, _THRESHOLDS) is False


@pytest.mark.parametrize(
    "pose",
    [
        FacePose(yaw_deg=40.0, pitch_deg=0.0, roll_deg=0.0),
        FacePose(yaw_deg=0.0, pitch_deg=40.0, roll_deg=0.0),
        FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=40.0),
    ],
)
def test_pose_gate_is_closed_outside_any_single_axis(pose: FacePose) -> None:
    """罠: yaw だけを見る実装は pitch/roll の逸脱を見逃す。"""
    assert pose_gate_is_open(pose, _THRESHOLDS) is False


# --- evaluate_age_gate_for_frame ---------------------------------------------


def test_calls_the_service_with_a_normalized_landmark_array_when_ready() -> None:
    service = _ServiceStub(_IN_RANGE_OUTCOME)

    outcome = evaluate_age_gate_for_frame(
        service,  # type: ignore[arg-type]
        frame=_frame(),
        outcome=_outcome(),
        config_3d=_config(),
    )

    assert outcome is _IN_RANGE_OUTCOME
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["face_count"] == 1
    assert call["pose_gate_open"] is True
    assert call["pose"] is _OPEN_POSE
    normalized = call["normalized_landmarks"]
    assert normalized.shape == (len(_POINTS), 3)
    assert np.isfinite(normalized).all()


def test_does_not_call_the_service_without_a_single_face() -> None:
    service = _ServiceStub(_IN_RANGE_OUTCOME)

    outcome = evaluate_age_gate_for_frame(
        service,  # type: ignore[arg-type]
        frame=_frame(face_count=0),
        outcome=_outcome(),
        config_3d=_config(),
    )

    assert outcome.status is AgeGateStatus.NOT_EVALUATED
    assert outcome.may_show_reference_age is False
    assert service.calls == []


def test_does_not_call_the_service_when_the_pose_gate_is_closed() -> None:
    service = _ServiceStub(_IN_RANGE_OUTCOME)

    outcome = evaluate_age_gate_for_frame(
        service,  # type: ignore[arg-type]
        frame=_frame(pose=_CLOSED_POSE),
        outcome=_outcome(pose=_CLOSED_POSE),
        config_3d=_config(),
    )

    assert outcome.status is AgeGateStatus.NOT_EVALUATED
    assert service.calls == []


def test_does_not_call_the_service_without_a_pose() -> None:
    service = _ServiceStub(_IN_RANGE_OUTCOME)

    outcome = evaluate_age_gate_for_frame(
        service,  # type: ignore[arg-type]
        frame=_frame(pose=None),
        outcome=_outcome(pose=None),
        config_3d=_config(),
    )

    assert outcome.status is AgeGateStatus.NOT_EVALUATED
    assert service.calls == []


def test_is_undetermined_without_symmetry_3d() -> None:
    service = _ServiceStub(_IN_RANGE_OUTCOME)

    outcome = evaluate_age_gate_for_frame(
        service,  # type: ignore[arg-type]
        frame=_frame(),
        outcome=_outcome(symmetry_3d=None),
        config_3d=_config(),
    )

    assert outcome.status is AgeGateStatus.UNDETERMINED
    assert service.calls == []


def test_is_undetermined_without_depth() -> None:
    service = _ServiceStub(_IN_RANGE_OUTCOME)

    outcome = evaluate_age_gate_for_frame(
        service,  # type: ignore[arg-type]
        frame=_frame(with_depth=False),
        outcome=_outcome(),
        config_3d=_config(),
    )

    assert outcome.status is AgeGateStatus.UNDETERMINED
    assert service.calls == []


def test_does_not_mutate_the_frame_or_outcome() -> None:
    service = _ServiceStub(_IN_RANGE_OUTCOME)
    frame = _frame()
    outcome = _outcome()
    points_before = np.array(frame.points)

    evaluate_age_gate_for_frame(
        service,  # type: ignore[arg-type]
        frame=frame,
        outcome=outcome,
        config_3d=_config(),
    )

    assert np.array_equal(np.array(frame.points), points_before)
    assert outcome.symmetry_3d is not None
    assert outcome.symmetry_3d.distance("mouth") == pytest.approx(0.02)
