"""FR-038: 参考年齢の表示ゲートの use case（§9.2, §10.11, AC-AGEGATE-02〜03）。

この層が守るのは 2 つ。**安全側の既定**（判定できないなら参考年齢を表示しない）
と、**失敗分離**（ゲートが何をしようと左右差分析は動き続ける、不変条件 12）で
ある。ゲートの内部判断（`AgeGateDecision`・`reason`）はここでは開発者向けの
短い説明に留め、UI へそのまま出さない。利用者向けの固定文言は
`ui.age_gate_notice` が持つ。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.ports import AgeGateClassifier
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.domain.age_gate import (
    AgeGateDecision,
    AgeGateError,
    SymmetryDistances,
    build_features,
    decide,
)
from face_symmetry_lab.domain.geometry import GeometryError, normalize_landmarks_3d
from face_symmetry_lab.domain.landmarks import LandmarkError, LandmarkFrame, to_pixel_array_3d
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.pose import FacePose

__all__ = [
    "AgeGateOutcome",
    "AgeGateService",
    "AgeGateStatus",
    "evaluate_age_gate_for_frame",
    "pose_gate_is_open",
]


class AgeGateStatus(enum.Enum):
    """ゲートを呼んだ結果、または呼ばなかった理由（§9.2 `age_gate_decision`）。

    `NOT_EVALUATED`（呼ばなかった）と `UNDETERMINED`（呼んだが決められなかった）
    を混ぜない。エクスポート行から「顔が無かった・姿勢ゲートが閉じていた」のか
    「モデルが壊れている」のかを区別できなくなるためである。
    """

    IN_RANGE = "in_range"
    OUT_OF_RANGE = "out_of_range"
    UNDETERMINED = "undetermined"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True)
class AgeGateOutcome:
    """ゲート 1 回分の結果。"""

    status: AgeGateStatus
    decision: AgeGateDecision | None
    #: 開発者向けの短い説明。UI へそのまま出さない（`ui.age_gate_notice` を使う）。
    reason: str | None

    @property
    def may_show_reference_age(self) -> bool:
        """参考年齢・参考性別を表示してよいか。`IN_RANGE` のときだけ真。"""
        return self.status is AgeGateStatus.IN_RANGE


class AgeGateService:
    """FR-038 のゲートを、失敗分離と安全側の既定を保ったまま呼ぶ use case。"""

    def __init__(
        self,
        classifier: AgeGateClassifier | None,
        *,
        threshold: float,
        spec_version: str,
        spec_revision: int | None = None,
        model_sha256: str | None = None,
    ) -> None:
        self._classifier = classifier
        self._threshold = threshold
        self._spec_version = spec_version
        #: §9.2: 記録へ残す来歴。**閾値そのものは列に持たない**——この版から
        #: `configs/age-gate-v1.json` を引ける（§10.11-8）。classifier が無い
        #: フォールバックでは `None` のままで、記録側も `None` になる。
        self._spec_revision = spec_revision
        self._model_sha256 = model_sha256

    @property
    def spec_version(self) -> str:
        return self._spec_version

    @property
    def spec_revision(self) -> int | None:
        return self._spec_revision

    @property
    def model_sha256(self) -> str | None:
        return self._model_sha256

    def evaluate(
        self,
        normalized_landmarks: NDArray[np.float64],
        *,
        pose: FacePose,
        symmetry_3d: SymmetryDistances,
        face_count: int,
        pose_gate_open: bool,
    ) -> AgeGateOutcome:
        """単一顔かつ姿勢ゲート通過時にだけモデルを呼ぶ（§10.11-1・6）。

        順序: (a) 単一顔でない、または姿勢ゲートが閉じていれば `NOT_EVALUATED`
        で classifier を呼ばない。(b) 特徴量を組み立てられなければ
        `UNDETERMINED` で classifier を呼ばない。(c) classifier が無ければ
        `UNDETERMINED`。(d) 推論が例外を投げれば `UNDETERMINED`。(e) それ以外は
        `decide` の結果で `IN_RANGE` / `OUT_OF_RANGE`。
        """
        if face_count != 1 or not pose_gate_open:
            return AgeGateOutcome(
                status=AgeGateStatus.NOT_EVALUATED,
                decision=None,
                reason="単一顔かつ姿勢ゲート通過ではないため、ゲートを呼びませんでした。",
            )

        features = build_features(normalized_landmarks, pose=pose, symmetry_3d=symmetry_3d)
        if features is None:
            return AgeGateOutcome(
                status=AgeGateStatus.UNDETERMINED,
                decision=None,
                reason="特徴量を組み立てられなかったため、判定できませんでした。",
            )

        if self._classifier is None:
            return AgeGateOutcome(
                status=AgeGateStatus.UNDETERMINED,
                decision=None,
                reason="年齢ゲートのモデルが利用できません。",
            )

        try:
            probability = self._classifier.predict(features.as_vector())
        except Exception:
            return AgeGateOutcome(
                status=AgeGateStatus.UNDETERMINED,
                decision=None,
                reason="年齢ゲートの推論に失敗しました。",
            )

        try:
            decision = decide(
                probability,
                threshold=self._threshold,
                spec_version=self._spec_version,
            )
        except AgeGateError:
            return AgeGateOutcome(
                status=AgeGateStatus.UNDETERMINED,
                decision=None,
                reason="年齢ゲートの推論結果が不正でした。",
            )

        status = AgeGateStatus.IN_RANGE if decision.in_range else AgeGateStatus.OUT_OF_RANGE
        return AgeGateOutcome(status=status, decision=decision, reason=None)


def pose_gate_is_open(pose: FacePose | None, thresholds: QualityThresholds) -> bool:
    """FR-038 の姿勢ゲート（|yaw|<=15°, |pitch|<=15°, |roll|<=25°, §10.11-1）。

    既定の `QualityThresholds` の inner 角度は FR-038 の `pose_gate_deg`
    （`configs/age-gate-v1.json`）と同じ値であり、品質ゲート閾値は FR-030 に
    より固定既定値のままなので、両者は常に一致する。学習も同じ姿勢ゲートを
    通過したフレームだけで行っている（model card の限界）。
    """
    if pose is None:
        return False
    return (
        abs(pose.yaw_deg) <= thresholds.yaw_inner_deg
        and abs(pose.pitch_deg) <= thresholds.pitch_inner_deg
        and abs(pose.roll_deg) <= thresholds.roll_inner_deg
    )


def evaluate_age_gate_for_frame(
    service: AgeGateService,
    *,
    frame: LandmarkFrame,
    outcome: AnalysisOutcome,
    config_3d: MetricConfig,
    thresholds: QualityThresholds | None = None,
) -> AgeGateOutcome:
    """`FrameAnalyzer` の出力から `AgeGateService.evaluate` の引数を組み立てる、薄い配線。

    左右差の解析結果（``frame`` / ``outcome``）を読むだけで、一切書き換えない
    （不変条件 12）。正規化に失敗する場合、または対称度・奥行きがまだ確定して
    いない場合は、classifier を呼ばずに `UNDETERMINED` を返す。``thresholds``
    省略時は既定の `QualityThresholds()`（= FR-038 の `pose_gate_deg` と同じ
    inner 角度）を使う。
    """
    active_thresholds = thresholds if thresholds is not None else QualityThresholds()
    pose = outcome.pose
    gate_open = pose_gate_is_open(pose, active_thresholds)
    if not frame.has_single_face or not gate_open or pose is None:
        return AgeGateOutcome(
            status=AgeGateStatus.NOT_EVALUATED,
            decision=None,
            reason="単一顔かつ姿勢ゲート通過ではないため、ゲートを呼びませんでした。",
        )

    symmetry_3d = outcome.symmetry_3d
    if symmetry_3d is None or not frame.has_depth:
        return AgeGateOutcome(
            status=AgeGateStatus.UNDETERMINED,
            decision=None,
            reason="対称度または奥行きが未確定のため、参考年齢を判定できませんでした。",
        )

    try:
        normalized = normalize_landmarks_3d(
            to_pixel_array_3d(frame),
            left_eye_corners=config_3d.left_eye_corners,
            right_eye_corners=config_3d.right_eye_corners,
        )
    except (LandmarkError, GeometryError):
        return AgeGateOutcome(
            status=AgeGateStatus.UNDETERMINED,
            decision=None,
            reason="ランドマークを正規化できなかったため、参考年齢を判定できませんでした。",
        )

    return service.evaluate(
        normalized,
        pose=pose,
        symmetry_3d=symmetry_3d,
        face_count=frame.face_count,
        pose_gate_open=True,
    )
