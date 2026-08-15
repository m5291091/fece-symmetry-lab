"""FR-038 / AC-AGEGATE-02, AC-AGEGATE-03: 年齢ゲートの use case と失敗分離。

この層が守るのは 2 つ。**安全側の既定**（判定できないなら参考年齢を表示しない）と、
**失敗分離**（ゲートが何をしようと左右差分析は動き続ける、不変条件 12）である。
"""

from __future__ import annotations

import numpy as np
import pytest

from face_symmetry_lab.application.age_gate import (
    AgeGateOutcome,
    AgeGateService,
    AgeGateStatus,
)
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.pose import FacePose
from tests.domain.test_age_gate import IDX, _landmarks


class _StubSymmetry3D:
    def __init__(self, value: float = 0.03):
        self._value = value
        self.equal_weight_distance = value

    def distance(self, name: str) -> float | None:
        return self._value


class _Classifier:
    """`AgeGateClassifier` port の fake。呼ばれた回数を数える。"""

    def __init__(self, probability: float = 0.9):
        self.probability = probability
        self.calls: list[tuple[float, ...]] = []

    def predict(self, vector: tuple[float, ...]) -> float:
        self.calls.append(vector)
        return self.probability


class _FailingClassifier:
    def __init__(self, error: Exception):
        self._error = error
        self.calls = 0

    def predict(self, vector: tuple[float, ...]) -> float:
        self.calls += 1
        raise self._error


def _service(classifier: object | None, threshold: float = 0.20) -> AgeGateService:
    return AgeGateService(classifier, threshold=threshold, spec_version="age-gate/1.0.0")


def _evaluate(service: AgeGateService, **overrides) -> AgeGateOutcome:
    kwargs = {
        "normalized_landmarks": _landmarks(),
        "pose": FacePose(1.0, 1.0, 1.0),
        "symmetry_3d": _StubSymmetry3D(),
        "face_count": 1,
        "pose_gate_open": True,
    }
    kwargs.update(overrides)
    return service.evaluate(**kwargs)


def test_in_range_when_probability_exceeds_the_threshold() -> None:
    classifier = _Classifier(0.9)

    outcome = _evaluate(_service(classifier))

    assert outcome.status is AgeGateStatus.IN_RANGE
    assert outcome.may_show_reference_age is True
    assert outcome.decision is not None
    assert outcome.decision.probability_in_range == pytest.approx(0.9)
    assert len(classifier.calls) == 1
    assert len(classifier.calls[0]) == 41


def test_out_of_range_hides_the_reference_age() -> None:
    outcome = _evaluate(_service(_Classifier(0.05)))

    assert outcome.status is AgeGateStatus.OUT_OF_RANGE
    assert outcome.may_show_reference_age is False


def test_missing_classifier_is_undetermined_and_hides_the_reference_age() -> None:
    """テスト 9・10: モデルが無くても例外を投げず、安全側（非表示）へ倒す。"""
    outcome = _evaluate(_service(None))

    assert outcome.status is AgeGateStatus.UNDETERMINED
    assert outcome.may_show_reference_age is False
    assert outcome.decision is None
    assert outcome.reason is not None


@pytest.mark.parametrize(
    "error", [RuntimeError("推論に失敗"), ValueError("shape が違う"), OSError("読めない")]
)
def test_inference_failure_is_absorbed_into_undetermined(error: Exception) -> None:
    """テスト 15: 推論の例外を上位へ伝播させない。型を先読みしない。"""
    outcome = _evaluate(_service(_FailingClassifier(error)))

    assert outcome.status is AgeGateStatus.UNDETERMINED
    assert outcome.may_show_reference_age is False


def test_degenerate_features_are_undetermined_without_calling_the_model() -> None:
    """特徴量が作れないときはモデルを呼ばない（無駄な推論をしない）。"""
    classifier = _Classifier(0.9)
    points = _landmarks()
    points[IDX["chin"]] = (0.0, -1.0, 0.0)  # face_height = 0

    outcome = _evaluate(_service(classifier), normalized_landmarks=points)

    assert outcome.status is AgeGateStatus.UNDETERMINED
    assert outcome.may_show_reference_age is False
    assert classifier.calls == []


def test_gate_is_not_called_without_a_single_face() -> None:
    """テスト 12: 単一顔でなければ呼ばない。"""
    classifier = _Classifier(0.9)

    for face_count in (0, 2, 3):
        outcome = _evaluate(_service(classifier), face_count=face_count)
        assert outcome.status is AgeGateStatus.NOT_EVALUATED
        assert outcome.may_show_reference_age is False

    assert classifier.calls == []


def test_gate_is_not_called_while_the_pose_gate_is_closed() -> None:
    """テスト 12: 姿勢ゲートが閉じている間は呼ばない。

    学習も姿勢ゲート通過フレームだけで行っており、外側は評価されていない
    （model card の限界）。
    """
    classifier = _Classifier(0.9)

    outcome = _evaluate(_service(classifier), pose_gate_open=False)

    assert outcome.status is AgeGateStatus.NOT_EVALUATED
    assert outcome.may_show_reference_age is False
    assert classifier.calls == []


def test_not_evaluated_is_distinguishable_from_undetermined() -> None:
    """「呼ばなかった」と「呼んだが決められなかった」を混ぜない（§9.2）。

    罠: 両方を同じ状態にすると、エクスポートの行から「顔が無かった」のか
    「モデルが壊れている」のかを区別できなくなる。
    """
    not_evaluated = _evaluate(_service(_Classifier()), face_count=0)
    undetermined = _evaluate(_service(None))

    assert not_evaluated.status is not undetermined.status


def test_every_status_that_is_not_in_range_hides_the_reference_age() -> None:
    """AC-AGEGATE-02: 安全側の既定を全状態について固定する。

    罠: 新しい状態を足したときに `may_show_reference_age` の分岐を書き忘れると、
    既定が「表示する」側へ倒れる。
    """
    for status in AgeGateStatus:
        assert (status is AgeGateStatus.IN_RANGE) == AgeGateOutcome(
            status=status, decision=None, reason=None
        ).may_show_reference_age


def test_the_gate_does_not_touch_the_symmetry_result() -> None:
    """テスト 11 / 不変条件 12: ゲートは左右差の入力も出力も変えない。"""
    symmetry = _StubSymmetry3D(0.037)
    points = _landmarks()
    before = points.copy()

    _evaluate(_service(_Classifier(0.01)), normalized_landmarks=points, symmetry_3d=symmetry)

    assert np.array_equal(points, before)
    assert symmetry.equal_weight_distance == pytest.approx(0.037)
    assert all(symmetry.distance(name) == pytest.approx(0.037) for name in PART_NAMES)


def test_service_reports_the_operating_point_it_was_built_with() -> None:
    """版と閾値をエクスポートへ載せられること（§9.2）。"""
    service = _service(_Classifier(0.9), threshold=0.20)

    outcome = _evaluate(service)

    assert outcome.decision is not None
    assert outcome.decision.threshold == pytest.approx(0.20)
    assert outcome.decision.spec_version == "age-gate/1.0.0"
