from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from face_symmetry_lab.domain.export import AgeGateDecisionLabel, ExportRecord, InputMode
from face_symmetry_lab.domain.models import ReasonCode, Status


def _record(**overrides: object) -> ExportRecord:
    values: dict[str, object] = {
        "session_id": str(uuid4()),
        "segment_id": str(uuid4()),
        "frame_id": 7,
        "elapsed_ms": 1250,
        "captured_at": datetime(2026, 8, 1, tzinfo=UTC),
        "input_mode": InputMode.CAMERA,
        "analysis_status": Status.OK,
        "face_count": 1,
        "yaw_deg": 1.25,
        "pitch_deg": -2.5,
        "roll_deg": 0.5,
        "face_area_ratio": 0.25,
        "distance_contour_raw": 0.010,
        "distance_eyes_raw": 0.011,
        "distance_nose_raw": 0.012,
        "distance_mouth_raw": 0.013,
        "distance_overall_raw": 0.0115,
        "distance3d_contour_raw": 0.050,
        "distance3d_eyes_raw": 0.051,
        "distance3d_nose_raw": 0.090,
        "distance3d_mouth_raw": 0.052,
        "distance3d_overall_raw": 0.060,
        "distance3d_overall_smooth": 0.061,
        "distance3d_parts_equal_raw": 0.064,
        "skin_quality_score_raw": 72.0,
        "skin_tone_uniformity_raw": 74.0,
        "skin_texture_uniformity_raw": 70.0,
        "skin_correction_gain_r": 0.9,
        "skin_correction_gain_g": 1.0,
        "skin_correction_gain_b": 1.1,
        "skin_metric_version": "skin-surface/0.1.0",
        "skin_quality_frozen": False,
        "landmark_model_version": "landmark/1",
        "landmark_model_sha256": "a" * 64,
        "metric_version": "symmetry-3d/1.0.0",
        "metric_version_2d": "symmetry-2d/2.0.0",
        "app_version": "0.1.0",
        "lock_digest": "b" * 64,
        "config_revision": 1,
        "config_fingerprint": "c" * 64,
        "processing_ms": 12.3456,
        "pipeline_latency_ms": 15.6789,
    }
    values.update(overrides)
    return ExportRecord.model_validate(values)


def test_record_uses_the_breaking_distance_schema() -> None:
    record = _record()

    assert record.schema_version == "4.0.0"
    assert record.metric_version == "symmetry-3d/1.0.0"
    assert record.metric_version_2d == "symmetry-2d/2.0.0"
    assert record.config_fingerprint == "c" * 64
    assert record.distance_mouth_raw == pytest.approx(0.013)
    assert record.distance_overall_raw == pytest.approx(0.0115)
    assert record.skin_quality_score_raw == pytest.approx(72.0)
    assert record.skin_quality_frozen is False
    assert "age_band" not in ExportRecord.model_fields


def test_the_schema_no_longer_carries_the_zero_to_hundred_score_columns() -> None:
    """§8.6/§9.2: `score_*` を距離の別名にせず、削除した。"""
    for removed in (
        "score_contour_raw",
        "score_eyes_raw",
        "score_nose_raw",
        "score_mouth_raw",
        "score_overall_raw",
        "score_overall_smooth",
    ):
        assert removed not in ExportRecord.model_fields


def test_the_v4_schema_drops_the_two_dimensional_smoothed_column() -> None:
    """§8.6.5/§9.2: 2D の EMA は表示・エクスポートのどちらからも読まれなくなった。"""
    assert "distance_overall_smooth" not in ExportRecord.model_fields


def test_the_v4_schema_adds_the_three_dimensional_diagnostic_columns() -> None:
    """§8.6.5/§9.2: `symmetry-3d/1.0.0` の出力一式が列として揃っている。"""
    for added in (
        "distance3d_contour_raw",
        "distance3d_eyes_raw",
        "distance3d_nose_raw",
        "distance3d_mouth_raw",
        "distance3d_overall_raw",
        "distance3d_overall_smooth",
        "distance3d_parts_equal_raw",
    ):
        assert added in ExportRecord.model_fields


def test_the_part_equal_weight_column_can_differ_from_the_overall_column() -> None:
    """§8.6.5: `s_3d`（部位等重み）と `d_3d`（全体、点等重み）は別々の総合量である。

    同じ値へ丸めて書いてしまう実装ならこの罠で落ちる。
    """
    record = _record(distance3d_overall_raw=0.060, distance3d_parts_equal_raw=0.064)

    assert record.distance3d_overall_raw != record.distance3d_parts_equal_raw
    assert record.distance3d_overall_raw != record.distance_overall_raw


def test_a_negative_distance_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(distance_overall_raw=-0.001)


def test_a_distance_above_one_hundred_is_accepted() -> None:
    """距離に上限は無い。上限を持っていたのは旧 0〜100 のスコアだけである。"""
    record = _record(distance_overall_raw=120.0)

    assert record.distance_overall_raw == pytest.approx(120.0)


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(processing_ms=float("nan"))


def test_non_ok_records_cannot_carry_current_distances() -> None:
    with pytest.raises(ValidationError, match="OK"):
        _record(analysis_status=Status.HOLD)


def test_missing_values_are_null_not_zero_filled() -> None:
    record = _record(
        analysis_status=Status.HOLD,
        reason_codes=(ReasonCode.OFF_AXIS_YAW,),
        distance_overall_raw=None,
        distance_contour_raw=None,
        distance_eyes_raw=None,
        distance_nose_raw=None,
        distance_mouth_raw=None,
        distance3d_contour_raw=None,
        distance3d_eyes_raw=None,
        distance3d_nose_raw=None,
        distance3d_mouth_raw=None,
        distance3d_overall_raw=None,
        distance3d_overall_smooth=None,
        distance3d_parts_equal_raw=None,
        skin_quality_score_raw=None,
        skin_tone_uniformity_raw=None,
        skin_texture_uniformity_raw=None,
        skin_correction_gain_r=None,
        skin_correction_gain_g=None,
        skin_correction_gain_b=None,
        skin_metric_version=None,
        skin_quality_frozen=None,
    )

    assert record.distance_overall_raw is None
    assert record.distance3d_overall_raw is None
    assert record.distance3d_overall_smooth is None
    assert record.distance3d_parts_equal_raw is None
    assert record.reason_codes == (ReasonCode.OFF_AXIS_YAW,)


# --- §9.2 / FR-038: 年齢ゲートの列（2026-08-16） -----------------------------


def test_a_record_can_carry_the_gate_decision_and_its_version() -> None:
    """§9.2: この列が無いと、`estimated_age_years` が null のとき「ゲートで
    弾かれた」のか「モデル未配置」なのかを行から区別できない。"""
    record = _record(
        age_gate_decision=AgeGateDecisionLabel.OUT_OF_RANGE,
        age_gate_version="age-gate/1.0.0",
        age_gate_spec_revision=3,
        age_gate_model_sha256="b" * 64,
    )

    assert record.age_gate_decision is AgeGateDecisionLabel.OUT_OF_RANGE
    assert record.age_gate_spec_revision == 3


def test_the_gate_columns_default_to_missing() -> None:
    """ゲートを呼ばなかったフレーム（顔なし・姿勢ゲート閉）は 4 列とも欠測。"""
    record = _record()

    assert record.age_gate_decision is None
    assert record.age_gate_version is None
    assert record.age_gate_spec_revision is None
    assert record.age_gate_model_sha256 is None


def test_the_gate_label_has_no_name_for_a_frame_it_never_judged() -> None:
    """罠: `not_evaluated` を列の値として持つと、行を読む側が「ゲートが何かを
    決めた」と受け取る。呼ばなかったことは `None` で表す。"""
    assert {label.value for label in AgeGateDecisionLabel} == {
        "in_range",
        "out_of_range",
        "undetermined",
    }


def test_the_raw_gate_probability_has_no_column() -> None:
    """§9.2: 確率を記録すると、凍結閾値以外での再判定や順位付けが可能になる。"""
    assert not any("probab" in name for name in ExportRecord.model_fields)


@pytest.mark.parametrize("bad", ["", "zz" * 32, "a" * 63])
def test_a_malformed_gate_model_hash_is_refused(bad: str) -> None:
    with pytest.raises(ValidationError):
        _record(age_gate_model_sha256=bad)


def test_a_non_positive_spec_revision_is_refused() -> None:
    with pytest.raises(ValidationError):
        _record(age_gate_spec_revision=0)
