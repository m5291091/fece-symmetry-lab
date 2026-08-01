from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from face_symmetry_lab.domain.export import ExportRecord, InputMode
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
        "score_contour_raw": 80.0,
        "score_eyes_raw": 81.0,
        "score_nose_raw": 82.0,
        "score_mouth_raw": 83.0,
        "score_overall_raw": 81.5,
        "score_overall_smooth": 81.25,
        "distance_contour_raw": 0.010,
        "distance_eyes_raw": 0.011,
        "distance_nose_raw": 0.012,
        "distance_mouth_raw": 0.013,
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
        "metric_version": "symmetry-2d/1.0.0",
        "app_version": "0.1.0",
        "lock_digest": "b" * 64,
        "config_revision": 1,
        "config_fingerprint": "c" * 64,
        "processing_ms": 12.3456,
        "pipeline_latency_ms": 15.6789,
    }
    values.update(overrides)
    return ExportRecord.model_validate(values)


def test_record_uses_the_breaking_skin_and_demographic_schema() -> None:
    record = _record()

    assert record.schema_version == "2.0.0"
    assert record.metric_version == "symmetry-2d/1.0.0"
    assert record.config_fingerprint == "c" * 64
    assert record.distance_mouth_raw == pytest.approx(0.013)
    assert record.skin_quality_score_raw == pytest.approx(72.0)
    assert record.skin_quality_frozen is False
    assert "age_band" not in ExportRecord.model_fields


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(processing_ms=float("nan"))


def test_non_ok_records_cannot_carry_current_scores_or_distances() -> None:
    with pytest.raises(ValidationError, match="OK"):
        _record(analysis_status=Status.HOLD)


def test_missing_values_are_null_not_zero_filled() -> None:
    record = _record(
        analysis_status=Status.HOLD,
        reason_codes=(ReasonCode.OFF_AXIS_YAW,),
        score_contour_raw=None,
        score_eyes_raw=None,
        score_nose_raw=None,
        score_mouth_raw=None,
        score_overall_raw=None,
        score_overall_smooth=None,
        distance_contour_raw=None,
        distance_eyes_raw=None,
        distance_nose_raw=None,
        distance_mouth_raw=None,
        skin_quality_score_raw=None,
        skin_tone_uniformity_raw=None,
        skin_texture_uniformity_raw=None,
        skin_correction_gain_r=None,
        skin_correction_gain_g=None,
        skin_correction_gain_b=None,
        skin_metric_version=None,
        skin_quality_frozen=None,
    )

    assert record.score_overall_raw is None
    assert record.reason_codes == (ReasonCode.OFF_AXIS_YAW,)
