from datetime import UTC, datetime

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.application.session import (
    SessionRecorder,
    build_config_settings,
    config_fingerprint,
)
from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.metrics import PART_NAMES, PartMap, build_metric_config
from face_symmetry_lab.domain.models import AGE_BANDS, AgeEstimate, AnalysisState, Status
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.symmetry import SymmetryResult


def _metric_config():
    return build_metric_config(
        metric_version="symmetry/test",
        landmark_map_version="map/test",
        landmark_count=16,
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8),), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
            "nose": PartMap(pairs=((9, 10),), min_pairs=1),
            "mouth": PartMap(pairs=((11, 12),), min_pairs=1),
        },
        sensitivity=dict.fromkeys(PART_NAMES, 30.0),
        weights=dict.fromkeys(PART_NAMES, 0.25),
        sensitivity_frozen=False,
    )


def test_config_fingerprint_is_canonical_and_covers_metric_and_gate_settings() -> None:
    config = _metric_config()
    settings = build_config_settings(
        config,
        QualityThresholds(),
        tau_seconds=0.4,
        analysis_interval_ms=125,
    )

    assert config_fingerprint(settings) == config_fingerprint(dict(reversed(settings.items())))
    changed = dict(settings)
    changed["ema_tau_seconds"] = 0.5
    assert config_fingerprint(changed) != config_fingerprint(settings)
    assert settings["sensitivity"] == dict.fromkeys(PART_NAMES, 30.0)
    changed_interval = dict(settings)
    changed_interval["analysis_interval_ms"] = 200
    assert config_fingerprint(changed_interval) != config_fingerprint(settings)


def test_session_recorder_is_bounded_and_reset_starts_a_new_segment() -> None:
    recorder = SessionRecorder(
        metric_config=_metric_config(),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
        history_capacity=2,
    )
    old_segment = recorder.segment_id

    recorder.append_error(frame_id=1, elapsed_ms=1, processing_ms=1.0)
    recorder.append_error(frame_id=2, elapsed_ms=2, processing_ms=1.0)
    recorder.append_error(frame_id=3, elapsed_ms=3, processing_ms=1.0)

    assert [record.frame_id for record in recorder.records()] == [2, 3]

    recorder.reset()

    assert recorder.records() == ()
    assert recorder.segment_id != old_segment
    assert recorder.config_revision == 2


def test_analysis_records_keep_scores_distances_and_reproducibility_metadata() -> None:
    config = _metric_config()
    recorder = SessionRecorder(
        metric_config=config,
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
    )
    frame = build_landmark_frame(
        frame_id=9,
        face_count=1,
        points=((0.4, 0.4), (0.6, 0.4)),
        pose=FacePose(yaw_deg=1.0, pitch_deg=2.0, roll_deg=3.0),
        width=640,
        height=480,
    )
    symmetry = SymmetryResult(
        scores={"contour": 80.0, "eyes": 81.0, "nose": 82.0, "mouth": 83.0},
        distances={"contour": 0.01, "eyes": 0.02, "nose": 0.03, "mouth": 0.04},
        overall=81.5,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
    outcome = AnalysisOutcome(
        frame_id=9,
        state=AnalysisState(status=Status.OK),
        symmetry=symmetry,
        overall_smooth=81.25,
        pose=frame.pose,
        face_area_ratio=frame.face_area_ratio,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
        sensitivity_frozen=False,
    )

    record = recorder.append_analysis(
        frame,
        outcome,
        elapsed_ms=125,
        processing_ms=4.567,
    )

    assert record.score_overall_raw == 81.5
    assert record.distance_mouth_raw == 0.04
    assert record.metric_version == config.metric_version
    assert (
        record.config_fingerprint
        == recorder.export_bundle().manifest.config_revisions[0].fingerprint
    )

    enriched = recorder.attach_age_estimate(
        AgeEstimate(
            band=AGE_BANDS[-1],
            confidence=0.6,
            probabilities=(0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.6),
            model_version="age/1",
            model_sha256="d" * 64,
            observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        observed_elapsed_ms=130,
    )

    assert enriched is not None
    assert enriched.age_band == "70+"
    assert enriched.age_prob_70_plus == 0.6
    assert recorder.export_bundle().manifest.models[-1].version == "age/1"
