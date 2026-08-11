from datetime import UTC, datetime

import pytest

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.application.session import (
    SessionRecorder,
    build_config_settings,
    config_fingerprint,
)
from face_symmetry_lab.domain.export import InputMode
from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.models import (
    AnalysisState,
    DemographicEstimate,
    GenderLabel,
    Status,
)
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.skin_quality import CameraCorrection, SkinQualityResult
from face_symmetry_lab.domain.symmetry import SymmetryResult
from face_symmetry_lab.domain.symmetry_3d import AxisSplit, Symmetry3DResult

#: §8.1.1/§8.6.5: 2D bundle と 3D bundle は同じ対応ペア表を共有し、`metric_version`
#: だけが異なる（本物の bundle と同じ形）。テストの罠として、2D と 3D の版文字列は
#: 必ず異なるものを使う。
_PARTS = {
    "contour": PartMap(pairs=((7, 8),), min_pairs=1),
    "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
    "nose": PartMap(pairs=((9, 10),), min_pairs=1),
    "mouth": PartMap(pairs=((11, 12),), min_pairs=1),
}


def _metric_config_2d():
    return build_metric_config(
        metric_version="symmetry-2d/test",
        landmark_map_version="map/test",
        landmark_count=16,
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts=_PARTS,
    )


def _metric_config_3d():
    return build_metric_config(
        metric_version="symmetry-3d/test",
        landmark_map_version="map/test",
        landmark_count=16,
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts=_PARTS,
    )


def test_config_fingerprint_is_canonical_and_covers_metric_and_gate_settings() -> None:
    """§9.2: fingerprint の基になる設定は主指標（3D bundle）を反映する。"""
    config_3d = _metric_config_3d()
    settings = build_config_settings(
        config_3d,
        QualityThresholds(),
        tau_seconds=0.4,
        analysis_interval_ms=125,
    )

    assert config_fingerprint(settings) == config_fingerprint(dict(reversed(settings.items())))
    changed = dict(settings)
    changed["ema_tau_seconds"] = 0.5
    assert config_fingerprint(changed) != config_fingerprint(settings)
    # §8.6 は係数を残さなかった。したがって 2 つの区間を比較可能にするのは、
    # まとめた総合距離を折りたたむ元になるペアの対応表である。両 bundle は同じ
    # ペア表を共有するため、3D から取っても値は変わらない。
    assert settings["part_pair_counts"] == {
        "contour": 1,
        "eyes": 2,
        "nose": 1,
        "mouth": 1,
    }
    assert settings["overall_min_pairs"] == 4
    assert settings["metric_version"] == "symmetry-3d/test"
    changed_interval = dict(settings)
    changed_interval["analysis_interval_ms"] = 200
    assert config_fingerprint(changed_interval) != config_fingerprint(settings)


def test_config_fingerprint_changes_when_the_primary_bundle_version_changes() -> None:
    """§9.2: 主指標が 3D になったので、3D の版が変われば fingerprint も変わる。"""
    settings_a = build_config_settings(_metric_config_3d(), QualityThresholds(), tau_seconds=0.4)
    other_3d = build_metric_config(
        metric_version="symmetry-3d/other",
        landmark_map_version="map/test",
        landmark_count=16,
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts=_PARTS,
    )
    settings_b = build_config_settings(other_3d, QualityThresholds(), tau_seconds=0.4)

    assert config_fingerprint(settings_a) != config_fingerprint(settings_b)


def test_session_recorder_is_bounded_and_reset_starts_a_new_segment() -> None:
    recorder = SessionRecorder(
        metric_config=_metric_config_2d(),
        metric_config_3d=_metric_config_3d(),
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


def test_a_bare_reset_keeps_the_same_config_fingerprint() -> None:
    recorder = SessionRecorder(
        metric_config=_metric_config_2d(),
        metric_config_3d=_metric_config_3d(),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
    )
    before = recorder.export_bundle().manifest.config_revisions[0].fingerprint

    recorder.reset()

    after = recorder.export_bundle().manifest.config_revisions[0].fingerprint
    assert after == before


def test_reset_with_new_thresholds_recomputes_the_config_fingerprint() -> None:
    """§15.2: Advanced 設定の変更は revision と fingerprint の両方を更新する。"""
    recorder = SessionRecorder(
        metric_config=_metric_config_2d(),
        metric_config_3d=_metric_config_3d(),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
    )
    before = recorder.export_bundle().manifest.config_revisions[0]

    recorder.reset(thresholds=QualityThresholds(valid_frames_to_ok=1), tau_seconds=0.8)

    after = recorder.export_bundle().manifest.config_revisions[0]
    assert after.revision == before.revision + 1
    assert after.fingerprint != before.fingerprint
    assert after.settings["valid_frames_to_ok"] == 1
    assert after.settings["ema_tau_seconds"] == 0.8


def test_reset_does_not_accept_a_replacement_metric_bundle() -> None:
    """§8.6/§15.2: bundle に調整対象が残っていないので、セッション単位で固定である。"""
    recorder = SessionRecorder(
        metric_config=_metric_config_2d(),
        metric_config_3d=_metric_config_3d(),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
    )

    with pytest.raises(TypeError):
        recorder.reset(metric_config=_metric_config_2d())  # type: ignore[call-arg]


def test_reset_with_a_new_history_capacity_rebounds_the_buffer() -> None:
    """§15.2: history_capacity は直後の記録から効かなければならない。"""
    recorder = SessionRecorder(
        metric_config=_metric_config_2d(),
        metric_config_3d=_metric_config_3d(),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
        history_capacity=100,
    )
    recorder.append_error(frame_id=1, elapsed_ms=1, processing_ms=1.0)

    recorder.reset(history_capacity=2)
    recorder.append_error(frame_id=10, elapsed_ms=1, processing_ms=1.0)
    recorder.append_error(frame_id=11, elapsed_ms=2, processing_ms=1.0)
    recorder.append_error(frame_id=12, elapsed_ms=3, processing_ms=1.0)

    assert [record.frame_id for record in recorder.records()] == [11, 12]


def test_resize_history_capacity_preserves_existing_records() -> None:
    """§14.3d: cadence の自動調整が、reset() のように履歴を分断してはならない。"""
    recorder = SessionRecorder(
        metric_config=_metric_config_2d(),
        metric_config_3d=_metric_config_3d(),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
        history_capacity=100,
    )
    old_segment = recorder.segment_id
    old_revision = recorder.config_revision
    recorder.append_error(frame_id=1, elapsed_ms=1, processing_ms=1.0)
    recorder.append_error(frame_id=2, elapsed_ms=2, processing_ms=1.0)

    recorder.resize_history_capacity(5)

    assert [record.frame_id for record in recorder.records()] == [1, 2]
    assert recorder.segment_id == old_segment
    assert recorder.config_revision == old_revision


def test_resize_history_capacity_drops_the_oldest_records_when_shrinking() -> None:
    recorder = SessionRecorder(
        metric_config=_metric_config_2d(),
        metric_config_3d=_metric_config_3d(),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
        history_capacity=100,
    )
    recorder.append_error(frame_id=1, elapsed_ms=1, processing_ms=1.0)
    recorder.append_error(frame_id=2, elapsed_ms=2, processing_ms=1.0)
    recorder.append_error(frame_id=3, elapsed_ms=3, processing_ms=1.0)

    recorder.resize_history_capacity(2)

    assert [record.frame_id for record in recorder.records()] == [2, 3]


def test_resize_history_capacity_rejects_less_than_one() -> None:
    recorder = SessionRecorder(
        metric_config=_metric_config_2d(),
        metric_config_3d=_metric_config_3d(),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
    )

    with pytest.raises(ValueError):
        recorder.resize_history_capacity(0)


def _axis_split(value: float) -> AxisSplit:
    return AxisSplit(distance=value, plane=value, depth=0.0)


def test_analysis_records_keep_distances_and_reproducibility_metadata() -> None:
    """§8.6.5/§9.2: v4.0 のレコードは 2D を診断列として残しつつ、主指標は 3D になる。

    2D と 3D の距離は意図的に大きく離した値を使う。表示・エクスポートの配線が
    誤って 2D へ戻っていれば、この差で assertion が落ちる。
    """
    config = _metric_config_2d()
    config_3d = _metric_config_3d()
    recorder = SessionRecorder(
        metric_config=config,
        metric_config_3d=config_3d,
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
        distances={"contour": 0.01, "eyes": 0.02, "nose": 0.03, "mouth": 0.04},
        overall=0.025,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
    # §8.6.5: 部位を意図的に不揃いにし、s_3d（部位等重み）と d_3d（点等重み、全体）
    # が一致しない fixture にする（AC-SCORE-10 と同じ罠）。
    symmetry_3d = Symmetry3DResult(
        splits={
            "contour": _axis_split(0.05),
            "eyes": _axis_split(0.05),
            "nose": _axis_split(0.60),
            "mouth": _axis_split(0.05),
        },
        overall=_axis_split(0.20),
        metric_version=config_3d.metric_version,
        landmark_map_version=config_3d.landmark_map_version,
    )
    outcome = AnalysisOutcome(
        frame_id=9,
        state=AnalysisState(status=Status.OK),
        symmetry=symmetry,
        overall_distance_smooth=0.024,
        pose=frame.pose,
        face_area_ratio=frame.face_area_ratio,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
        skin_quality=SkinQualityResult(
            score=72.0,
            tone_uniformity=74.0,
            texture_uniformity=70.0,
            pair_count=12,
            correction=CameraCorrection(0.9, 1.0, 1.1, 0.2, 0.8),
        ),
        symmetry_3d=symmetry_3d,
        overall_distance_3d_smooth=0.21,
    )

    record = recorder.append_analysis(
        frame,
        outcome,
        elapsed_ms=125,
        processing_ms=4.567,
    )

    # 2D は診断列として残る。
    assert record.distance_overall_raw == 0.025
    assert record.distance_mouth_raw == 0.04
    assert not hasattr(record, "distance_overall_smooth")

    # 3D が主指標としてエクスポートへ届く。
    assert record.distance3d_contour_raw == pytest.approx(0.05)
    assert record.distance3d_nose_raw == pytest.approx(0.60)
    assert record.distance3d_overall_raw == pytest.approx(0.20)
    assert record.distance3d_overall_smooth == pytest.approx(0.21)
    assert record.distance3d_parts_equal_raw == pytest.approx(symmetry_3d.equal_weight_distance)

    # 罠: 2D と 3D の生値、および d_3d と s_3d は別の値である。
    assert record.distance3d_overall_raw != record.distance_overall_raw
    assert record.distance3d_parts_equal_raw != record.distance3d_overall_raw

    # 罠: 主指標（3D）と 2D、両方の版が行から読める。
    assert record.metric_version == config_3d.metric_version
    assert record.metric_version_2d == config.metric_version
    assert record.metric_version != record.metric_version_2d

    assert record.skin_quality_score_raw == 72.0
    assert record.skin_tone_uniformity_raw == 74.0
    assert record.skin_metric_version == "skin-surface/0.1.0"
    assert (
        record.config_fingerprint
        == recorder.export_bundle().manifest.config_revisions[0].fingerprint
    )


def test_a_missing_smoothed_three_dimensional_value_is_recorded_as_null() -> None:
    """§8.7: stateless な解析（静止画）の欠測平滑値は、0 や生値で埋めず null のまま
    記録する。"""
    config = _metric_config_2d()
    config_3d = _metric_config_3d()
    recorder = SessionRecorder(
        metric_config=config,
        metric_config_3d=config_3d,
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
    )
    frame = build_landmark_frame(
        frame_id=1,
        face_count=1,
        points=((0.4, 0.4), (0.6, 0.4)),
        pose=FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
        width=640,
        height=480,
    )
    symmetry_3d = Symmetry3DResult(
        splits={
            "contour": _axis_split(0.05),
            "eyes": _axis_split(0.06),
            "nose": _axis_split(0.07),
            "mouth": _axis_split(0.08),
        },
        overall=_axis_split(0.065),
        metric_version=config_3d.metric_version,
        landmark_map_version=config_3d.landmark_map_version,
    )
    outcome = AnalysisOutcome(
        frame_id=1,
        state=AnalysisState(status=Status.OK),
        symmetry=None,
        overall_distance_smooth=None,
        pose=frame.pose,
        face_area_ratio=frame.face_area_ratio,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
        symmetry_3d=symmetry_3d,
        overall_distance_3d_smooth=None,
    )

    record = recorder.append_analysis(
        frame,
        outcome,
        elapsed_ms=0,
        processing_ms=1.0,
        input_mode=InputMode.IMAGE,
    )

    assert record.distance3d_overall_raw == pytest.approx(0.065)
    assert record.distance3d_overall_smooth is None

    enriched = recorder.attach_demographic_estimate(
        DemographicEstimate(
            estimated_age_years=42.5,
            gender=GenderLabel.MALE,
            gender_confidence=0.6,
            gender_probabilities=(0.4, 0.6),
            model_version="demographic/1",
            model_sha256="d" * 64,
            observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        observed_elapsed_ms=130,
    )

    assert enriched is not None
    assert enriched.estimated_age_years == 42.5
    assert enriched.gender_prob_male == 0.6
    assert recorder.export_bundle().manifest.models[-1].version == "demographic/1"
