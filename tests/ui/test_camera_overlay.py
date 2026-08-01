import numpy as np

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.skin_quality import CameraCorrection, SkinQualityResult
from face_symmetry_lab.domain.symmetry import SymmetryResult
from face_symmetry_lab.ui.camera_overlay import draw_camera_metrics, metric_overlay_lines


def _outcome(*, ok: bool = True) -> AnalysisOutcome:
    symmetry = (
        SymmetryResult(
            scores={"contour": 80.0, "eyes": 81.0, "nose": 82.0, "mouth": 83.0},
            distances={"contour": 0.01, "eyes": 0.02, "nose": 0.03, "mouth": 0.04},
            overall=81.5,
            metric_version="symmetry/test",
            landmark_map_version="map/test",
        )
        if ok
        else None
    )
    return AnalysisOutcome(
        frame_id=5,
        state=AnalysisState(
            status=Status.OK if ok else Status.HOLD,
            reason_codes=() if ok else (ReasonCode.OFF_AXIS_YAW,),
        ),
        symmetry=symmetry,
        overall_smooth=81.25 if ok else None,
        pose=FacePose(yaw_deg=1.2, pitch_deg=-2.3, roll_deg=0.4),
        face_area_ratio=0.25,
        metric_version="symmetry/test",
        landmark_map_version="map/test",
        sensitivity_frozen=False,
        skin_quality=SkinQualityResult(
            score=74.0,
            tone_uniformity=75.0,
            texture_uniformity=73.0,
            pair_count=8,
            correction=CameraCorrection(0.9, 1.0, 1.1, 0.2, 0.8),
        ),
    )


def test_overlay_lines_put_scores_and_pose_in_the_camera_view() -> None:
    lines = metric_overlay_lines(_outcome())

    assert lines[0] == "LIVE SAMPLE | OK"
    assert "OVERALL 81.2" in lines
    assert any("C 80.0" in line for line in lines)
    assert any("YAW +1.2" in line for line in lines)
    assert any("SKIN REF 74.0" in line for line in lines)


def test_overlay_labels_reused_numbers_as_the_last_sample() -> None:
    lines = metric_overlay_lines(_outcome(), current_sample=False)

    assert lines[0] == "LAST SAMPLE | OK"


def test_missing_scores_are_shown_as_missing_not_zero() -> None:
    lines = metric_overlay_lines(_outcome(ok=False))

    assert "OVERALL --" in lines
    assert all("OVERALL 0" not in line for line in lines)


def test_camera_overlay_is_high_contrast_and_does_not_mutate_the_source() -> None:
    source = np.full((720, 1280, 3), 220, dtype=np.uint8)

    painted = draw_camera_metrics(source, _outcome())

    assert painted is not source
    assert np.array_equal(source, np.full_like(source, 220))
    assert painted[:230, :650].min() < 30
    assert painted[:230, :650].max() > 220
    assert float(painted[210, 20].mean()) < 80
