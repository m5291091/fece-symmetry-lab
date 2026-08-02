from datetime import UTC, datetime

import numpy as np
import pytest

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.models import AnalysisState, DemographicEstimate, ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.skin_quality import CameraCorrection, SkinQualityResult
from face_symmetry_lab.domain.symmetry import SymmetryResult
from face_symmetry_lab.ui.camera_overlay import (
    OverlayPosition,
    draw_camera_metrics,
    guidance_message,
    metric_overlay_lines,
)


def _outcome(*, ok: bool = True, reason: ReasonCode = ReasonCode.OFF_AXIS_YAW) -> AnalysisOutcome:
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
            reason=None if ok else reason,
            reason_codes=() if ok else (reason,),
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


def _demographic() -> DemographicEstimate:
    return build_demographic_estimate(
        estimated_age_years=34.2,
        gender_probabilities=(0.3, 0.7),
        model_version="demographic/test",
        model_sha256="a" * 64,
        observed_at=datetime.now(UTC),
    )


def test_overlay_lines_put_scores_and_pose_in_the_camera_view() -> None:
    """The compact video panel shows the smoothed value only — the raw
    pre-smoothing value stays in the side panel/export (§8.7, requirement.md
    "UIは平滑値を主表示し、生値は詳細・エクスポートで確認可能にする")."""
    lines = metric_overlay_lines(_outcome())

    assert lines[0] == "LIVE SAMPLE | OK"
    assert "OVERALL 81.2" in lines
    assert all("RAW" not in line for line in lines)
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


def test_guidance_message_is_none_when_the_gate_is_satisfied() -> None:
    state = AnalysisState(status=Status.OK, reason=None, reason_codes=())

    assert guidance_message(state) is None


@pytest.mark.parametrize("reason", list(ReasonCode))
def test_every_reason_code_has_a_japanese_guidance_message(reason: ReasonCode) -> None:
    state = AnalysisState(status=Status.HOLD, reason=reason, reason_codes=(reason,))

    message = guidance_message(state)

    assert message is not None
    assert message != ""


def test_guidance_message_is_actionable_for_common_reasons() -> None:
    too_far = AnalysisState(
        status=Status.HOLD, reason=ReasonCode.TOO_FAR, reason_codes=(ReasonCode.TOO_FAR,)
    )
    too_close = AnalysisState(
        status=Status.HOLD, reason=ReasonCode.TOO_CLOSE, reason_codes=(ReasonCode.TOO_CLOSE,)
    )

    assert guidance_message(too_far) != guidance_message(too_close)


def test_metric_overlay_lines_can_append_reference_age_and_gender() -> None:
    lines = metric_overlay_lines(_outcome(), demographic=_demographic())

    assert lines[0] == "LIVE SAMPLE | OK"  # unchanged prefix
    assert any("34" in line for line in lines)
    assert any("MALE" in line for line in lines)


def test_metric_overlay_lines_omit_the_reference_line_without_a_demographic_estimate() -> None:
    with_demo = metric_overlay_lines(_outcome(), demographic=_demographic())
    without_demo = metric_overlay_lines(_outcome())

    assert len(with_demo) == len(without_demo) + 1


def test_camera_overlay_draws_reference_age_and_gender_for_fullscreen_video() -> None:
    """FR-033: fullscreen only shows the <video> element, so age/gender must
    be burned into the frame pixels the same way the score already is."""
    source = np.full((720, 1280, 3), 220, dtype=np.uint8)

    without_demo = draw_camera_metrics(source, _outcome())
    with_demo = draw_camera_metrics(source, _outcome(), demographic=_demographic())

    assert not np.array_equal(without_demo, with_demo)


@pytest.mark.parametrize(
    ("position", "region"),
    [
        (OverlayPosition.TOP_LEFT, np.s_[:230, :650]),
        (OverlayPosition.TOP_RIGHT, np.s_[:230, 630:]),
        (OverlayPosition.BOTTOM_LEFT, np.s_[490:, :650]),
        (OverlayPosition.BOTTOM_RIGHT, np.s_[490:, 630:]),
    ],
)
def test_the_score_panel_is_drawn_at_the_selected_corner(position, region) -> None:
    source = np.full((720, 1280, 3), 220, dtype=np.uint8)

    painted = draw_camera_metrics(source, _outcome(), position=position)

    assert painted[region].min() < 30


def test_the_default_position_is_top_left_for_backward_compatibility() -> None:
    source = np.full((720, 1280, 3), 220, dtype=np.uint8)

    default_position = draw_camera_metrics(source, _outcome())
    explicit_top_left = draw_camera_metrics(source, _outcome(), position=OverlayPosition.TOP_LEFT)

    assert np.array_equal(default_position, explicit_top_left)


def test_draw_camera_metrics_never_draws_a_guidance_banner() -> None:
    """Regression guard: Japanese guidance text drawn onto the frame with
    PIL's default bitmap font rendered as unreadable tofu boxes (no CJK
    glyphs). Guidance now lives only in the "ライブ解析" side panel, which
    renders Japanese natively through the browser."""
    source = np.full((720, 1280, 3), 20, dtype=np.uint8)

    ok_frame = draw_camera_metrics(source, _outcome(ok=True), position=OverlayPosition.BOTTOM_RIGHT)
    hold_frame = draw_camera_metrics(
        source,
        _outcome(ok=False, reason=ReasonCode.TOO_FAR),
        position=OverlayPosition.BOTTOM_RIGHT,
    )

    # With the score panel confined to the bottom-right, the top band must be
    # identical regardless of gate status -- nothing else gets drawn there.
    assert np.array_equal(ok_frame[:80, :], hold_frame[:80, :])
