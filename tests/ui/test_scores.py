import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.models import ReasonCode
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE


def _run(body: str) -> AppTest:
    at = AppTest.from_string(
        "from face_symmetry_lab.domain.landmarks import TrackingStatus\n"
        "from face_symmetry_lab.ui.scores import (\n"
        "    render_headline_metrics,\n"
        "    render_tracking_status,\n"
        ")\n"
        "MESSAGES = {status: status.name.lower().replace('_', ' ') for status in TrackingStatus}\n"
        f"{body}\n"
    )
    at.run()
    return at


@pytest.mark.parametrize(
    ("status", "severity"),
    [
        (TrackingStatus.ERROR, "error"),
        (TrackingStatus.MULTI_FACE, "warning"),
        (TrackingStatus.SINGLE_FACE, "success"),
        (TrackingStatus.NO_FACE, "info"),
    ],
)
def test_each_tracking_state_renders_at_its_assigned_severity(
    status: TrackingStatus,
    severity: str,
) -> None:
    """§5.2 fixes the colour per state, so Camera and Image cannot disagree."""
    at = _run(f"render_tracking_status(TrackingStatus.{status.name}, MESSAGES)")

    expected = status.name.lower().replace("_", " ")
    assert not at.exception
    assert [element.value for element in getattr(at, severity)] == [expected]


def test_missing_headline_values_render_as_a_dash_rather_than_a_number() -> None:
    """NFR-013: absent is shown as absent, never as a stale or zeroed value."""
    at = _run("render_headline_metrics(overall=None, estimate=None)")

    assert not at.exception
    assert [element.value for element in at.metric] == ["—", "—", "—"]


def test_the_detail_panel_words_reasons_from_the_shared_guidance_table() -> None:
    """§5.3/§6.3: the panel and the live banner read one table, so a reason
    code is worded once — and `test_guidance.py`'s full-coverage check keeps
    this panel from silently rendering nothing for a newly added code."""
    at = AppTest.from_string(
        "from face_symmetry_lab.application.analyzer import AnalysisOutcome\n"
        "from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status\n"
        "from face_symmetry_lab.ui.scores import render_guidance\n"
        "render_guidance(\n"
        "    AnalysisOutcome(\n"
        "        frame_id=1,\n"
        "        state=AnalysisState(\n"
        "            status=Status.HOLD,\n"
        "            reason=ReasonCode.TOO_FAR,\n"
        "            reason_codes=(ReasonCode.TOO_FAR, ReasonCode.OFF_AXIS_YAW),\n"
        "        ),\n"
        "        symmetry=None,\n"
        "        overall_smooth=None,\n"
        "        pose=None,\n"
        "        face_area_ratio=None,\n"
        "        metric_version='metric/test',\n"
        "        landmark_map_version='map/test',\n"
        "        sensitivity_frozen=False,\n"
        "    )\n"
        ")\n"
    )
    at.run()

    rendered = " ".join(element.value for element in at.markdown)
    assert not at.exception
    assert REASON_GUIDANCE[ReasonCode.TOO_FAR] in rendered
    assert REASON_GUIDANCE[ReasonCode.OFF_AXIS_YAW] in rendered
