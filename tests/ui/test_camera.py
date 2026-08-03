import json
from fractions import Fraction

import av
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.application.analyzer import AnalysisOutcome, FrameAnalyzer
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.models import AnalysisState, GenderLabel, Status
from face_symmetry_lab.ui.camera import (
    _HUD_PLACEMENTS,
    _HUD_POSITION_OPTIONS,
    FULLSCREEN_STATE_KEY,
    LOOPBACK_RTC_CONFIGURATION,
    MEDIA_STREAM_CONSTRAINTS,
    PANEL_LAYOUT_STATE_KEY,
    SYMMETRIZATION_STATE_KEY,
    _hud_html,
    _hud_payload,
    build_video_frame_callback,
)
from face_symmetry_lab.ui.scores import GENDER_LABELS, format_symmetry


class StubLandmarker:
    """Reports one face, so geometry gets drawn (FR-013)."""

    def __init__(self, face_count: int = 1) -> None:
        self.sizes: list[tuple[int, int]] = []
        self.face_count = face_count

    def detect_video(
        self,
        rgb_image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.sizes.append((rgb_image.shape[1], rgb_image.shape[0]))
        points = ((0.5, 0.5),) if self.face_count == 1 else ()
        return build_landmark_frame(frame_id=frame_id, face_count=self.face_count, points=points)

    def close(self) -> None:
        return None


class StubAnalyzer:
    def analyze(
        self,
        frame: LandmarkFrame,
        *,
        at: float,
        rgb_image: np.ndarray | None = None,
    ) -> AnalysisOutcome:
        return AnalysisOutcome(
            frame_id=frame.frame_id,
            state=AnalysisState(status=Status.HOLD),
            symmetry=None,
            overall_distance_smooth=None,
            pose=None,
            face_area_ratio=None,
            metric_version="metric/test",
            landmark_map_version="map/test",
        )

    def reset(self) -> None:
        return None


class BlackoutSegmenter:
    """Reports everything as background, so compositing visibly changes pixels."""

    def segment(self, rgb_image: np.ndarray, *, timestamp_ms: int) -> np.ndarray:
        height, width = rgb_image.shape[:2]
        return np.zeros((height, width), dtype=np.float32)

    def close(self) -> None:
        return None


def _source_frame(height: int = 48, width: int = 64, value: int = 200) -> av.VideoFrame:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
    frame.pts = 1
    frame.time_base = Fraction(1, 30)
    return frame


def test_webrtc_configuration_uses_no_external_ice_servers() -> None:
    assert LOOPBACK_RTC_CONFIGURATION == {"iceServers": []}


def test_media_constraints_request_video_only() -> None:
    assert MEDIA_STREAM_CONSTRAINTS["audio"] is False
    assert MEDIA_STREAM_CONSTRAINTS["video"] is not False


def test_callback_draws_geometry_onto_the_returned_frame() -> None:
    """FR-013: landmarks and the midline are the only things drawn."""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analyzer=StubAnalyzer())  # type: ignore[arg-type]
    callback = build_video_frame_callback(pipeline)
    source = _source_frame(value=0)

    result = callback(source)

    assert result is not source
    assert result.to_ndarray(format="rgb24").sum() > 0


def test_callback_returns_the_source_frame_when_there_is_nothing_to_draw() -> None:
    """With no face detected there is no geometry, so the frame that arrived
    is handed straight back — no pixel change and no re-encode."""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(face_count=0),  # type: ignore[arg-type]
        analyzer=StubAnalyzer(),  # type: ignore[arg-type]
    )
    callback = build_video_frame_callback(pipeline)
    source = _source_frame()

    result = callback(source)

    assert result is source


def test_callback_still_feeds_every_frame_to_the_pipeline() -> None:
    """Pass-through display must not mean pass-through analysis."""
    landmarker = StubLandmarker()
    callback = build_video_frame_callback(RealtimeLandmarkPipeline(landmarker=landmarker))

    callback(_source_frame())

    assert landmarker.sizes == [(64, 48)]


def test_analysis_still_reaches_the_snapshot_the_side_panel_reads() -> None:
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analyzer=StubAnalyzer())  # type: ignore[arg-type]
    callback = build_video_frame_callback(pipeline)

    callback(_source_frame())

    snapshot = pipeline.snapshot()
    assert snapshot.analysis is not None
    assert snapshot.frame_id == 1


def test_background_replacement_still_produces_a_new_frame_with_timing_intact() -> None:
    """The one case that legitimately rewrites pixels keeps pts/time_base."""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(face_count=0),  # type: ignore[arg-type]
        segmenter=BlackoutSegmenter(),  # type: ignore[arg-type]
        background_mode=BackgroundMode.BLACK,
    )
    callback = build_video_frame_callback(pipeline)
    source = _source_frame(value=200)

    result = callback(source)

    assert result is not source
    assert (result.width, result.height) == (64, 48)
    assert result.pts == source.pts
    assert result.time_base == source.time_base
    assert result.to_ndarray(format="rgb24").max() == 0  # fully replaced


def test_callback_never_raises_when_the_pipeline_fails() -> None:
    class BrokenPipeline:
        def process_with_analysis(self, rgb_image: np.ndarray) -> tuple[np.ndarray, None]:
            raise RuntimeError("pipeline failure")

    callback = build_video_frame_callback(BrokenPipeline())  # type: ignore[arg-type]

    result = callback(_source_frame(height=8, width=8))

    assert isinstance(result, av.VideoFrame)


def test_worker_thread_callback_never_touches_streamlit_state() -> None:
    """§14.3: the WebRTC callback runs off-thread and must not call `st.*`."""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.build_video_frame_callback)

    assert "session_state" not in source
    assert "st." not in source


# --- FR-034 capture wiring ---------------------------------------------------


#: A symmetric face, spread out enough to fold about a real midline.
_SYMMETRIZABLE_POINTS = (
    (0.35, 0.40),
    (0.45, 0.40),
    (0.55, 0.40),
    (0.65, 0.40),
    (0.50, 0.30),
    (0.50, 0.50),
    (0.50, 0.62),
    (0.30, 0.55),
    (0.70, 0.55),
    (0.32, 0.46),
    (0.68, 0.46),
    (0.46, 0.52),
    (0.54, 0.52),
    (0.44, 0.58),
    (0.56, 0.58),
)


class SymmetrizableLandmarker(StubLandmarker):
    """Reports a face whose landmarks the versioned map can actually fold."""

    def detect_video(
        self,
        rgb_image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.sizes.append((rgb_image.shape[1], rgb_image.shape[0]))
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=1,
            points=_SYMMETRIZABLE_POINTS,
            width=rgb_image.shape[1],
            height=rgb_image.shape[0],
        )


def _symmetrizable_config():
    return build_metric_config(
        metric_version="symmetry-2d/test",
        landmark_map_version="test/1.0.0",
        landmark_count=len(_SYMMETRIZABLE_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8), (9, 10)), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
            "nose": PartMap(pairs=((11, 12),), min_pairs=1),
            "mouth": PartMap(pairs=((13, 14),), min_pairs=1),
        },
    )


def _panel_script() -> str:
    return (
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.camera import _render_symmetrization_panel\n"
        "_render_symmetrization_panel(st.session_state['pipeline'])\n"
    )


def test_the_capture_button_arms_the_pipeline_without_stopping_it() -> None:
    """FR-034: requesting a capture must not disturb the running analysis."""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analysis_interval_ms=0)
    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()
    assert not at.exception

    at.button(key="capture_symmetrization").click().run()

    assert pipeline.capture_pending is True


def test_a_captured_frame_becomes_a_symmetrization_view() -> None:
    """§14.3g: the held frame is consumed once and rendered as a still result."""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=SymmetrizableLandmarker(),
        analyzer=FrameAnalyzer(config=_symmetrizable_config()),
        analysis_interval_ms=0,
    )
    pipeline.request_capture()
    pipeline.process(np.full((80, 80, 3), 30, dtype=np.uint8))

    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()

    assert not at.exception
    view = at.session_state[SYMMETRIZATION_STATE_KEY]
    assert not isinstance(view, str)
    assert view.symmetrized_point_count > 0
    assert pipeline.take_capture() is None  # consumed exactly once


def test_a_capture_that_cannot_be_symmetrized_is_reported_not_retried() -> None:
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(),  # a single degenerate point cannot fold
        analyzer=FrameAnalyzer(config=_symmetrizable_config()),
        analysis_interval_ms=0,
    )
    pipeline.request_capture()
    pipeline.process(np.full((40, 40, 3), 30, dtype=np.uint8))

    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()

    assert not at.exception
    assert isinstance(at.session_state[SYMMETRIZATION_STATE_KEY], str)
    assert at.warning  # surfaced to the user rather than silently dropped


def test_clearing_the_view_also_drops_the_held_frame() -> None:
    """§10.4: no face-derived image is kept after the user clears the panel."""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analysis_interval_ms=0)
    pipeline.request_capture()

    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()
    at.button(key="clear_symmetrization").click().run()

    assert pipeline.capture_pending is False
    assert pipeline.take_capture() is None
    assert SYMMETRIZATION_STATE_KEY not in at.session_state


def test_the_symmetrization_panel_avoids_value_laden_wording() -> None:
    """§5.3/§10.7: a symmetrized face is a construction, not an ideal."""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_symmetrization_panel)

    for forbidden in ("理想", "正解", "補正後", "あるべき", "完璧", "欠点"):
        assert forbidden not in source


# --- §5.1 layout: the video stays at the top ---------------------------------


def test_the_layout_split_reads_the_stored_preset_without_rendering_a_widget() -> None:
    """§5.1: the video is placed before the control that moves its panel.

    The split has to be decided at the top of the section, while the radio
    that changes it lives in the settings tab below — so this reads the
    preference rather than drawing the widget.
    """
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.camera import _split_for_panel_layout\n"
        "preview, headline = _split_for_panel_layout()\n"
        "st.session_state['stacked'] = headline is None\n"
    )
    at.session_state[PANEL_LAYOUT_STATE_KEY] = "動画の下"
    at.run()

    assert not at.exception
    assert at.session_state["stacked"] is True
    assert not at.radio  # the preset widget belongs to the settings tab


def test_the_default_layout_places_the_headline_beside_the_video() -> None:
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.camera import _split_for_panel_layout\n"
        "preview, headline = _split_for_panel_layout()\n"
        "st.session_state['stacked'] = headline is None\n"
    )
    at.run()

    assert not at.exception
    assert at.session_state["stacked"] is False


def test_the_panel_layout_control_writes_the_key_the_split_reads() -> None:
    """A change in the settings tab has to reach the split on the next run."""
    at = AppTest.from_string(
        "from face_symmetry_lab.ui.camera import _render_panel_layout_control\n"
        "_render_panel_layout_control()\n"
    )
    at.run()

    assert not at.exception
    assert at.radio[0].key == PANEL_LAYOUT_STATE_KEY

    at.radio[0].set_value("動画の左").run()

    assert at.session_state[PANEL_LAYOUT_STATE_KEY] == "動画の左"


def test_the_camera_section_puts_the_video_above_every_tab() -> None:
    """§5.1/AC-LAYOUT-01: settings must not push the preview off screen.

    Asserted on the composition itself, since `webrtc_streamer` needs a live
    browser: the streamer call has to precede `st.tabs`, and the display
    controls must not be rendered before it.
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    assert source.index("webrtc_streamer(") < source.index("st.tabs(")
    assert source.index("_split_for_panel_layout()") < source.index("webrtc_streamer(")
    assert source.index("webrtc_streamer(") < source.index("_render_display_controls(")


def test_every_below_video_panel_lives_in_a_tab() -> None:
    """One panel at a time keeps the page short enough to keep the video visible."""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)
    tabs_at = source.index("st.tabs(")

    for panel in (
        "_render_detail_panel(",
        "_render_symmetrization_panel(",
        "_render_export_panel(",
        "_render_session_controls(",
    ):
        assert source.index(panel) > tabs_at, panel


# --- §14.3e HUD --------------------------------------------------------------


def _estimate(age: float = 34.0, female: float = 0.8):
    from datetime import UTC, datetime

    from face_symmetry_lab.domain.demographics import build_demographic_estimate

    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(female, 1.0 - female),
        model_version="age-gender/test",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )


def _payload(**overrides):
    settings = {
        "overall": 0.0215,
        "estimate": None,
        "show_demographics": False,
        "gate": "#2E9E5B",
        "position": "左上",
    }
    settings.update(overrides)
    return _hud_payload(**settings)


def test_the_hud_reuses_the_shared_symmetry_formatter() -> None:
    """No second way of rendering the same measurement (§14.3e)."""
    assert _payload()["symmetry"] == format_symmetry(0.0215)


def test_a_missing_symmetry_value_renders_as_a_dash() -> None:
    """NFR-013: absent is absent, never a stale or zeroed number."""
    assert _payload(overall=None)["symmetry"] == "—"
    assert "—" in _hud_html(**{**_payload_kwargs(), "overall": None})


def test_the_hud_carries_the_experimental_label() -> None:
    assert _payload()["label"] == "対称度（実験値）"


def test_the_hud_avoids_any_value_judgement() -> None:
    """§5.3: reorienting the number must not turn the HUD into a verdict."""
    rendered = repr(_payload(estimate=_estimate(), show_demographics=True))

    for forbidden in ("良い", "悪い", "改善", "スコアが高い", "優秀", "理想"):
        assert forbidden not in rendered


def test_the_hud_shows_age_and_gender_through_the_shared_labels() -> None:
    demo = _payload(estimate=_estimate(age=34.4), show_demographics=True)["demo"]

    assert "参考年齢 34 歳" in demo
    assert GENDER_LABELS[GenderLabel.FEMALE] in demo


def test_the_demographic_row_is_dropped_rather_than_dashed_when_absent() -> None:
    """FR-024: an empty row would imply the feature is running and failing."""
    payload = _payload(estimate=None, show_demographics=True)

    assert payload["demo"] == ""
    assert payload["symmetry"] == format_symmetry(0.0215)


def test_the_demographic_row_is_dropped_when_the_user_turned_it_off() -> None:
    assert _payload(estimate=_estimate(), show_demographics=False)["demo"] == ""


def test_the_hud_carries_the_gate_colour() -> None:
    assert _payload(gate="#D9A21B")["gate"] == "#D9A21B"


@pytest.mark.parametrize("position", ["左上", "右上", "左下", "右下"])
def test_every_visible_preset_marks_the_hud_visible(position: str) -> None:
    assert _payload(position=position)["visible"] is True


def test_the_hidden_preset_marks_the_hud_invisible() -> None:
    assert _payload(position="非表示")["visible"] is False


def test_an_unknown_preset_hides_rather_than_floating_loose() -> None:
    assert _payload(position="斜め上")["visible"] is False


def test_every_hud_preset_is_accepted_by_the_payload() -> None:
    for position in _HUD_POSITION_OPTIONS:
        assert isinstance(_hud_payload(**{**_payload_kwargs(), "position": position}), dict)


def test_the_hud_preset_list_offers_a_hidden_option_and_defaults_to_top_left() -> None:
    assert _HUD_POSITION_OPTIONS[0] == "左上"
    assert _HUD_POSITION_OPTIONS[-1] == "非表示"
    assert set(_HUD_PLACEMENTS) == set(_HUD_POSITION_OPTIONS) - {"非表示"}


# --- §14.3e: the 5 Hz payload stays cheap ------------------------------------


def test_the_payload_element_is_hidden_and_carries_only_json() -> None:
    """This is the only thing crossing the wire at 5 Hz."""
    html = _hud_html(**_payload_kwargs())

    assert "display:none" in html
    assert html.count("<div") == 1
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["symmetry"] == format_symmetry(0.0215)


def test_the_five_hertz_update_never_ships_css() -> None:
    """Re-sending a stylesheet at 5 Hz forced a whole-document style recalc,
    which is what made the video stutter. It must not come back."""
    html = _hud_html(**_payload_kwargs())

    assert "<style" not in html
    assert "backdrop-filter" not in html
    assert "transition" not in html


def test_the_hud_never_uses_backdrop_filter_anywhere() -> None:
    """Blurring the backdrop re-runs on every *video* frame, not every update."""
    from face_symmetry_lab.ui import camera

    assert "backdrop-filter" not in camera._HUD_STYLESHEET
    assert "backdrop-filter" not in camera._hud_bridge_script(fullscreen=False, position="左上")


def test_the_hud_is_published_from_the_existing_headline_fragment() -> None:
    """A third 5 Hz fragment was pure overhead; the payload rides along."""
    import inspect

    from face_symmetry_lab.ui import camera

    assert not hasattr(camera, "_render_hud_fragment")
    assert "_render_hud_payload" in inspect.getsource(camera._render_headline_fragment)


# --- §14.3e: the bridge ------------------------------------------------------


def _bridge(**overrides) -> str:
    settings = {"fullscreen": False, "position": "左上"}
    settings.update(overrides)
    from face_symmetry_lab.ui import camera

    return camera._hud_bridge_script(**settings)


def test_the_hud_shell_stays_click_through_except_for_its_grip() -> None:
    """It sits over the video's own STOP control."""
    from face_symmetry_lab.ui import camera

    css = camera._HUD_STYLESHEET

    assert "#fsl-hud {" in css
    assert "pointer-events: none;" in css
    # The grip is the one interactive part, so the HUD can be dragged.
    assert "#fsl-hud-grip" in css
    assert "pointer-events: auto;" in css


def test_the_hud_keeps_digit_widths_stable_at_five_hertz() -> None:
    from face_symmetry_lab.ui import camera

    assert "tabular-nums" in camera._HUD_STYLESHEET


def test_the_bridge_makes_the_hud_draggable() -> None:
    script = _bridge()

    assert "pointerdown" in script
    assert "pointermove" in script
    assert "setPointerCapture" in script


def test_the_bridge_attaches_the_hud_to_the_page_body() -> None:
    """On <body> nothing can clip it and `position: fixed` behaves — which is
    why the HUD used to vanish in fullscreen."""
    from face_symmetry_lab.ui import camera

    assert "document.body.appendChild(hud)" in camera._HUD_RUNTIME_JS


def test_fullscreen_targets_the_video_through_a_stable_hook() -> None:
    """The old `data-testid` selector matched nothing, so only the HUD moved."""
    from face_symmetry_lab.ui import camera

    assert ".st-key-camera_preview_frame iframe" in camera._HUD_STYLESHEET
    assert "data-testid" not in camera._HUD_STYLESHEET


def test_the_bridge_toggles_fullscreen_by_class_rather_than_reinjecting_css() -> None:
    assert "classList.toggle('fsl-fullscreen', !!CFG.fullscreen)" in _bridge()
    assert "fsl-fullscreen" in _bridge(fullscreen=True)


def test_the_bridge_replaces_its_timer_instead_of_stacking_them() -> None:
    """The script re-runs on every full rerun; timers must not accumulate."""
    script = _bridge()

    assert "clearInterval" in script
    assert script.count("setInterval") == 1


def test_the_bridge_is_installed_once_and_never_inside_a_fragment() -> None:
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    assert "_render_hud_bridge(" in source
    assert "_render_hud_bridge" not in inspect.getsource(camera._render_headline_fragment)


def test_the_bridge_creates_the_hud_only_once() -> None:
    script = _bridge()

    assert "if (!hud)" in script
    assert "getElementById('fsl-hud')" in script


# --- §14.3e fullscreen -------------------------------------------------------


def test_the_frame_path_is_untouched_by_the_overlay_work() -> None:
    """The overlay is a DOM layer: the callback must stay Streamlit-free."""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.build_video_frame_callback)

    assert "st." not in source
    assert "hud" not in source.lower()
    assert "fullscreen" not in source.lower()


def test_no_streamlit_internal_testid_is_relied_on_any_more() -> None:
    """The `data-testid` hook silently broke fullscreen; nothing selects on it.

    Checked against the CSS and the bridge script — the strings the browser
    actually evaluates — so the comment explaining why it was dropped does not
    trip this.
    """
    from face_symmetry_lab.ui import camera

    assert "data-testid" not in camera._HUD_STYLESHEET
    assert "data-testid" not in camera._hud_bridge_script(fullscreen=True, position="左上")


def test_the_overlay_css_avoids_structural_selectors() -> None:
    """§5.2's exception requires stable hooks, not `nth-child` positions."""
    from face_symmetry_lab.ui import camera

    assert "nth-child" not in camera._HUD_STYLESHEET
    assert "nth-of-type" not in camera._HUD_STYLESHEET


def test_the_exit_control_clears_the_fullscreen_toggle() -> None:
    """Without this the tabs are unreachable and the user is stuck."""
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.camera import FULLSCREEN_STATE_KEY, _exit_fullscreen\n"
        "st.session_state.setdefault(FULLSCREEN_STATE_KEY, True)\n"
        "st.button('全画面を終了', key='exit', on_click=_exit_fullscreen)\n"
    )
    at.run()
    assert at.session_state[FULLSCREEN_STATE_KEY] is True

    at.button(key="exit").click().run()

    assert at.session_state[FULLSCREEN_STATE_KEY] is False


def test_fullscreen_skips_the_tab_block_entirely() -> None:
    """No point repainting 5 Hz fragments behind a full-viewport video."""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    assert source.index("if fullscreen:") < source.index("st.tabs(")


def _payload_kwargs() -> dict:
    return {
        "overall": 0.0215,
        "estimate": None,
        "show_demographics": False,
        "gate": "#2E9E5B",
        "position": "左上",
    }


def test_the_runtime_is_injected_into_the_page_realm_not_the_component_iframe() -> None:
    """A timer defined in the component iframe dies when Streamlit replaces it,
    leaving the HUD frozen while its timer id survives on the parent."""
    script = _bridge()

    assert "createElement('script')" in script
    assert "fsl-hud-runtime" in script
    assert "doc.head.appendChild(runtime)" in script


def test_the_runtime_owns_the_timer_and_the_element() -> None:
    from face_symmetry_lab.ui import camera

    runtime = camera._HUD_RUNTIME_JS

    assert "setInterval(sync" in runtime
    assert "document.body.appendChild(hud)" in runtime
    assert "clearInterval(window.__fslHudTimer)" in runtime


def test_fullscreen_stacks_above_streamlit_chrome() -> None:
    """Streamlit's sidebar/header sit at z-index 999990+, so anything lower is
    covered — which hid a third of the video and the whole HUD."""
    from face_symmetry_lab.ui import camera

    css = camera._HUD_STYLESHEET

    assert "body.fsl-fullscreen .stSidebar" in css
    assert "body.fsl-fullscreen .stAppHeader" in css
    assert "display: none !important;" in css
    # video < HUD < exit control, all above Streamlit's own chrome.
    for value in (1000000, 1000050, 1000060):
        assert f"z-index: {value}" in css


# --- fullscreen chrome: controls, native buttons, HUD lifetime ---------------


def test_the_exit_control_shrinks_to_its_button() -> None:
    """Streamlit's vertical block is width:100%, so a `right`-anchored fixed
    container stretched across the viewport and dropped the button off the
    left edge (measured at x=-24)."""
    from face_symmetry_lab.ui import camera

    css = camera._HUD_STYLESHEET

    assert "width: auto !important;" in css
    assert "left: auto;" in css


def test_the_native_video_fullscreen_control_is_removed() -> None:
    """Native fullscreen promotes the <video> into the browser's top layer,
    above every page element — the HUD included — so it defeats the overlay."""
    from face_symmetry_lab.ui import camera

    css = camera._VIDEO_CHROME_CSS

    for control in (
        "media-controls-fullscreen-button",
        "media-controls-overflow-button",
        "media-controls-overflow-menu-button",
        "media-controls-picture-in-picture-button",
    ):
        assert control in css
    assert "display: none !important;" in css


def test_the_runtime_also_disables_the_controls_by_attribute() -> None:
    """CSS hides the buttons; the attributes stop the behaviour being reached
    another way (keyboard shortcut, context menu)."""
    from face_symmetry_lab.ui import camera

    runtime = camera._HUD_RUNTIME_JS

    assert "nofullscreen" in runtime
    assert "disablePictureInPicture" in runtime


def test_fullscreen_centres_and_enlarges_the_start_stop_row() -> None:
    """At the far-left edge of a fullscreen viewport the row was hard to hit."""
    from face_symmetry_lab.ui import camera

    css = camera._VIDEO_CHROME_CSS

    assert "html.fsl-fs .fsl-controls" in css
    assert "left: 50%" in css
    assert "translateX(-50%)" in css
    assert "min-height: 48px" in css


def test_the_control_row_is_tagged_from_js_rather_than_by_a_vendor_class() -> None:
    """The component ships only emotion/MUI hashes, so the app adds its own."""
    from face_symmetry_lab.ui import camera

    assert "classList.add('fsl-controls')" in camera._HUD_RUNTIME_JS


def test_the_hud_disappears_when_the_camera_is_not_running() -> None:
    """The panel that publishes the payload only renders while playing, so a
    missing payload must hide the HUD instead of freezing its last values."""
    from face_symmetry_lab.ui import camera

    assert "if (!holder) { hudEl.hidden = true; return; }" in camera._HUD_RUNTIME_JS


def test_the_component_iframe_access_is_guarded() -> None:
    """A cross-origin or not-yet-ready iframe must not throw on every tick."""
    from face_symmetry_lab.ui import camera

    runtime = camera._HUD_RUNTIME_JS

    assert "try { vdoc = host && host.contentDocument; } catch (err) { return; }" in runtime
