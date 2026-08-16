"""カメラ画面の合成に対するテスト。

映像フレームの受け渡し、取り込み配線、§5.1 のレイアウト、そして HUD を
「どこから呼ぶか」という配線を検証する。HUD が browser へ送る資産そのもの
（CSS・実行時 JS・payload の中身）は `test_hud.py` が受け持つ。
"""

from fractions import Fraction

import av
import numpy as np
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.application.analyzer import AnalysisOutcome, FrameAnalyzer
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.models import AnalysisState, Status
from face_symmetry_lab.ui.camera import (
    FULLSCREEN_STATE_KEY,
    LOOPBACK_RTC_CONFIGURATION,
    MEDIA_STREAM_CONSTRAINTS,
    SYMMETRIZATION_STATE_KEY,
    build_video_frame_callback,
)


class StubLandmarker:
    """顔を 1 つ報告し、幾何が描画されるようにする（FR-013）。"""

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
    """すべてを背景として報告し、合成が画素を目に見えて変えるようにする。"""

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
    """FR-013: 描くのはランドマークと正中線だけである。"""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analyzer=StubAnalyzer())  # type: ignore[arg-type]
    callback = build_video_frame_callback(pipeline)
    source = _source_frame(value=0)

    result = callback(source)

    assert result is not source
    assert result.to_ndarray(format="rgb24").sum() > 0


def test_callback_returns_the_source_frame_when_there_is_nothing_to_draw() -> None:
    """顔が検出されなければ描く幾何も無いので、届いたフレームをそのまま返す。
    画素も変わらず、再エンコードも起きない。"""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(face_count=0),  # type: ignore[arg-type]
        analyzer=StubAnalyzer(),  # type: ignore[arg-type]
    )
    callback = build_video_frame_callback(pipeline)
    source = _source_frame()

    result = callback(source)

    assert result is source


def test_callback_still_feeds_every_frame_to_the_pipeline() -> None:
    """表示が素通しでも、解析まで素通しになってはならない。"""
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
    """画素を正当に書き換える唯一の経路でも pts/time_base は保たれる。"""
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
    assert result.to_ndarray(format="rgb24").max() == 0  # 完全に置き換わっている


def test_callback_never_raises_when_the_pipeline_fails() -> None:
    class BrokenPipeline:
        def process_with_analysis(self, rgb_image: np.ndarray) -> tuple[np.ndarray, None]:
            raise RuntimeError("pipeline failure")

    callback = build_video_frame_callback(BrokenPipeline())  # type: ignore[arg-type]

    result = callback(_source_frame(height=8, width=8))

    assert isinstance(result, av.VideoFrame)


def test_worker_thread_callback_never_touches_streamlit_state() -> None:
    """§14.3: WebRTC のコールバックは別スレッドで動くため `st.*` を呼べない。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.build_video_frame_callback)

    assert "session_state" not in source
    assert "st." not in source


# --- FR-034 取り込みの配線 ----------------------------------------------------


#: 対称な顔。実際の正中線で折り返せる程度に広がりを持たせてある。
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
    """版付きの対応表で実際に折り返せるランドマークを持つ顔を報告する。"""

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
    """FR-034: 取り込みの要求が、走っている解析を乱してはならない。"""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analysis_interval_ms=0)
    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()
    assert not at.exception

    at.button(key="capture_symmetrization").click().run()

    assert pipeline.capture_pending is True


def test_a_captured_frame_becomes_a_symmetrization_view() -> None:
    """§14.3g: 保持したフレームは一度だけ消費され、静止結果として描画される。"""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=SymmetrizableLandmarker(),
        analyzer=FrameAnalyzer(config=_symmetrizable_config(), config_3d=_symmetrizable_config()),
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
    assert pipeline.take_capture() is None  # ちょうど 1 回で消費される


def test_a_capture_that_cannot_be_symmetrized_is_reported_not_retried() -> None:
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(),  # 退化した 1 点だけでは折り返せない
        analyzer=FrameAnalyzer(config=_symmetrizable_config(), config_3d=_symmetrizable_config()),
        analysis_interval_ms=0,
    )
    pipeline.request_capture()
    pipeline.process(np.full((40, 40, 3), 30, dtype=np.uint8))

    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()

    assert not at.exception
    assert isinstance(at.session_state[SYMMETRIZATION_STATE_KEY], str)
    assert at.warning  # 黙って捨てず利用者へ提示する


def test_clearing_the_view_also_drops_the_held_frame() -> None:
    """§10.4: 利用者がパネルを消した後、顔由来の画像を一切残さない。"""
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
    """§5.3/§10.7: 対称化した顔は構成物であって規範ではない。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_symmetrization_panel)

    for forbidden in ("理想", "正解", "補正後", "あるべき", "完璧", "欠点"):
        assert forbidden not in source


def test_the_export_tab_explains_itself_before_the_first_record() -> None:
    """開始直後の「エクスポート」タブを真っ白にしない（FR-017 の提示条件は不変）。

    `render_export_panel` は記録が無いと節ごと何も描かないため、タブの中身が
    空白になっていた。**罠: `empty_notice` を渡さない実装は、利用者から見て
    「壊れている」と「まだ何も無い」を区別できないままにする。**
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_export_panel)

    assert "empty_notice=" in source


# --- §5.1 レイアウト: 映像を最上部に保つ --------------------------------------


def test_the_camera_section_puts_the_video_above_every_tab() -> None:
    """§5.1/AC-LAYOUT-01: 設定が表示を画面外へ押し出してはならない。

    `webrtc_streamer` は実ブラウザを必要とするため、合成そのものに対して
    検証する。streamer の呼び出しは `st.tabs` より前で、表示設定はそれより
    先に描画されてはならない。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    assert source.index("webrtc_streamer(") < source.index("st.tabs(")
    assert source.index("webrtc_streamer(") < source.index("_render_display_controls(")


def test_every_below_video_panel_lives_in_a_tab() -> None:
    """一度に 1 つだけ表示することで、映像が見えたままの高さに収める。"""
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


def test_render_camera_section_does_not_split_the_video_with_columns() -> None:
    """項目26（H3）: 「ライブ解析」パネルの配置プリセット（`_split_for_panel_layout`）
    が消え、映像は `st.columns` で分割せず画面幅いっぱいを使う。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)
    before_video = source[: source.index("webrtc_streamer(")]

    assert "st.columns(" not in before_video


# --- 項目26（H3）: 「ライブ解析」パネルの削除 ----------------------------------


def test_the_headline_panel_and_its_wiring_are_fully_removed() -> None:
    """パネル本体・配置プリセット・Camera 側の `render_tracking_status` /
    `guidance_message` の呼び出しを跡形もなく消す（requirement.md §14.3e、
    項目26、H3）。HUD が同じ内容を表示するようになったため、これらが残って
    いれば同じ測定値が画面上の 2 か所へ出てしまう。"""
    import inspect

    from face_symmetry_lab.ui import camera

    for name in (
        "_render_headline_panel",
        "_render_headline_fragment",
        "_split_for_panel_layout",
        "PANEL_LAYOUT_STATE_KEY",
    ):
        assert not hasattr(camera, name), name

    source = inspect.getsource(camera)
    for symbol in ("render_headline_metrics", "render_tracking_status", "guidance_message"):
        assert symbol not in source, symbol


# --- §14.3e HUD の配線 --------------------------------------------------------


def test_the_hud_is_published_from_a_single_existing_fragment() -> None:
    """3 つ目の 5 Hz fragment は純粋な負荷だった。payload は既存 fragment へ
    相乗りさせ、増やさない——モジュール全体で `@st.fragment` はちょうど 2 個
    （HUD publish 用と詳細用）のままである。"""
    import inspect

    from face_symmetry_lab.ui import camera

    assert "render_hud_payload(" in inspect.getsource(camera._render_hud_fragment)
    assert inspect.getsource(camera).count("@st.fragment(") == 2


def test_the_hud_fragment_only_publishes_and_never_renders_a_panel() -> None:
    """項目26（H3）: この fragment は payload の publish だけを行い、見出しパネル
    が描いていたもの（`st.metric`・`render_headline_metrics`・枠付き container）
    を一切持たない。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_hud_fragment)

    assert "st.metric" not in source
    assert "render_headline_metrics" not in source
    assert "st.container(border=True)" not in source


def test_the_bridge_is_installed_once_and_never_inside_a_fragment() -> None:
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    assert "render_hud_bridge(" in source
    assert "render_hud_bridge" not in inspect.getsource(camera._render_hud_fragment)


def test_the_hud_position_control_renders_a_single_radio() -> None:
    """項目26（H3）: 「ライブ解析」パネルの配置プリセットが消えたので、この
    コントロールが描く radio は映像上 HUD の位置だけになる。"""
    from face_symmetry_lab.ui.hud import HUD_POSITION_STATE_KEY

    at = AppTest.from_string(
        "from face_symmetry_lab.ui.camera import _render_hud_position_control\n"
        "_render_hud_position_control()\n"
    )
    at.run()

    assert not at.exception
    assert len(at.radio) == 1
    assert at.radio[0].key == HUD_POSITION_STATE_KEY


def test_the_detail_fragment_suppresses_the_duplicated_part_breakdown() -> None:
    """項目26（H3）: HUD が既に部位別 4 つ・全体の対称度・全体のスコアを表示
    するため、詳細タブでは `show_parts=False` を渡して重複を避ける。

    罠: 既定のまま（`show_parts` を渡さず）呼ぶ実装ではこの assertion が落ちる。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_detail_fragment)

    assert (
        "render_score_details(snapshot.analysis, show_smoothed=False, show_parts=False)" in source
    )


def test_the_frame_path_is_untouched_by_the_overlay_work() -> None:
    """オーバーレイは DOM の層である。コールバックは Streamlit に触れないまま保つ。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.build_video_frame_callback)

    assert "st." not in source
    assert "hud" not in source.lower()
    assert "fullscreen" not in source.lower()


# --- §14.3e 全画面 ------------------------------------------------------------


def test_the_exit_control_clears_the_fullscreen_toggle() -> None:
    """これが無いとタブへ到達できず、利用者が戻れなくなる。"""
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
    """全画面の映像の背後で 5 Hz の fragment を再描画する意味は無い。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    assert source.index("if fullscreen:") < source.index("st.tabs(")


def test_the_exit_control_is_labelled_as_an_exit() -> None:
    import inspect

    from face_symmetry_lab.ui import camera

    assert "✕ 全画面を終了" in inspect.getsource(camera.render_camera_section)


# --- 映像コンポーネントの属性 -------------------------------------------------


def test_the_native_video_controls_are_switched_off_through_the_component_api() -> None:
    """ネイティブの全画面は <video> をブラウザの top layer へ昇格させ、HUD を
    含むあらゆるページ要素の上へ出すため、オーバーレイを無効化してしまう。
    `controls: False` がバーごと取り除き、各属性は隠したバーでも残る経路
    （コンテキストメニュー、リモート再生）を塞ぐ。"""
    from face_symmetry_lab.ui.camera import VIDEO_HTML_ATTRS

    assert VIDEO_HTML_ATTRS["controls"] is False
    assert VIDEO_HTML_ATTRS["disablePictureInPicture"] is True
    assert "nofullscreen" in VIDEO_HTML_ATTRS["controlsList"]


def test_the_video_attributes_reach_the_component() -> None:
    import inspect

    from face_symmetry_lab.ui import camera

    assert "video_html_attrs=VIDEO_HTML_ATTRS" in inspect.getsource(camera.render_camera_section)


# --- FR-040: 表示平滑化時間の配線 ---------------------------------------------


def test_the_display_controls_offer_the_smoothing_selector_and_apply_it() -> None:
    """FR-040: 選んだ時定数が、その場で pipeline へ届く。

    罠: `render_smoothing_control` を描くだけで `set_smoothing_tau_seconds` を
    呼ばない実装は、UI だけが動いて平滑化が既定のまま残る。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_display_controls)

    assert "render_smoothing_control(" in source
    assert "pipeline.set_smoothing_tau_seconds(" in source
    assert "pipeline.analysis_interval_ms" in source, "実効サンプル数は実周期から出す"


def test_the_hud_fragment_publishes_the_active_smoothing_window() -> None:
    """FR-040/§5.4: 全画面中でも「何秒の平均か」を映像の上で読めるようにする。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_hud_fragment)

    assert "smoothing_tau_seconds=pipeline.smoothing_tau_seconds" in source


def test_the_settings_tab_groups_its_controls_into_titled_sections() -> None:
    """§5.1: 設定が平らな 1 列に並ぶと、どれが何のための操作か読み取れない。"""
    import inspect

    from face_symmetry_lab.ui import camera
    from face_symmetry_lab.ui.smoothing_panel import SMOOTHING_HEADING

    source = inspect.getsource(camera._render_display_controls)

    assert source.count("st.container(border=True)") >= 3
    for heading in ("映像の見え方", "自動調整の現在値"):
        assert heading in source
    # 平滑化区画の見出しは panel 自身が持つ。**ここで重ねて出さない**（罠: 両方が
    # 出すと同じ囲みの中に見出しが 2 行並ぶ。実ブラウザで見つけた回帰）。
    assert 'st.markdown("##### 数値の落ち着き' not in source
    assert SMOOTHING_HEADING.startswith("##### 数値の落ち着き")
    # HUD の位置は別の関数が持つが、同じ作法で囲う。
    assert "映像上のHUD" in inspect.getsource(camera._render_hud_position_control)


def test_the_tabs_keep_their_names_while_gaining_icons() -> None:
    """アイコンは見分けの手掛かりであって、文言の置き換えではない（§5.4）。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    for label in ("詳細結果", "左右対称との比較", "表示設定", "エクスポート"):
        assert label in source
