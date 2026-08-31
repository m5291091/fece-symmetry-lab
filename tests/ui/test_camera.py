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
    build_video_frame_callback,
)
from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result


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


def test_the_camera_has_no_symmetrisation_path_at_all() -> None:
    """§0 項目 33（H10, 2026-08-29）: Camera から対称化と一発取り込みを削除した。

    罠: UI だけ消して pipeline の capture API を残す実装は、リアルタイム経路が
    生フレームを 1 枚保持する仕組みを温存する（§10.4）。
    """
    import inspect

    from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
    from face_symmetry_lab.ui import camera

    for gone in (
        "_render_symmetrization_panel",
        "_render_stopped_symmetrization_panel",
        "_consume_pending_capture",
        "_clear_symmetrization",
        "SYMMETRIZATION_STATE_KEY",
        "SYMMETRIZATION_CAPTION",
    ):
        assert not hasattr(camera, gone), gone
    for gone in ("request_capture", "take_capture", "capture_pending", "clear_capture"):
        assert not hasattr(RealtimeLandmarkPipeline, gone), gone
    assert "symmetry=" not in inspect.getsource(camera.render_camera_section)


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

    assert source.index("webrtc_streamer(") < source.index("render_result_tabs(")
    assert source.index("webrtc_streamer(") < source.index("_render_display_controls(")


def test_every_below_video_panel_lives_in_a_tab() -> None:
    """一度に 1 つだけ表示することで、映像が見えたままの高さに収める。

    2026-08-29: 表示設定（`_render_display_controls`）とセッション操作
    （`_render_session_controls`）はタブではなく映像直下のポップオーバーへ移った
    ため、この 2 つは「タブの closure より後」ではなく「映像より後・タブより
    前」であることを別に確かめる（項目 29）。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)
    # タブの中身は closure として組み立てられ、`render_result_tabs` だけが
    # それを開く。どの panel も、その closure の組み立てが始まってから先にしか
    # 現れない——映像の直後に直接描画する panel が残っていればここで落ちる。
    closures_at = source.index("def _detail_tab(")

    for panel in (
        "_render_detail_panel(",
        "_render_demographic_fragment(",
        "_render_export_panel(",
        "_render_stopped_detail_panel(",
        "_render_stopped_demographic_panel(",
    ):
        assert panel in source, panel
        assert source.index(panel) > closures_at, panel
    assert source.count("render_result_tabs(") == 1

    # 表示設定とセッション操作は映像より後・タブの closure より前で呼ばれる。
    video_at = source.index("webrtc_streamer(")
    display_controls_at = source.index("_render_display_controls(")
    assert video_at < display_controls_at < closures_at
    # `_render_session_controls` はポップオーバーの中身（`_render_display_controls`
    # 自身）が呼ぶので、`render_camera_section` の source には現れない。
    assert "_render_session_controls(" not in source
    assert "_render_session_controls(pipeline)" in inspect.getsource(
        camera._render_display_controls
    )


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
    """HUD の payload は既存 fragment へ相乗りさせ、5 Hz の口を増やさない。

    モジュール全体の `@st.fragment` はちょうど 3 個である——HUD publish 用（5 Hz）と、
    詳細用（1 Hz、2026-08-31 の利用者報告対応）と、参考年齢・参考性別用（1 Hz）。
    詳細 fragment を 5 Hz でフル再描画するとブラウザのメインスレッドで WebRTC
    映像と競合し、長いセッションで映像が次第に重くなり停止する（利用者報告）。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    assert "render_hud_payload(" in inspect.getsource(camera._render_hud_fragment)
    assert inspect.getsource(camera).count("@st.fragment(") == 3
    # 回帰: HUD は 5 Hz のまま、詳細タブは 1 Hz に抑える。
    assert "run_every=STATUS_REFRESH_INTERVAL" in inspect.getsource(camera._render_hud_fragment)
    assert "run_every=DETAIL_REFRESH_INTERVAL" in inspect.getsource(camera._render_detail_fragment)


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

    # 枠は `ui.detail_panel` と共有するが、部位別の可否は入力ごとに違う
    # （AC-LAYOUT-01）。共有化のついでに握りつぶしていないことを固定する。
    assert "render_detail_panel(" in source
    assert "show_parts=False" in source


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

    assert source.index("if fullscreen:") < source.index("render_result_tabs(")


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
    assert "pipeline.set_display_hold_seconds(" in source
    assert "pipeline.analysis_interval_ms" in source, "実効サンプル数は実周期から出す"


def test_the_hud_fragment_publishes_the_active_smoothing_window() -> None:
    """FR-040/§5.4: 全画面中でも「何秒の平均か」を映像の上で読めるようにする。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_hud_fragment)

    assert "smoothing_tau_seconds=pipeline.smoothing_tau_seconds" in source
    assert "display_hold_seconds=pipeline.display_hold_seconds" in source


def test_the_settings_live_in_a_popover_below_the_video() -> None:
    """§0 項目 29（2026-08-29）: 表示設定はタブではなく、映像直下に畳んで置く
    ポップオーバー 1 つへ集約する。

    罠: ポップオーバーを `playing` の分岐の中や early return の後へ置く実装は、
    停止中・全画面中に設定 widget が描画されず、選んだ値が黙って既定へ戻る。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    assert source.index("webrtc_streamer(") < source.index("st.popover(")
    assert source.index("st.popover(") < source.index("playing = context.state.playing")

    popover_line = next(line for line in source.splitlines() if "st.popover(" in line)
    assert "on_change" not in popover_line

    controls_source = inspect.getsource(camera._render_display_controls)
    assert "st.container(border=True)" not in controls_source
    assert "##### " not in controls_source


def test_the_popover_settings_render_and_keep_their_values_across_reruns() -> None:
    """§14.3e/§5.1（2026-08-29）: `on_change` を渡さない `st.popover` の既定
    (`"ignore"`) は、開閉に関わらず中身を毎回実行し、widget の値をそのまま保つ。

    `AppTest` は開閉そのものを模擬しないが、それは実機と同じ理由による——値は
    widget 自身の session state に残るのであって、ポップオーバーの開閉状態には
    紐づかない。`Block` は子孫を再帰的に yield するため、`at.toggle` / `at.radio`
    / `at.select_slider` はポップオーバーの中の widget も列挙できる。

    罠: ポップオーバーへ `on_change="rerun"` かコールバックを渡す実装は、この
    ような widget の列挙結果そのものは変えないが、実機では閉じている間に中身の
    再実行を止め、選んだ値がフル再実行のたびに黙って既定へ戻る（`on_change` の
    不在は `test_the_settings_live_in_a_popover_below_the_video` が別に固定する）。
    """
    from face_symmetry_lab.ui.camera import DEMOGRAPHIC_ENABLED_STATE_KEY

    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker())
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.camera import _render_display_controls\n"
        "with st.popover('表示設定'):\n"
        "    _render_display_controls(st.session_state['pipeline'])\n"
    )
    at.session_state["pipeline"] = pipeline
    at.run()
    assert not at.exception

    assert {toggle.key for toggle in at.toggle} >= {
        "camera_midline_enabled",
        DEMOGRAPHIC_ENABLED_STATE_KEY,
    }
    assert at.radio  # HUD 位置
    assert at.select_slider  # 平滑化・保持時間
    assert at.button  # セッション操作

    midline = next(toggle for toggle in at.toggle if toggle.key == "camera_midline_enabled")
    assert midline.value is True
    midline.set_value(False).run()

    assert at.session_state["camera_midline_enabled"] is False
    # 値を変えた後にもう一度フル再実行しても、選んだ値が黙って既定へ戻らない。
    at.run()
    assert at.session_state["camera_midline_enabled"] is False


def test_the_tabs_come_from_the_shared_component() -> None:
    """§5.1: 名前・順序・アイコンは `ui.result_tabs` が Image と共有する。

    罠: ここで `st.tabs([...])` を手書きへ戻す実装は、Image 側と食い違っても
    どちらの test も落ちない(実際に「まとめ・エクスポート」と「エクスポート」に
    分かれていた)。ラベル文字列がこの module に無いことまで固定する。
    """
    import inspect

    from face_symmetry_lab.ui import camera
    from face_symmetry_lab.ui.result_tabs import result_tab_labels

    source = inspect.getsource(camera.render_camera_section)

    assert "render_result_tabs(" in source
    # 停止中の 2 タブも同じ定数から取る(STOP の前後で名前が変わらない)。Camera
    # は「左右対称との比較」を持たない(項目 33)。
    for label in result_tab_labels(with_symmetry=False):
        assert label not in source, label


def test_every_live_fragment_reads_the_same_held_snapshot() -> None:
    """FR-041/AC-HOLD-01: 保持中に HUD と詳細結果が別々の瞬間の値を出さない。

    罠: 片方だけ `pipeline.snapshot()` を直接読む実装だと、保持区間の境界で
    「HUD の部位別」と「計算過程の部位別」が食い違い、AC-DISPLAY-04 の文字列
    一致がそこでだけ壊れる。参考年齢の fragment（2026-08-26）も同じ snapshot を
    読む——HUD と別の瞬間のゲート判定を読むと、片方だけが数値を出す。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    for fragment in (
        camera._render_hud_fragment,
        camera._render_detail_fragment,
        camera._render_demographic_fragment,
    ):
        source = inspect.getsource(fragment)
        assert "_snapshot_for_display(pipeline)" in source, fragment.__name__
        assert "pipeline.snapshot()" not in source, fragment.__name__


def test_the_display_snapshot_keys_the_latch_on_the_shared_signature() -> None:
    """AC-HOLD-02: 鍵は `ui.readings.display_signature`（値の有無）から作る。

    罠: 値そのものを鍵にすると毎フレーム鍵が変わり、保持がまったく効かない。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._snapshot_for_display)

    assert "hold_for_display(" in source
    assert "display_signature(snapshot.analysis)" in source


# --- FR-021: セッション要約の配線 ---------------------------------------------


def test_the_export_tab_summarises_the_session_before_the_download_buttons() -> None:
    """FR-021: 要約はセッション単位の成果物なので、エクスポートと同じ場所へ置く。

    罠: 5Hz の詳細 fragment へ入れる実装は、最大3,000行の要約を毎秒5回計算する。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_export_panel)

    assert "summarize_session(" in source
    assert "render_session_summary(" in source
    # 関数名 `_render_export_panel` 自身に一致しないよう、呼び出し行を探す。
    assert source.index("render_session_summary(") < source.index("\n    render_export_panel(")


def test_the_session_summary_is_not_recomputed_by_the_five_hertz_fragment() -> None:
    import inspect

    from face_symmetry_lab.ui import camera

    for fragment in (
        camera._render_hud_fragment,
        camera._render_detail_fragment,
        camera._render_demographic_fragment,
    ):
        assert "summarize_session" not in inspect.getsource(fragment), fragment.__name__


def test_the_report_tab_still_carries_both_of_its_contents() -> None:
    """§5.1（2026-08-27）: タブ名は Image とそろえて「レポート」にした。

    名前をそろえた代わりに、**中身が減っていないこと**をここで固定する——
    Camera のレポートタブは FR-021 の要約を画面にも出し、その下に FR-042 の
    ダウンロードを置く。罠: 名前だけそろえて要約を別の場所へ移す実装は、
    「まとめはどこへ行ったのか」を利用者に探させる。
    """
    import inspect

    from face_symmetry_lab.ui import camera
    from face_symmetry_lab.ui.result_tabs import REPORT_TAB_LABEL

    source = inspect.getsource(camera._render_export_panel)

    assert "render_session_summary(" in source
    assert "render_export_panel(" in source
    assert REPORT_TAB_LABEL == "📄 レポート"


def test_the_camera_report_carries_no_photo_at_all() -> None:
    """2026-08-16: リアルタイムのレポートに顔写真は載せない。

    図を持たないので、Camera 側は不変条件 1 の限定解除を必要としない——顔写真が
    ディスクへ出る経路は Image だけに閉じる。罠: 取り込み済みビューを埋める旧
    実装や、直近フレームから作る実装はここで落ちる。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_export_panel)

    assert "build_camera_report(" in source
    assert "trend_series(" in source
    assert "build_report_figures" not in source
    assert "report_has_figures=False" in source


def test_every_tab_survives_stopping_the_camera() -> None:
    """§5.1（2026-08-27）: STOP でタブが減らない。

    結果は 1 セッションが終わってから読むものであり、押した瞬間に詳細結果・
    参考年齢が画面から消えてはならない。2026-08-29: 「左右対称との比較」は
    Camera から削除された（項目 33）ので、ここで確かめる停止中の中身も 2 つに
    減る。

    罠: `if not context.state.playing:` の枝で減らしたタブだけの `st.tabs([...])`
    を別に組む旧実装は、名前も数も並びもライブ中と食い違わせる（そして片方だけ
    ラベルを直せば、両方の test が通ったまま画面がずれる）。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.render_camera_section)

    # 停止中も同じ 1 つの合成を通る。タブを別に組む枝が無いことを固定する。
    assert "st.tabs(" not in source
    assert source.count("render_result_tabs(") == 1
    for stopped_body in (
        "_render_stopped_detail_panel(",
        "_render_stopped_demographic_panel(",
    ):
        assert stopped_body in source, stopped_body
    # レポートと表示設定は再生状態で分岐しない（同じ中身がそのまま残る）。
    assert source.count("_render_export_panel(pipeline)") == 1


def test_the_camera_report_takes_its_population_from_the_shown_estimates() -> None:
    """AC-BASELINE-05: 記録ではなく pipeline の観測から母集団を決める。

    罠: `bundle.records` の `estimated_age_years` から作る実装は、年齢ゲートが
    表示を止めたフレームの年齢も混ぜる——§9.2 の `age_gate_decision` 列が未実装
    なので記録側では区別も付かず、ゲートが隠した判断が母集団の年齢から復元できて
    しまう（§10.12-3）。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_export_panel)

    assert "observations = pipeline.demographic_observations" in source
    assert "resolve_session_baseline(observations)" in source
    assert "estimated_age_years" not in source


# --- §5.1（2026-08-26）: 参考年齢・参考性別の独立区画 -------------------------
#
# 区画そのものの描画（状態ごとの 1 行・ゲートの適用・来歴）は Image と共有する
# ようになったため、`tests/ui/test_demographic_panel.py` が受け持つ。ここに残す
# のは **Camera 側の配線** ——共有パネルへ何を渡しているか——だけである。


def test_the_camera_panel_is_the_shared_one() -> None:
    """§5.1: Image と同じ区画を使う。罠: Camera 側にだけ分岐を書き足す実装は、
    「片方の入力でしか出ない状態」を作り、そこからゲートの判断が漏れる。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_demographic_area)

    assert "render_demographic_panel(" in source
    # 表示可否の分岐は共有パネルが持つ。ここへ書き戻していないことを固定する。
    assert "resolve_reference_age_view(" not in source
    assert "render_demographic_estimate(" not in source
    assert not hasattr(camera, "_render_demographic_panel")


def test_the_camera_panel_is_fed_the_gate_and_the_worker_state() -> None:
    """共有パネルは引数だけを読む。渡し忘れると常に「表示できる」側へ倒れる。

    罠: `available` を渡さない実装は既定 `True` になり、worker を持たない
    セッションでも「まだ推定できていない」と言い続ける（FR-024）。
    """
    import inspect

    from face_symmetry_lab.ui import camera

    # ライブと停止中で 1 つの関数を共有する（罠: 2 か所に分けると、停止中だけ
    # ゲートや worker の状態を渡し忘れても気づけない）。
    source = inspect.getsource(camera._render_demographic_area)

    for argument in (
        "estimate=snapshot.demographic_estimate",
        "gate_outcome=snapshot.age_gate_outcome",
        "available=pipeline.has_demographic_worker",
        "show_demographics=",
        "pending_notice=",
        "footnote=",
    ):
        assert argument in source, argument

    for caller in (camera._render_demographic_fragment, camera._render_stopped_demographic_panel):
        assert "_render_demographic_area(" in inspect.getsource(caller), caller.__name__


def test_the_camera_panel_reports_the_range_it_showed_this_session() -> None:
    """§9.6.1 と同じ材料を画面にも出す（保存物だけが知っている状態にしない）。"""
    from face_symmetry_lab.application.demographic_observations import DemographicOverview
    from face_symmetry_lab.domain.reference_baseline import ObservedAges
    from face_symmetry_lab.ui import camera

    caption = camera._session_demographic_caption(
        DemographicOverview(
            ages=ObservedAges(median=31, low=29, high=34, sample_count=42),
            gender_probabilities=(0.3, 0.7),
        )
    )

    assert "29〜34 歳" in caption
    assert "42" in caption
    assert "70%" in caption
    assert "男性" in caption


def test_the_detail_tab_no_longer_carries_the_reference_age() -> None:
    """同じ測定値を画面の 2 か所へ出さない（§5.1）。専用タブが唯一の受け皿になる。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_detail_fragment)

    assert "render_demographic_estimate" not in source
    assert "demographic_estimate" not in source


def test_the_camera_report_carries_what_the_session_actually_showed() -> None:
    """§9.6.1（2026-08-26）: レポートの参考年齢も、表示できた観測だけから作る。"""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_export_panel)

    # 母集団（`resolve_session_baseline`）と同じ 1 つの snapshot から作る。2 度
    # 取ると、その間に届いた推定の分だけ節と母集団がずれる。
    assert "observations = pipeline.demographic_observations" in source
    assert "summarize_observations(observations)" in source
    assert "overview=overview" in source


# --- §5.1（2026-08-27）: 停止後の各区画の中身 ---------------------------------


class ScoredAnalyzer:
    """品質ゲートを通った 1 フレーム分の結果を返す（停止後の表示検証用）。

    合成フレームでは実際の品質ゲート（顔の大きさ・向き）が閉じるため、表示側の
    検証には結果そのものを与える。`StubAnalyzer` は `Status.HOLD` で `symmetry`
    を持たないので、部位別の描画までは届かない。
    """

    def analyze(
        self,
        frame: LandmarkFrame,
        *,
        at: float,
        rgb_image: np.ndarray | None = None,
    ) -> AnalysisOutcome:
        del frame, at, rgb_image
        return _outcome_for_display(
            symmetry_3d=_symmetry_3d_result(
                contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018
            ),
            smooth_3d=None,
        )

    def reset(self) -> None:
        return None


def _stopped_pipeline(*, scored: bool = True) -> RealtimeLandmarkPipeline:
    """1 フレームだけ解析して止めた pipeline（= `STOP` を押した直後の状態）。

    WebRTC のコールバックが呼ばれなくなるだけなので、pipeline も snapshot も
    そのまま残る。停止中の区画が読むのはこの「最後に解析したフレーム」である。
    """
    analyzer = (
        ScoredAnalyzer()
        if scored
        else FrameAnalyzer(config=_symmetrizable_config(), config_3d=_symmetrizable_config())
    )
    pipeline = RealtimeLandmarkPipeline(
        landmarker=SymmetrizableLandmarker(),
        analyzer=analyzer,  # type: ignore[arg-type]
        analysis_interval_ms=0,
    )
    pipeline.process_with_analysis(np.full((80, 80, 3), 30, dtype=np.uint8))
    return pipeline


def _stopped_panel(function: str, *, pipeline: RealtimeLandmarkPipeline) -> AppTest:
    at = AppTest.from_string(
        "import streamlit as st\n"
        f"from face_symmetry_lab.ui.camera import {function}\n"
        f"{function}(st.session_state['pipeline'])\n"
    )
    at.session_state["pipeline"] = pipeline
    at.run()
    assert not at.exception
    return at


def test_the_stopped_detail_panel_shows_the_last_analysed_frame() -> None:
    """§5.1: 停止中は部位別 4 つもここへ出る（HUD が数値を publish しないため）。

    罠: ライブ用の `show_parts=False` をそのまま持ち込む実装は、停止した瞬間に
    部位別 4 つを画面のどこからも読めなくする——項目26（H3）の重複回避は、HUD が
    動いている間にだけ意味を持つ。
    """
    from face_symmetry_lab.ui.scores import SCORE_HEADING

    at = _stopped_panel("_render_stopped_detail_panel", pipeline=_stopped_pipeline())

    assert SCORE_HEADING in [element.value for element in at.subheader]
    labels = [metric.label for metric in at.metric]
    for part in ("輪郭", "目", "鼻", "口"):
        assert part in labels, part


def test_the_stopped_detail_panel_says_which_frame_it_is_showing() -> None:
    """不変条件 11: 古い値を現在値として見せない。最後のフレームだと明示する。"""
    at = _stopped_panel("_render_stopped_detail_panel", pipeline=_stopped_pipeline())

    captions = " ".join(caption.value for caption in at.caption)
    assert "最後に解析したフレーム" in captions


def test_the_stopped_detail_panel_explains_a_session_with_nothing_to_show() -> None:
    """FR-024: 一度も解析できなかったセッションでも空の枠を出さない。"""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(face_count=0))

    at = _stopped_panel("_render_stopped_detail_panel", pipeline=pipeline)

    assert at.info


def test_the_stopped_reference_age_panel_never_goes_silent() -> None:
    """FR-024/§5.1: 停止中も、何が起きているかを必ず 1 行で述べる。"""
    at = _stopped_panel("_render_stopped_demographic_panel", pipeline=_stopped_pipeline())

    assert at.info or at.error or at.warning


def test_the_idle_panels_do_not_claim_a_session_was_stopped() -> None:
    """§5.1（2026-08-27）: 開始前に「停止しました」と言わない。

    `START` を一度も押していない画面でも同じ 3 つのタブを出すが、そこで
    「最後に解析したフレーム」を名乗るのは、起きていないことを起きたように
    説明することになる。

    罠: 再生状態だけで分岐する実装は、初回表示でいきなり「計測を停止しました」
    と述べる——利用者はまだ何も始めていない。
    """
    import inspect

    from face_symmetry_lab.ui import camera
    from face_symmetry_lab.ui.camera import LAST_FRAME_CAPTION

    idle = RealtimeLandmarkPipeline(landmarker=StubLandmarker())
    assert camera._has_analysed_frame(idle.snapshot()) is False

    for function in ("_render_stopped_detail_panel", "_render_stopped_demographic_panel"):
        at = _stopped_panel(function, pipeline=idle)

        assert LAST_FRAME_CAPTION not in [caption.value for caption in at.caption], function
        assert at.info, function  # 黙らない（FR-024）

    assert any(
        "START" in element.value
        for element in _stopped_panel("_render_stopped_detail_panel", pipeline=idle).info
    )
    # 参考年齢側は、この pipeline に worker が無いのでモデル未配置の理由が先に
    # 出る（`ui.demographic_panel` の順序）。開始前の 1 行を選ぶ分岐そのものを
    # 固定しておく——罠: `pending_notice` を停止中の文で固定すると、開始前の
    # 画面が「表示できたフレームがありませんでした」と述べる。
    source = inspect.getsource(camera._render_stopped_demographic_panel)
    assert "IDLE_DEMOGRAPHIC_NOTICE" in source
    assert "_DEMOGRAPHIC_STOPPED_NOTICE" in source


def test_the_stopped_panels_do_say_which_frame_they_show_once_a_frame_exists() -> None:
    """裏返し: 1 フレームでも解析していれば、いつの値かを必ず述べる。"""
    from face_symmetry_lab.ui.camera import LAST_FRAME_CAPTION

    for function in ("_render_stopped_detail_panel", "_render_stopped_demographic_panel"):
        at = _stopped_panel(function, pipeline=_stopped_pipeline())

        assert LAST_FRAME_CAPTION in [caption.value for caption in at.caption], function
