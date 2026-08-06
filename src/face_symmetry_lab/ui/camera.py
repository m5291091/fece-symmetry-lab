"""リアルタイムのランドマーク表示を構成する Streamlit + streamlit-webrtc の層。

構成だけを担う。MediaPipe adapter をリアルタイム use case へ配線し、結果を描画
するのみで、幾何計算も推論もモデル取得もここでは行わない。WebRTC のコールバック
は worker thread で動くため、Streamlit の状態には決して触れない（§14.3）。

映像へ重ねる HUD は `ui.hud` が持つ。ブラウザ側資産（CSS・JS）は変更理由が
この構成層とは異なるためである。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import av
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_webrtc import WebRtcMode, WebRtcStreamerContext, webrtc_streamer

from face_symmetry_lab import __version__
from face_symmetry_lab.adapters.imaging.piecewise_warp import OpenCvPiecewiseWarper
from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    LandmarkDetectionError,
    MediaPipeFaceLandmarker,
)
from face_symmetry_lab.adapters.inference.mediapipe_segmenter import (
    MediaPipeSegmenter,
    SegmentationError,
)
from face_symmetry_lab.adapters.inference.opencv_demographic_estimator import (
    DemographicInferenceError,
    OpenCvDemographicEstimator,
)
from face_symmetry_lab.adapters.inference.tracking_zoom_landmarker import (
    TrackingZoomLandmarker,
)
from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.adapters.io.model_store import LOCK_PATH, InstallStatus, file_sha256
from face_symmetry_lab.application.analyzer import FrameAnalyzer
from face_symmetry_lab.application.demographic_worker import DemographicWorker
from face_symmetry_lab.application.realtime import (
    DEFAULT_ANALYSIS_INTERVAL_MS,
    RealtimeLandmarkPipeline,
)
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.application.symmetrization import symmetrize_frame
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfigError
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.domain.symmetrization import SymmetrizationError
from face_symmetry_lab.ui.export_panel import render_export_panel
from face_symmetry_lab.ui.guidance import guidance_message
from face_symmetry_lab.ui.hud import (
    HUD_POSITION_OPTIONS,
    HUD_POSITION_STATE_KEY,
    render_hud_bridge,
    render_hud_payload,
)
from face_symmetry_lab.ui.model_setup import (
    configured_demographic_path,
    configured_segmenter_path,
    demographic_asset,
    demographic_model_status,
    landmark_asset,
    resolve_ready_landmarker,
    segmenter_asset,
    segmenter_status,
)
from face_symmetry_lab.ui.scores import (
    DEVIATION_UNIT_CAPTION,
    format_distance,
    render_demographic_estimate,
    render_headline_metrics,
    render_score_details,
    render_symmetrized_image,
    render_tracking_status,
)

_BACKGROUND_MODE_LABELS: Final[dict[BackgroundMode, str]] = {
    BackgroundMode.OFF: "しない",
    BackgroundMode.BLACK: "黒",
    BackgroundMode.WHITE: "白",
}

#: FR-032: 表示位置のプリセット。採点には一切影響しない。
_PANEL_LAYOUT_RIGHT: Final[str] = "動画の右"
_PANEL_LAYOUT_LEFT: Final[str] = "動画の左"
_PANEL_LAYOUT_BELOW: Final[str] = "動画の下"
_PANEL_LAYOUT_OPTIONS: Final[tuple[str, ...]] = (
    _PANEL_LAYOUT_RIGHT,
    _PANEL_LAYOUT_LEFT,
    _PANEL_LAYOUT_BELOW,
)
PANEL_LAYOUT_STATE_KEY = "camera_panel_layout"

FULLSCREEN_STATE_KEY = "camera_fullscreen"
DEMOGRAPHIC_ENABLED_STATE_KEY = "camera_demographic_enabled"

#: §10.2 は STUN と TURN を禁じている。ICE を空にすることでセッションを
#: loopback へ閉じ込める。
LOOPBACK_RTC_CONFIGURATION: Final[dict[str, list[object]]] = {"iceServers": []}

#: streamlit-webrtc 自身の `video_html_attrs`。`<video>` を設定する公式の手段
#: であり、コンポーネントの iframe へ手を伸ばすより安定している。要点は 2 つ。
#:
#: * `controls: False` はネイティブのコントロールバーごと取り除く。その全画面
#:   ボタンは `<video>` をブラウザの top layer へ昇格させ、HUD を含むあらゆる
#:   ページ要素の上へ出してしまう。再生・一時停止・シークもライブ表示には
#:   意味がない。START/STOP はコンポーネント自身のボタンであり、これらとは別。
#: * 既定は `style={"width": "100%"}` かつ高さ auto である。これが全高フレーム
#:   の上端で映像が帯状のままになる原因だった。全画面時のサイズ調整は
#:   `ui.hud` の `VIDEO_CHROME_CSS` で仕上げる。
VIDEO_HTML_ATTRS: Final[dict[str, object]] = {
    "autoPlay": True,
    "controls": False,
    "muted": True,
    "playsInline": True,
    "controlsList": "nofullscreen noremoteplayback nodownload",
    "disablePictureInPicture": True,
    "style": {"width": "100%", "background": "#000"},
}

#: FR-005: 映像のみ。音声の許可は要求しない。
MEDIA_STREAM_CONSTRAINTS: Final[dict[str, object]] = {
    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
    "audio": False,
}

WEBRTC_KEY = "face-landmark-preview"
PIPELINE_STATE_KEY = "landmark_pipeline"

#: FR-034: 利用者が取り込んだフレームの対称化ビュー。セッション状態のみに置き、
#: ディスクへ書かず、解析のリセットと一緒に消す（§10.4）。
SYMMETRIZATION_STATE_KEY = "symmetrization_view"

#: §11 NFR-003: 結果パネルは映像の fps とは独立に 4〜5 Hz で更新する。
STATUS_REFRESH_INTERVAL = "200ms"

_STATUS_MESSAGES: Final[dict[TrackingStatus, str]] = {
    TrackingStatus.NO_FACE: "顔が見つかりません。カメラの正面に入ってください。",
    TrackingStatus.SINGLE_FACE: "1 名の顔を追跡しています。",
    TrackingStatus.MULTI_FACE: "2 名以上が写っています。1 名だけが写るようにしてください。",
    TrackingStatus.ERROR: "このフレームを処理できませんでした。そのままお待ちください。",
}

VideoFrameCallback = Callable[[av.VideoFrame], av.VideoFrame]


def build_video_frame_callback(pipeline: RealtimeLandmarkPipeline) -> VideoFrameCallback:
    """リアルタイム pipeline を streamlit-webrtc のフレームコールバックへ適合させる。

    FR-013: pipeline が描くのはランドマークと正中線だけである（1 フレーム
    あたり約 1ms）。数値パネルを意図的にここへ描かないのは、720p/30fps で 1
    フレームに使える 33ms のうち 18.6ms を消費し、それが遅延として表面化した
    ためである。数値は映像の隣のパネルが受け持つ。画素が変わらなかった場合は、
    届いたフレームを再エンコードせずそのまま返す。
    """

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        try:
            rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            painted, _ = pipeline.process_with_analysis(rgb)
            if painted is rgb:
                # 無変更なので、新しい VideoFrame への再エンコードを省く。
                return frame
            replaced = av.VideoFrame.from_ndarray(painted, format="rgb24")
        except Exception:
            # 元のフレームを返すことで表示を生かし続ける（NFR-017）。
            return frame
        if frame.pts is not None:
            replaced.pts = frame.pts
        if frame.time_base is not None:
            replaced.time_base = frame.time_base
        return replaced

    return callback


def _build_segmenter() -> MediaPipeSegmenter | None:
    """背景置換は任意機能である（§14.3b）。カメラを止める理由にはしない。"""
    path = configured_segmenter_path()
    if segmenter_status(path) is not InstallStatus.READY:
        return None
    try:
        return MediaPipeSegmenter(model_path=path, model_version=segmenter_asset().version)
    except SegmentationError:
        return None


def _build_demographic_worker() -> DemographicWorker | None:
    """年齢・性別推論は任意機能である（§14.3, FR-024）。カメラを止める理由にはしない。

    `resolve_ready_demographic` ではなく状態を黙って問い合わせる関数を使う。
    前者は自前の状態ブロックを描画するが、builder が UI を描いてはならない。
    """
    path = configured_demographic_path()
    if demographic_model_status(path) is not InstallStatus.READY:
        return None
    try:
        estimator = OpenCvDemographicEstimator(
            onnx_path=path,
            model_version=demographic_asset().version,
        )
    except DemographicInferenceError:
        return None
    return DemographicWorker(estimator=estimator)


def _build_pipeline(model_path: Path) -> RealtimeLandmarkPipeline:
    config = load_metric_config()
    landmarker = MediaPipeFaceLandmarker(
        model_path=model_path,
        model_version=landmark_asset().version,
    )
    # §14.3a: PC の webcam はスマートフォンの自撮りよりも顔が小さく写るため、
    # 直前の検出位置を中心に切り出して拡大してから MediaPipe へ渡す。wrapper は
    # 下の `landmarker` 参照を通じて `model_version`/`model_sha256` を報告する。
    zoomed_landmarker = TrackingZoomLandmarker(landmarker)
    analyzer = FrameAnalyzer(config=config)
    recorder = SessionRecorder(
        metric_config=config,
        thresholds=analyzer.thresholds,
        tau_seconds=DEFAULT_TAU_SECONDS,
        landmark_model_version=landmarker.model_version,
        landmark_model_sha256=landmarker.model_sha256,
        app_version=__version__,
        lock_digest=file_sha256(LOCK_PATH),
        analysis_interval_ms=DEFAULT_ANALYSIS_INTERVAL_MS,
    )
    return RealtimeLandmarkPipeline(
        landmarker=zoomed_landmarker,
        analyzer=analyzer,
        recorder=recorder,
        segmenter=_build_segmenter(),
        demographic_worker=_build_demographic_worker(),
        analysis_interval_ms=DEFAULT_ANALYSIS_INTERVAL_MS,
        midline_indices=config.midline_indices,
        # §15.2: Advanced 設定に手動 UI は無い。明るさ・ズーム・cadence は無条件
        # に AI が自動調整し（FR-029, FR-030）、品質ゲートの閾値は上のコンストラクタ
        # 既定値のまま固定して編集手段を設けない。自動調整の対象はいずれも採点に
        # 影響しない。
        auto_tune_enabled=True,
    )


def _session_pipeline(model_path: Path) -> RealtimeLandmarkPipeline:
    """1 つの Streamlit セッションにつき landmark runtime を 1 つに保つ（§13.4）。"""
    pipeline = st.session_state.get(PIPELINE_STATE_KEY)
    if not isinstance(pipeline, RealtimeLandmarkPipeline):
        pipeline = _build_pipeline(model_path)
        st.session_state[PIPELINE_STATE_KEY] = pipeline
    return pipeline


def release_session_pipeline() -> None:
    """landmark runtime を閉じ、セッションからの参照も落とす（FR-004, §10.4）。"""
    pipeline = st.session_state.pop(PIPELINE_STATE_KEY, None)
    st.session_state.pop(SYMMETRIZATION_STATE_KEY, None)
    if isinstance(pipeline, RealtimeLandmarkPipeline):
        pipeline.close()


@st.fragment(run_every=STATUS_REFRESH_INTERVAL)
def _render_headline_fragment(pipeline: RealtimeLandmarkPipeline) -> None:
    """不変 snapshot を 5 Hz で読み、見出しブロックだけを描画する。

    FR-013/FR-032: 総合スコア・参考性別・参考年齢・追跡状態・位置誘導は映像の
    隣（配置はプリセットに従う）へ置き、カメラから目を離さずに読めるようにする。
    それ以外は詳細であり、常に画面幅いっぱいで下に出る `_render_detail_fragment`
    が受け持つ（§5.1, AC-LAYOUT-01）。
    """
    snapshot = pipeline.snapshot()
    with st.container(border=True):
        symmetry = snapshot.analysis.symmetry if snapshot.analysis is not None else None
        overall = (
            snapshot.analysis.overall_distance_smooth if snapshot.analysis is not None else None
        )
        if overall is None and symmetry is not None:
            overall = symmetry.overall
        render_headline_metrics(overall=overall, estimate=snapshot.demographic_estimate)
        st.divider()
        render_tracking_status(snapshot.status, _STATUS_MESSAGES)

        if snapshot.analysis is not None:
            guidance = guidance_message(snapshot.analysis.state)
            if guidance is not None:
                st.warning(guidance)
        if snapshot.needs_recovery:
            st.warning("処理が続けて失敗しています。カメラを停止して再開してください。")

    # §14.3e: 3 つ目の fragment を新設せず、既存の 5 Hz fragment へ相乗りさせる。
    # 対象要素は隠されていて id で参照されるため、DOM 上のどこへ出ても構わない。
    render_hud_payload(
        snapshot,
        show_demographics=bool(st.session_state.get(DEMOGRAPHIC_ENABLED_STATE_KEY, True)),
    )


def _render_headline_panel(
    pipeline: RealtimeLandmarkPipeline,
    context: WebRtcStreamerContext[Any, Any],
) -> None:
    """「ライブ解析」の見出しと見出しブロック。配置は FR-032 のプリセットに従う。"""
    st.markdown("#### ライブ解析")
    if context.state.playing:
        _render_headline_fragment(pipeline)
    else:
        st.info("`START` を押すと、ブラウザがカメラの許可を求めます。")


@st.fragment(run_every=STATUS_REFRESH_INTERVAL)
def _render_detail_fragment(pipeline: RealtimeLandmarkPipeline) -> None:
    """不変 snapshot を 5 Hz で読み、画面幅いっぱいの詳細ブロックを描画する。"""
    snapshot = pipeline.snapshot()
    with st.container(border=True):
        # 測定結果を先頭に置く。下の実行時カウンタは診断用であり、利用者が
        # 手を打てないフレーム統計で結果を埋もれさせていた。
        if snapshot.analysis is not None:
            render_score_details(snapshot.analysis)
        else:
            st.info("品質ゲートを満たすと、ここに部位別の数値が表示されます。")

        if snapshot.demographic_estimate is not None:
            st.divider()
            # グラフは描かない。この fragment は 5 Hz で再描画される（§11 NFR-003）。
            render_demographic_estimate(
                snapshot.demographic_estimate,
                show_distribution=False,
            )

        st.divider()
        with st.expander("処理の状態（診断用）"):
            tracking, points = st.columns(2)
            tracking.metric("追跡状態", snapshot.status.value)
            points.metric("ランドマーク点数", snapshot.landmark_count)
            sampled, busy = st.columns(2)
            sampled.metric("プレビュー専用フレーム", snapshot.throttled_frames)
            busy.metric("処理中スキップ", snapshot.dropped_frames)
            st.caption(
                f"解析サンプル #{snapshot.frame_id} · 処理 {snapshot.processing_ms:.1f} ms · "
                f"最大 {1000 // DEFAULT_ANALYSIS_INTERVAL_MS} Hz"
            )
            # §14.3f: 上の合計を段階別に分けた NFR-002 の診断表示。遅いセッション
            # を GPU や配備の変更を検討する前に段階まで絞り込めるようにする。
            # 背景置換と scoring はその段階が実際に走ったときだけ出す。
            breakdown = [f"landmark {snapshot.landmark_ms:.1f} ms"]
            if snapshot.segmentation_ms > 0.0:
                breakdown.append(f"背景置換 {snapshot.segmentation_ms:.1f} ms")
            if snapshot.scoring_ms > 0.0:
                breakdown.append(f"scoring {snapshot.scoring_ms:.1f} ms")
            st.caption("内訳: " + " · ".join(breakdown))
            _render_auto_tune_status(pipeline)


def _render_detail_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """部位別・姿勢・肌表面参考値などの詳細結果。配置プリセットに関わらず常に

    動画とヘッダー結果の下へ画面幅いっぱいで表示する（FR-032, AC-LAYOUT-01）。
    """
    _render_detail_fragment(pipeline)


def _clear_symmetrization(pipeline: RealtimeLandmarkPipeline) -> None:
    """取り込んだフレームとその表示を一緒に破棄する（§10.4, FR-034）。"""
    pipeline.clear_capture()
    st.session_state.pop(SYMMETRIZATION_STATE_KEY, None)


def _consume_pending_capture(pipeline: RealtimeLandmarkPipeline, *, warp: bool) -> None:
    """取り込み済みフレームを、リアルタイム経路の外で一度だけ表示用へ変換する。

    pipeline はフレームをちょうど 1 回だけ引き渡すので、ここで一発の対称化を
    実行し、結果だけを保持する。対称化できなかった取り込みは、黙って再試行せず
    利用者へ報告する。

    ``warp`` は §8.6.4 の画像を要求する。全画素を作り直すため任意指定とし、
    取り込み時点の値を読む。生フレームはここで消費され、以後は保持しない
    （§10.4）。
    """
    captured = pipeline.take_capture()
    if captured is None:
        return
    config = pipeline.metric_config
    if config is None:
        st.session_state[SYMMETRIZATION_STATE_KEY] = "指標設定が読み込まれていません。"
        return
    try:
        st.session_state[SYMMETRIZATION_STATE_KEY] = symmetrize_frame(
            captured.rgb_image,
            captured.frame,
            config=config,
            warper=OpenCvPiecewiseWarper() if warp else None,
        )
    except SymmetrizationError as error:
        st.session_state[SYMMETRIZATION_STATE_KEY] = str(error)


def _render_symmetrization_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """FR-034: ある瞬間を取り込み、自身の正中線で折り返した姿を並べて示す。

    5 Hz の fragment の外に置いている。これは 1 フレームから一度だけ計算する
    静止結果であり、表示中もライブ表示はそのまま動き続ける（§14.3g）。
    """
    st.markdown("#### 左右対称にそろえた位置との比較")
    st.caption(
        "この瞬間の顔を正中線で折りたたみ、左右の対応点の中点を「左右対称にそろえた位置」として"
        "実測点と重ねて表示します。ズレの大きさを形で見るための表示で、"
        "どちらが正しい・望ましいという意味はありません。"
    )

    # ボタンより前に読む。取り込みを起こすトグルの現在値が、その取り込みへ
    # 適用される値と一致するようにするためである。
    warp = st.checkbox(
        "左右対称にそろえた画像も生成する（幾何変形・処理が重くなります）",
        key="camera_symmetrization_warp",
        help=(
            "ランドマークを結んだ三角形ごとに画素を変形し、左右対称にそろえた顔画像を作ります。"
            "加工画像であり、無加工の写真ではありません。"
        ),
    )

    _consume_pending_capture(pipeline, warp=warp)
    controls = st.columns(2)
    if controls[0].button("この瞬間を対称化して見る", key="capture_symmetrization"):
        pipeline.request_capture()
        st.rerun()
    if controls[1].button("表示を消す", key="clear_symmetrization"):
        _clear_symmetrization(pipeline)
        st.rerun()

    view = st.session_state.get(SYMMETRIZATION_STATE_KEY)
    if view is None:
        if pipeline.capture_pending:
            st.info("次に顔を検出したフレームを取り込みます。そのままお待ちください。")
        return
    if isinstance(view, str):
        st.warning(view)
        return

    st.image(
        view.overlay,
        caption="実測点と、左右対称にそろえた位置（色違い）を線で結んだ表示",
        width="stretch",
    )
    mean_column, max_column = st.columns(2)
    mean_column.metric("平均のズレ", format_distance(view.mean_deviation))
    max_column.metric("最大のズレ", format_distance(view.max_deviation))
    st.caption(
        f"対称化した点 {view.symmetrized_point_count} 点。"
        "対応する左右のペアが定義されていない点は、実測のまま動かしていません。"
    )
    st.caption(DEVIATION_UNIT_CAPTION)
    render_symmetrized_image(view, requested=warp)
    st.caption("取り込んだ映像はこのブラウザセッションのメモリだけに残り、保存も送信もしません。")


def _render_export_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    render_export_panel(
        pipeline.export_bundle(),
        heading="セッションデータのエクスポート",
        warning=(
            "姿勢・左右差などの顔由来の派生データを含みます。"
            "共有先と利用目的を確認してから保存してください。"
        ),
        key_prefix="download_session",
        # ライブセッションは記録を積み上げるため、件数を示してこの
        # ダウンロードが実際にどこまでを含むかを伝える。
        show_record_count=True,
    )


def _render_auto_tune_status(pipeline: RealtimeLandmarkPipeline) -> None:
    """§15.2: Advanced 設定に手動 UI は無い。AI が選んだ値を表示する。

    品質ゲートの閾値は固定の定数であり、調整可能であるかのようにここへ並べる
    ことは意図的に避けている（§14.3d）。列挙するどの項目も測定値へは影響しない。
    §8.6 の指標には調整できる係数が残っていないためである。
    """
    zoom_policy = pipeline.zoom_policy
    zoom_caption = f" · ズーム目標 {zoom_policy.target_px}px" if zoom_policy is not None else ""
    st.caption(
        "Advanced設定はAIが自動調整します（明るさ・追跡ズーム・"
        "history_capacity・age_interval_ms・analysis_interval_ms）。"
        f"現在値: 検出用の明るさ補正 ×{pipeline.brightness_gain:.2f}{zoom_caption} · "
        f"analysis_interval_ms={pipeline.analysis_interval_ms}。"
        "明るさ補正はlandmark検出にだけ使い、表示される映像の明るさは変えません。"
        "品質ゲート閾値は固定既定値です。左右差の計算には調整可能な係数が無いため、"
        "自動調整が数値を動かすことはありません。"
    )


def _render_display_controls(pipeline: RealtimeLandmarkPipeline) -> None:
    """§15.1 の表示専用の設定を、そのまま pipeline へ書き込む。

    どれもスコアを変えない（FR-027, FR-028）。正中線と背景は解析の後に描かれ、
    年齢・性別のトグルは元から左右差とは独立したパネルの表示可否を切り替える
    だけである（FR-024）。ここで読み、1 箇所で書き込むことで、その主張を
    検証可能な形に保つ。
    """
    midline_column, demographic_column = st.columns(2)
    with midline_column:
        show_midline = st.toggle(
            "顔の中心線を表示",
            value=True,
            key="camera_midline_enabled",
            help="採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。",
        )
    with demographic_column:
        # この pipeline が実際に保持している worker を条件にする。worker が無いのに
        # 機能が動いていると示すことがないようにするためである。
        show_demographics = st.toggle(
            "年齢・性別の参考値を表示",
            value=True,
            key=DEMOGRAPHIC_ENABLED_STATE_KEY,
            disabled=not pipeline.has_demographic_worker,
            help=(
                "18〜75歳向けモデル推定の参考年齢・参考性別を最大1Hzで独立推論します。"
                "左右差の計算とは分離しており、失敗してもカメラ映像は止まりません。"
            ),
        )
        if not pipeline.has_demographic_worker:
            st.caption("年齢・性別モデルが未配置のため使用できません。")
    pipeline.set_midline_enabled(show_midline)
    pipeline.set_demographic_enabled(show_demographics)

    # §5.4/§14.3e: 枠の色は状態の識別子であって評価ではない。同じ情報は映像の
    # 隣のパネルにテキストとしても残している。
    st.caption(
        "映像の枠色と HUD の枠線は品質ゲートの状態を示します。"
        "緑は「計測条件を満たしている」という意味で、顔の良し悪しを表すものではありません。"
        "同じ状態は映像の隣のテキストでも確認できます。"
    )

    background_column, status_column = st.columns(2)
    with background_column:
        segmenter_path = configured_segmenter_path()
        if segmenter_status(segmenter_path) is InstallStatus.READY:
            background_mode = st.radio(
                "背景を置き換える（表示専用）",
                options=list(_BACKGROUND_MODE_LABELS),
                format_func=lambda mode: _BACKGROUND_MODE_LABELS[mode],
                key="camera_background_mode",
                horizontal=True,
                help="人物以外を単色にします。左右差・年齢性別・肌表面参考値の判定には使いません。",
            )
            pipeline.set_background_mode(background_mode)
        else:
            st.caption("背景置換モデルが未配置のため、この機能は使用できません。")
    with status_column:
        _render_auto_tune_status(pipeline)


def _split_for_panel_layout() -> tuple[DeltaGenerator, DeltaGenerator | None]:
    """保存済みの FR-032 プリセットに対応する (preview, headline) の列を返す。

    widget を描画せず設定値だけを読む。§5.1 は映像を最上部に置くと定めている
    一方、それを変える操作子は下の「表示設定」タブにあるためである。widget は
    同じセッション状態のキーへ書くので、変更はそれが起こす再実行で反映される。

    表示専用であり、マウスのドラッグではなくプリセット選択で「動く」（§14.3e）。
    headline 列が ``None`` のときは「映像の下へ積む」という意味で、これだけが
    専用の列を必要としないプリセットである。§5.1 の 7:5 比は headline が
    どちら側へ来ても保たれる。
    """
    panel_layout = st.session_state.get(PANEL_LAYOUT_STATE_KEY, _PANEL_LAYOUT_RIGHT)
    if panel_layout == _PANEL_LAYOUT_BELOW:
        return st.container(), None
    if panel_layout == _PANEL_LAYOUT_LEFT:
        result_column, preview_column = st.columns([5, 7], gap="large")
        return preview_column, result_column
    preview_column, result_column = st.columns([7, 5], gap="large")
    return preview_column, result_column


def _render_panel_layout_control() -> None:
    """FR-032 のプリセット widget。「表示設定」タブの中で描画する。

    `st.tabs` はどのタブが画面に出ていても毎回すべてのタブの中身を構築する
    ため、この widget は常に存在し、値も保たれる。
    """
    st.radio(
        "「ライブ解析」パネルの配置",
        options=_PANEL_LAYOUT_OPTIONS,
        key=PANEL_LAYOUT_STATE_KEY,
        horizontal=True,
        help="見出し（対称度・参考性別・参考年齢・状態）の位置だけを変えます。数値は変わりません。",
    )
    st.radio(
        "映像上の HUD の位置",
        options=HUD_POSITION_OPTIONS,
        key=HUD_POSITION_STATE_KEY,
        horizontal=True,
        help="映像へ重ねる対称度・参考年齢・参考性別の位置です。映像自体は加工しません。",
    )


def _exit_fullscreen() -> None:
    """固定表示のボタンから全画面を抜ける。

    直接代入ではなくコールバックにしている。Streamlit は widget が生成された
    後にその session state キーを書き換えることを拒み、トグルは同じ実行の
    早い段階で作られているためである。`on_click` は次の実行が widget を作る
    前に走るので、書き込みが正当に成立する。
    """
    st.session_state[FULLSCREEN_STATE_KEY] = False


def _render_session_controls(pipeline: RealtimeLandmarkPipeline) -> None:
    """停止・リセットの操作子。メイン画面から到達できる位置に置く（§5.1）。"""
    st.caption("カメラ映像は映像上の `STOP` で停止します。")
    controls = st.columns(2)
    if controls[0].button("結果・履歴・平滑化をリセット", key="reset_analysis"):
        pipeline.reset_analysis()
        # `reset_analysis` は取り込み済みフレームを破棄する。そこから導いた
        # 表示も一緒に消さないと、パネルが自身の区間より長く生き残ってしまう。
        st.session_state.pop(SYMMETRIZATION_STATE_KEY, None)
        st.rerun()
    if controls[1].button("推論を停止してモデルを解放", key="release_landmarker"):
        release_session_pipeline()
        st.rerun()


def render_camera_section() -> None:
    """リアルタイムのランドマーク表示を描画する。同意の後にのみ呼ぶこと。

    §5.1: 映像とその見出しを最初に置き、その位置を保つ。設定・詳細数値・
    対称化・エクスポートはすべて下のタブへ入れ、一度に高さを占めるのが 1 つ
    だけになるようにする。これにより表示が画面外へ押し出されない。タブの
    切り替えは表示専用で、測定値を変えず WebRTC の通信も中断しない
    （AC-LAYOUT-01）。
    """
    st.subheader("リアルタイム ランドマーク表示")
    st.caption(
        "映像は端末内のメモリだけで処理され、保存も外部送信もしません。"
        "映像にはランドマークと中心線だけを重ね、全体の対称度・参考性別・参考年齢は"
        "映像のすぐ隣に大きく表示します。設定と詳しい数値は映像の下のタブにまとめています。"
    )

    model_path = resolve_ready_landmarker()
    if model_path is None:
        return

    try:
        pipeline = _session_pipeline(model_path)
    except (LandmarkDetectionError, MetricConfigError) as error:
        st.error(str(error))
        return

    fullscreen = st.toggle(
        "映像を全画面表示",
        key=FULLSCREEN_STATE_KEY,
        help=(
            "映像をブラウザ画面いっぱいに広げます。"
            "ブラウザの F11 と併用すると完全な全画面になります。"
        ),
    )
    # §5.2 のオーバーレイ例外: 1 回の実行につき一度だけ設置し、fragment の内側
    # では決して呼ばない。これにより 5 Hz の更新が CSS を送り直すことはない。
    render_hud_bridge(
        fullscreen=fullscreen,
        position=st.session_state.get(HUD_POSITION_STATE_KEY, HUD_POSITION_OPTIONS[0]),
    )
    if fullscreen:
        # 全画面中は下のタブが描画されないため、これが唯一の戻り道である。
        # 何かに覆われる前に存在している必要がある。
        with st.container(key="fsl_exit_fullscreen"):
            st.button("✕ 全画面を終了", key="exit_fullscreen", on_click=_exit_fullscreen)

    preview_column, result_column = _split_for_panel_layout()

    with preview_column, st.container(border=True, key="camera_preview_frame"):
        context = webrtc_streamer(
            key=WEBRTC_KEY,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=LOOPBACK_RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            video_frame_callback=build_video_frame_callback(pipeline),
            video_html_attrs=VIDEO_HTML_ATTRS,
            async_processing=True,
        )

    if result_column is None:
        _render_headline_panel(pipeline, context)
    else:
        with result_column:
            _render_headline_panel(pipeline, context)

    if not context.state.playing:
        # 開始前でも設定は画面上にあるべきだが、提示できる結果・取り込み・
        # エクスポートはまだ無い。
        with st.expander("表示設定"):
            _render_display_controls(pipeline)
            _render_panel_layout_control()
        return

    if fullscreen:
        # 映像がビューポートを覆っている間、その下の要素へは到達できない。
        # 誰も見られないパネルを 5 Hz で再描画し続けないよう、まるごと省く。
        st.caption("全画面表示中です。設定・詳細結果は「全画面を終了」で戻ると表示されます。")
        return

    # §5.1, AC-LAYOUT-01: 映像の下はすべて画面幅いっぱいで一度に 1 つだけ表示し、
    # 上の表示がスクロールで手の届かない位置へ行かないようにする。
    detail_tab, symmetry_tab, settings_tab, export_tab = st.tabs(
        ["詳細結果", "左右対称との比較", "表示設定", "エクスポート"]
    )
    with detail_tab:
        _render_detail_panel(pipeline)
    with symmetry_tab:
        # FR-034/§14.3g: 一度きりの静止表示。上のライブ表示はこれが画面に出て
        # いる間も自身の周期で動き続ける。
        _render_symmetrization_panel(pipeline)
    with settings_tab:
        _render_display_controls(pipeline)
        _render_panel_layout_control()
        st.divider()
        _render_session_controls(pipeline)
    with export_tab:
        _render_export_panel(pipeline)
