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
from typing import Final

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from face_symmetry_lab import __version__
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
from face_symmetry_lab.adapters.io.metric_config import (
    SYMMETRY_3D_METRIC_CONFIG_PATH,
    load_metric_config,
)
from face_symmetry_lab.adapters.io.model_store import LOCK_PATH, InstallStatus, file_sha256
from face_symmetry_lab.application.analyzer import FrameAnalyzer
from face_symmetry_lab.application.demographic_observations import (
    DemographicOverview,
    summarize_observations,
)
from face_symmetry_lab.application.demographic_worker import DemographicWorker
from face_symmetry_lab.application.realtime import (
    DEFAULT_ANALYSIS_INTERVAL_MS,
    PipelineSnapshot,
    RealtimeLandmarkPipeline,
)
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.metrics import MetricConfigError
from face_symmetry_lab.domain.session_summary import summarize_session, trend_series
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.ui.age_gate_notice import SESSION_HIDDEN_NOTICE
from face_symmetry_lab.ui.baseline import resolve_session_baseline
from face_symmetry_lab.ui.demographic_panel import render_demographic_panel
from face_symmetry_lab.ui.detail_panel import render_detail_panel
from face_symmetry_lab.ui.export_panel import render_export_panel
from face_symmetry_lab.ui.hud import (
    HUD_POSITION_OPTIONS,
    HUD_POSITION_STATE_KEY,
    render_hud_bridge,
    render_hud_payload,
)
from face_symmetry_lab.ui.i18n import t, translate_message
from face_symmetry_lab.ui.model_setup import (
    build_age_gate_service,
    configured_demographic_path,
    configured_segmenter_path,
    demographic_asset,
    demographic_model_status,
    landmark_asset,
    resolve_ready_landmarker,
    segmenter_asset,
    segmenter_status,
)
from face_symmetry_lab.ui.readings import display_signature, format_gender_probabilities
from face_symmetry_lab.ui.report import build_camera_report
from face_symmetry_lab.ui.result_tabs import (
    ResultTabs,
    render_result_tabs,
)
from face_symmetry_lab.ui.smoothing_panel import render_smoothing_control
from face_symmetry_lab.ui.summary_panel import render_session_summary

_BACKGROUND_MODE_LABELS: Final[dict[BackgroundMode, str]] = {
    BackgroundMode.OFF: "しない",
    BackgroundMode.BLACK: "黒",
    BackgroundMode.WHITE: "白",
}


def _background_mode_label(mode: BackgroundMode) -> str:
    """`_BACKGROUND_MODE_LABELS` の値（識別子）は動かさず、表示だけを翻訳する。

    `t()` は文字列リテラルか module-level 名しか受け取れないため、辞書引き
    （`_BACKGROUND_MODE_LABELS[mode]`）をそのまま渡すことはできない
    （`tests/ui/test_i18n.py` の AST 走査）。分岐で同じ 3 語を明示的に渡す。
    """
    if mode is BackgroundMode.OFF:
        return t("しない")
    if mode is BackgroundMode.BLACK:
        return t("黒")
    return t("白")


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

#: §5.1（2026-08-27）: 停止中の区画が読んでいるのは「そのセッションで最後に
#: 解析したフレーム」である。不変条件 11 が禁じるのは古い値を**現在値として**
#: 見せることなので、いつの値かを述べたうえで出す。
LAST_FRAME_CAPTION: Final[str] = (
    "計測を停止しました。以下はこのセッションで最後に解析したフレームの値です。"
)

#: 開始前の 1 行。**「停止しました」と言わない**——`START` を一度も押していない
#: 画面で「最後に解析したフレーム」を名乗るのは、起きていないことを起きたように
#: 説明することになる（§5.1、2026-08-27）。
IDLE_DETAIL_NOTICE: Final[str] = (
    "`START` を押して計測を始めると、ここに計算過程などの数値が表示されます。"
)
IDLE_DEMOGRAPHIC_NOTICE: Final[str] = (
    "`START` を押して計測を始めると、参考年齢・参考性別をここに表示します。"
)

#: §11 NFR-003: 結果パネルは映像の fps とは独立に 4〜5 Hz で更新する。
STATUS_REFRESH_INTERVAL = "200ms"

#: 参考年齢・参考性別だけは 1 Hz で足りる。worker 自身が最大 1 秒に 1 度しか
#: 推論しない（`DEFAULT_DEMOGRAPHIC_INTERVAL_MS`）ので、5 Hz で描き直しても
#: 同じ値を 5 回描くだけである（§11 NFR-003 と同じ判断）。
DEMOGRAPHIC_REFRESH_INTERVAL = "1s"

#: §14.3 / 2026-08-31: 詳細タブ（計算過程・姿勢・診断）は二次表示なので 1 Hz で
#: 描き直す。ライブの主表示は映像上 HUD（5 Hz、NFR-003/FR-013）が担う。詳細
#: パネルを 5 Hz でフル再描画すると、ブラウザのメインスレッドで WebRTC 映像の
#: デコード・描画と競合し、長いセッションで映像が次第に重くなり最後に止まる
#: （利用者報告）。サーバ側の解析経路は有界であることを実測で確認済み。
DETAIL_REFRESH_INTERVAL = "1s"

VideoFrameCallback = Callable[[av.VideoFrame], av.VideoFrame]


def build_video_frame_callback(pipeline: RealtimeLandmarkPipeline) -> VideoFrameCallback:
    """リアルタイム pipeline を streamlit-webrtc のフレームコールバックへ適合させる。

    FR-013: pipeline が描くのはランドマークと正中線だけである（1 フレーム
    あたり約 1ms）。数値パネルを意図的にここへ描かないのは、720p/30fps で 1
    フレームに使える 33ms のうち 18.6ms を消費し、それが遅延として表面化した
    ためである。数値は映像へ描かず、ブラウザ側の表示層が受け持つ（§14.3e）。
    画素が変わらなかった場合は、届いたフレームを再エンコードせずそのまま返す。
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
    # §8.5.1/§8.6.5, FR-036: symmetry-3d/1.0.0 は主指標であり、表示・エクスポート
    # の両方へ接続する。`config_3d` を省略可能にすると、呼び出し側が渡し忘れた
    # ときに主指標が静かに無しへ落ちてしまうため、`FrameAnalyzer`・
    # `SessionRecorder` のどちらも必須引数にしてある。ここで明示的に読み込んで
    # 渡す。
    config_3d = load_metric_config(SYMMETRY_3D_METRIC_CONFIG_PATH)
    landmarker = MediaPipeFaceLandmarker(
        model_path=model_path,
        model_version=landmark_asset().version,
    )
    # §14.3a: PC の webcam はスマートフォンの自撮りよりも顔が小さく写るため、
    # 直前の検出位置を中心に切り出して拡大してから MediaPipe へ渡す。wrapper は
    # 下の `landmarker` 参照を通じて `model_version`/`model_sha256` を報告する。
    zoomed_landmarker = TrackingZoomLandmarker(landmarker)
    analyzer = FrameAnalyzer(config=config, config_3d=config_3d)
    recorder = SessionRecorder(
        metric_config=config,
        metric_config_3d=config_3d,
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
        # FR-038: 既存の worker/relay 構造（推論の independent cadence）は変えず、
        # 判定だけを追加する。モデル・spec が欠けても `build_age_gate_service` は
        # 例外を投げず `classifier=None` の service へ倒れる（AC-AGEGATE-03）。
        age_gate=build_age_gate_service(),
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
    if isinstance(pipeline, RealtimeLandmarkPipeline):
        pipeline.close()


def _snapshot_for_display(pipeline: RealtimeLandmarkPipeline) -> PipelineSnapshot:
    """FR-041: 表示用の snapshot を 1 か所で作る（ADR-0008 (c)）。

    HUD の fragment と詳細結果の fragment はどちらもこれを通す。**ラッチは
    pipeline に 1 つしか無い**ので、どちらが先に呼んでも同じ解析結果を受け取り、
    §8.6.3 の計算過程と HUD の部位別が同じ値を出し続ける（AC-DISPLAY-04）。
    2 つの fragment がそれぞれラッチを持つと、同じ画面の 2 つの区画が別々の
    保持区間を刻むことになる。

    鍵（保持を打ち切る手掛かり）は `ui.readings.display_signature` が作る。
    値そのものではなく「値が届いているかどうかの型」なので、顔を見失った瞬間に
    保持が切れて数値が `—` へ戻る（AC-HOLD-02）。
    """
    snapshot = pipeline.snapshot()
    return pipeline.hold_for_display(snapshot, key=display_signature(snapshot.analysis))


@st.fragment(run_every=STATUS_REFRESH_INTERVAL)
def _render_hud_fragment(pipeline: RealtimeLandmarkPipeline) -> None:
    """不変 snapshot を 5 Hz で読み、映像上の HUD へ payload を publish するだけ。

    項目26（H3）: 映像の隣にあった「ライブ解析」パネル（見出しの数値・追跡状態・
    位置誘導）を削除した。HUD（`ui.hud`）が同じ内容を吹き出しとして表示する
    ため、ここでは何も描画しない。fragment 自体は残す——`render_hud_payload`
    が書く隠し要素をこれが 5 Hz で更新し続けないと、HUD の数値が動かなくなる
    ためである。
    """
    snapshot = _snapshot_for_display(pipeline)
    render_hud_payload(
        snapshot,
        show_demographics=bool(st.session_state.get(DEMOGRAPHIC_ENABLED_STATE_KEY, True)),
        # FR-040: snapshot ではなく pipeline から直接読む。snapshot は解析フレーム
        # ごとにしか作り直されないため、そこ経由だと選び直してから次の解析まで
        # 古い秒数が HUD に残る。
        smoothing_tau_seconds=pipeline.smoothing_tau_seconds,
        display_hold_seconds=pipeline.display_hold_seconds,
    )


#: FR-043: 動的な値を含む文言は f-string を `t()` へ渡せない
#: （鍵が実行のたび変わり永久に訳せない）ため、テンプレートへ切り出してから
#: `t(TEMPLATE).format(...)` する（ADR-0010 (a)）。
_DETAIL_SAMPLE_CAPTION_TEMPLATE: Final[str] = (
    "解析サンプル #{frame_id} · 処理 {processing_ms:.1f} ms · 最大 {max_hz} Hz"
)
_SEGMENTATION_BREAKDOWN_TEMPLATE: Final[str] = "背景置換 {ms:.1f} ms"
_BREAKDOWN_CAPTION_PREFIX: Final[str] = "内訳: "


def _detail_diagnostics(
    pipeline: RealtimeLandmarkPipeline,
    snapshot: PipelineSnapshot,
) -> Callable[[], None]:
    """§14.3f/NFR-002: 実行時の数値を折りたたみへ入れる中身を作る。

    ライブと停止中で同じ 1 か所から作る。2 つに分けると、片方だけが段階別の
    内訳を落としても気づけない——どちらも「遅いセッションを段階まで絞り込む」
    ための同じ診断値である。
    """

    def _render() -> None:
        tracking, points = st.columns(2)
        tracking.metric(t("追跡状態"), snapshot.status.value)
        points.metric(t("ランドマーク点数"), snapshot.landmark_count)
        sampled, busy = st.columns(2)
        sampled.metric(t("プレビュー専用フレーム"), snapshot.throttled_frames)
        busy.metric(t("処理中スキップ"), snapshot.dropped_frames)
        st.caption(
            t(_DETAIL_SAMPLE_CAPTION_TEMPLATE).format(
                frame_id=snapshot.frame_id,
                processing_ms=snapshot.processing_ms,
                max_hz=1000 // DEFAULT_ANALYSIS_INTERVAL_MS,
            )
        )
        # 背景置換と scoring はその段階が実際に走ったときだけ出す。
        breakdown = [f"landmark {snapshot.landmark_ms:.1f} ms"]
        if snapshot.segmentation_ms > 0.0:
            breakdown.append(
                t(_SEGMENTATION_BREAKDOWN_TEMPLATE).format(ms=snapshot.segmentation_ms)
            )
        if snapshot.scoring_ms > 0.0:
            breakdown.append(f"scoring {snapshot.scoring_ms:.1f} ms")
        st.caption(t(_BREAKDOWN_CAPTION_PREFIX) + " · ".join(breakdown))
        _render_auto_tune_status(pipeline)

    return _render


@st.fragment(run_every=DETAIL_REFRESH_INTERVAL)
def _render_detail_fragment(pipeline: RealtimeLandmarkPipeline) -> None:
    """不変 snapshot を 1 Hz で読み、画面幅いっぱいの詳細ブロックを描画する。

    **更新頻度を 5 Hz から 1 Hz へ落とした理由（2026-08-31 の利用者報告対応）。**
    ライブの主表示は映像上の HUD（5 Hz、NFR-003/FR-013）が担う。詳細タブは
    計算過程・姿勢・診断という二次表示であり、5 Hz でフル再描画する必要は無い。
    5 Hz でのフル再描画（bordered container + 2×columns + metrics + expander）は
    ブラウザのメインスレッドで WebRTC 映像のデコード・描画と競合し、長いセッション
    で映像が次第に重くなり最後に止まる障害を起こした。サーバ側の解析経路は有界で
    あることを実測で確認済み。詳細タブを 1 Hz にすることで競合を解消する。

    項目26（H3）: HUD が既に部位別 4 つ・全体の対称度・全体のスコアを表示する
    ため、`render_score_details` へ `show_parts=False` を渡して重複を避ける。

    2026-08-26: 参考年齢・参考性別はここから **専用タブへ移した**
    （`_render_demographic_fragment`）。ここに間借りしていた間、この区画は
    推定が無いときに何も描かず、年齢ゲートの判定も読んでいなかった——HUD が
    固定文言へ切り替わっている間もここだけが数値を出し続けており、§10.12-3 の
    「表示していないときは併記しない」が区画によって食い違っていた。

    FR-041: HUD と同じ `_snapshot_for_display` を通す。片方だけ生の snapshot を
    読むと、保持中に「HUD の部位別」と「計算過程の部位別」が別の瞬間の値になり、
    AC-DISPLAY-04 の文字列一致が保持区間の境界でだけ壊れる。

    2026-08-27: 囲み・回復導線・診断用の折りたたみという**枠**は
    `ui.detail_panel` が Image と共有する。ここに残るのは、その枠へ渡す
    「この入力に固有の中身」——`show_parts=False`（HUD との重複回避）と、
    NFR-002 のフレーム統計——だけである。
    """
    snapshot = _snapshot_for_display(pipeline)
    render_detail_panel(
        snapshot.analysis,
        empty_notice=t("品質ゲートを満たすと、ここに計算過程などの数値が表示されます。"),
        # 項目26（H3）: HUD が既に部位別 4 つを表示するため、詳細タブでは省く。
        show_parts=False,
        # §14.4: 回復導線。以前は映像隣の見出しパネルにあったが、パネルの削除
        # に伴いここへ移した。
        warning=(
            t("処理が続けて失敗しています。カメラを停止して再開してください。")
            if snapshot.needs_recovery
            else None
        ),
        diagnostics=_detail_diagnostics(pipeline, snapshot),
    )


def _has_analysed_frame(snapshot: PipelineSnapshot) -> bool:
    """このセッションで 1 フレームでも解析したか（= `START` を押したことがあるか）。

    `frame_id` は解析のたびに進み、`reset_analysis` で 0 へ戻る。停止中の区画は
    これを見て「まだ計測していない」と「計測が終わった」を分ける——同じ 1 行で
    両方を説明すると、片方が必ず嘘になる。
    """
    return snapshot.frame_id > 0


def _render_stopped_detail_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """§5.1（2026-08-27）: `STOP` の後の詳細結果。

    読むのは **そのセッションで最後に解析したフレーム**である。WebRTC の
    コールバックが呼ばれなくなるだけなので snapshot はそのまま残っており、
    ここではそれを「最後のフレームの値」と明示したうえで出す（不変条件 11:
    古い値を現在値として見せない）。

    fragment にしない。停止中は新しいフレームが届かないので、5 Hz で描き直して
    も同じ値を描き直すだけである。

    `_snapshot_for_display` ではなく `pipeline.snapshot()` を読む。表示ラッチ
    （FR-041）は「読んでいる最中に数字が止まっていてほしい」という要求のための
    時間依存の機構であり、止まった後の 1 枚を読むのにその時計を進める理由が無い。

    `show_parts=True` にする理由は HUD である。停止中は `_render_hud_fragment`
    が payload を publish しないため、映像上の HUD は数値を持たない。項目26（H3）
    の重複回避はライブ中にだけ意味を持つので、ここが部位別 4 つの唯一の受け皿に
    なる（Image の詳細タブと同じ扱い）。
    """
    snapshot = pipeline.snapshot()
    analysed = _has_analysed_frame(snapshot)
    if analysed:
        st.caption(t(LAST_FRAME_CAPTION))
    render_detail_panel(
        snapshot.analysis,
        empty_notice=(
            t(
                "このセッションでは、品質ゲートを満たすフレームがありませんでした。"
                "明るさ・顔の大きさ・向きを整えて `START` から取り直してください。"
            )
            if analysed
            else t(IDLE_DETAIL_NOTICE)
        ),
        show_parts=True,
        diagnostics=_detail_diagnostics(pipeline, snapshot),
    )


#: FR-043: `_session_demographic_caption` は動的な値を含むため、テンプレートへ
#: 切り出してから `t(TEMPLATE).format(...)` する（ADR-0010 (a)）。
_AGE_SPAN_SINGLE_TEMPLATE: Final[str] = "{median} 歳"
_AGE_SPAN_RANGE_TEMPLATE: Final[str] = "{median} 歳（真ん中50%: {low}〜{high} 歳）"
_SESSION_DEMOGRAPHIC_CAPTION_TEMPLATE: Final[str] = (
    "このセッションで表示できた参考年齢 {span} · 参考性別 {genders} · {count} 件"
)


def _session_demographic_caption(overview: DemographicOverview) -> str:
    """§9.6.1 と同じ材料を 1 行にする（保存物だけが知っている状態にしない）。

    範囲が 1 歳へ収まったセッションでは「31〜31 歳」と書かない。同じ数を 2 度
    並べても幅は伝わらない（`ui.report._session_demographic_section` と同じ作法）。
    参考性別は観測フレームのモデル性別確率の平均を「女性 X% ・ 男性 Y%」形式で
    表示する（2026-08-31 の指示）。
    """
    ages = overview.ages
    genders = format_gender_probabilities(overview.gender_probabilities)
    span = (
        t(_AGE_SPAN_SINGLE_TEMPLATE).format(median=ages.median)
        if ages.low == ages.high
        else t(_AGE_SPAN_RANGE_TEMPLATE).format(median=ages.median, low=ages.low, high=ages.high)
    )
    return t(_SESSION_DEMOGRAPHIC_CAPTION_TEMPLATE).format(
        span=span, genders=genders, count=f"{ages.sample_count:,}"
    )


#: まだ推定できていない間の 1 行。Image は「単一顔を検出できた写真でのみ」と
#: 言うが、ライブ映像では顔が入ってくるのを待っている状態なので、待てば出うる
#: ことを述べる（`ui.demographic_panel` がこの 1 文だけを入力ごとに受け取る）。
_DEMOGRAPHIC_PENDING_NOTICE: Final[str] = (
    "顔が 1 名だけ正面に近い向きで写っている間に、参考年齢・参考性別を推定します。"
)

#: 停止後の同じ 1 行。**できなかった内訳は述べない**——ゲートが働いた回と顔が
#: 写っていなかった回を区別させると、§10.11-2 が禁じる「内部判断を属性として
#: 提示する」ことに近づく。左右差へ影響しないことだけを添える（FR-024）。
_DEMOGRAPHIC_STOPPED_NOTICE: Final[str] = (
    "このセッションでは参考年齢・参考性別を表示できたフレームがありませんでした。"
    "左右差の分析結果には影響しません。"
)


def _render_demographic_area(
    pipeline: RealtimeLandmarkPipeline,
    *,
    snapshot: PipelineSnapshot,
    pending_notice: str,
) -> None:
    """参考年齢・参考性別の区画を、ライブでも停止後でも同じ 1 か所から描く。

    区画そのもの（囲み・来歴の 1 行・状態を見る順序・ゲートの適用）は
    `ui.demographic_panel` が Image と共有する。ここに残るのは snapshot と
    pipeline からの読み出しだけで、**表示の分岐は 1 つも持たない**——分岐を
    ここへ書き足すと、Image 側とまた食い違う。

    ライブと停止中で変わるのは、まだ出せていないときの 1 文だけである。2 か所へ
    分けると、停止中だけゲートや worker の状態を渡し忘れても気づけない。
    """
    overview = summarize_observations(pipeline.demographic_observations)
    render_demographic_panel(
        estimate=snapshot.demographic_estimate,
        gate_outcome=snapshot.age_gate_outcome,
        available=pipeline.has_demographic_worker,
        show_demographics=bool(st.session_state.get(DEMOGRAPHIC_ENABLED_STATE_KEY, True)),
        # `pending_notice` は関数引数なので、`t()` は module-level 定数を渡す
        # 呼び出し側（`_render_demographic_fragment` / `_render_stopped_demographic_panel`）
        # で先に適用済みである（`tests/ui/test_i18n.py` の AST 走査は module-level
        # 名しか解決できない）。
        pending_notice=pending_notice,
        # §9.6.1: セッションを通して表示できた範囲。Image は 1 枚しか持たない
        # ため、添えるものがない（`footnote=None`）。
        footnote=None if overview is None else _session_demographic_caption(overview),
    )


@st.fragment(run_every=DEMOGRAPHIC_REFRESH_INTERVAL)
def _render_demographic_fragment(pipeline: RealtimeLandmarkPipeline) -> None:
    """1 Hz で snapshot を読み、参考年齢・参考性別の区画を描く。

    FR-041: HUD・詳細結果と同じ `_snapshot_for_display` を通す。別の snapshot を
    読むと、HUD が固定文言を出している瞬間にここだけが数値を出しうる。
    """
    _render_demographic_area(
        pipeline,
        snapshot=_snapshot_for_display(pipeline),
        pending_notice=t(_DEMOGRAPHIC_PENDING_NOTICE),
    )


def _render_stopped_demographic_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """§5.1（2026-08-27）: `STOP` の後の参考年齢・参考性別。

    詳細結果と同じく、最後に解析したフレームの値を明示して出す。fragment に
    しないのと `pipeline.snapshot()` を読む理由も同じである
    （`_render_stopped_detail_panel` を参照）。

    出せていないときの 1 文だけをセッション向けに差し替える。**理由の内訳は
    述べない**——ゲートが働いた回と顔が写っていなかった回を利用者から区別
    させないためである（§10.11-2）。
    """
    snapshot = pipeline.snapshot()
    analysed = _has_analysed_frame(snapshot)
    if analysed:
        st.caption(t(LAST_FRAME_CAPTION))
    _render_demographic_area(
        pipeline,
        snapshot=snapshot,
        pending_notice=(t(_DEMOGRAPHIC_STOPPED_NOTICE) if analysed else t(IDLE_DEMOGRAPHIC_NOTICE)),
    )


def _render_detail_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """HUD が持たない計算過程・姿勢などの詳細結果。

    常に動画の下へ画面幅いっぱいで表示する（FR-032, AC-LAYOUT-01）。
    """
    _render_detail_fragment(pipeline)


def _render_export_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """FR-021 の要約と FR-017 のダウンロードを、同じタブへ順に置く。

    **bundle は 1 度だけ作る。** `export_bundle()` は lock の内側で記録を複製
    するので、要約用とダウンロード用に 2 度呼ぶと、その間に届いたフレームの分だけ
    両者がずれる（要約が語るセッションと、保存されるセッションが別物になる）。

    要約を 5 Hz の fragment ではなくここへ置く理由は `ui.summary_panel` にある
    ——最大 3,000 行の要約を毎秒 5 回計算する必要はない。
    """
    bundle = pipeline.export_bundle()
    records = bundle.records if bundle is not None else ()
    summary = summarize_session(records)
    render_session_summary(summary)

    # FR-039/§9.6.1: 参考年齢・参考性別と母集団は、どちらも **pipeline が溜めた
    # 「表示を許した推定」** から決める。`bundle.records` から作ると、年齢ゲートが
    # 隠した判断が年齢から復元できてしまう（§10.12-3、AC-BASELINE-05）。
    # **snapshot は 1 度だけ取る**（`bundle` と同じ理由）——2 度取ると、その間に
    # 届いた推定の分だけ「参考年齢の節」と「参照平均の母集団」がずれる。
    observations = pipeline.demographic_observations
    overview = summarize_observations(observations)
    # §9.6.1: 1 度も表示できなかったセッションでは節ごと出さず、ゲートを実際に
    # 呼んでいたときだけ固定文言を 1 行置く。呼んでいないセッション（顔が写って
    # いなかった等）へこの文を出すと、機能が失敗したように読める（FR-024）。
    gate_ran = summary.overall is not None and summary.overall.reference_age_rate is not None
    notice = "" if overview is not None or not gate_ran else t(SESSION_HIDDEN_NOTICE)

    def _report(include_figures: bool) -> str:
        # 2026-08-16: リアルタイムのレポートに顔写真は載せない。`include_figures`
        # は Image だけが使う——Camera 側は不変条件 1 の限定解除を必要としない。
        del include_figures
        return build_camera_report(
            summary=summary,
            bundle=bundle,
            series=trend_series(records),
            baseline=resolve_session_baseline(observations),
            overview=overview,
            notice=notice,
        )

    render_export_panel(
        bundle,
        heading=t("セッションデータのエクスポート"),
        warning=t("姿勢・左右差などの顔由来の派生データを含みます。"),
        key_prefix="download_session",
        # ライブセッションは記録を積み上げるため、件数を示してこの
        # ダウンロードが実際にどこまでを含むかを伝える。
        show_record_count=True,
        # FR-017 は記録が 1 件以上あるときにだけエクスポートを提示する。その
        # ためこのタブは開始直後に何も描かず、中身が真っ白になっていた。提示
        # 条件は変えずに、空白を「まだ何も無い」という説明へ置き換える。
        empty_notice=t(
            "まだ書き出せる記録がありません。"
            "カメラを開始し、品質ゲートを満たすフレームが得られると記録が貯まります。"
        ),
        report_builder=_report,
        report_mode="session",
        # 2026-08-16: リアルタイムのレポートは顔写真を持たない。切り替えも注意も
        # 出す対象が無い。
        report_has_figures=False,
    )


#: FR-043: 動的な値を含むため、`t()` へは f-string ではなくテンプレートを渡す。
_AUTO_TUNE_ZOOM_TEMPLATE: Final[str] = " · ズーム目標 {target_px}px"
_AUTO_TUNE_STATUS_TEMPLATE: Final[str] = "明るさ補正 ×{gain:.2f}{zoom} · 解析周期 {interval_ms} ms"


def _render_auto_tune_status(pipeline: RealtimeLandmarkPipeline) -> None:
    """§15.2: Advanced 設定に手動 UI は無い。AI が選んだ値を表示する。

    品質ゲートの閾値は固定の定数であり、調整可能であるかのようにここへ並べる
    ことは意図的に避けている（§14.3d）。列挙するどの項目も測定値へは影響しない。
    §8.6 の指標には調整できる係数が残っていないためである。
    """
    zoom_policy = pipeline.zoom_policy
    zoom_caption = (
        t(_AUTO_TUNE_ZOOM_TEMPLATE).format(target_px=zoom_policy.target_px)
        if zoom_policy is not None
        else ""
    )
    # §5.3.1: 現在値だけを 1 行で出し、何をどう調整しているかの説明は `help` へ。
    st.caption(
        t(_AUTO_TUNE_STATUS_TEMPLATE).format(
            gain=pipeline.brightness_gain,
            zoom=zoom_caption,
            interval_ms=pipeline.analysis_interval_ms,
        ),
        help=t(
            "明るさ・追跡ズーム・cadence は AI が自動調整します（FR-029, FR-030）。"
            "明るさ補正は landmark 検出にだけ使い、表示される映像は変わりません。"
            "品質ゲート閾値は固定既定値で、左右差の計算には調整できる係数がないため、"
            "自動調整が数値を動かすことはありません。"
        ),
    )


def _render_display_controls(pipeline: RealtimeLandmarkPipeline) -> None:
    """§15.1 の利用者向け設定を、そのまま pipeline へ書き込む。

    どれもスコアを変えない。正中線と背景は解析の後に描かれ、年齢・性別のトグルは
    元から左右差とは独立したパネルの表示可否を切り替えるだけで（FR-024）、
    平滑化時間（FR-040）が変えるのは表示の時間応答だけである（§8.7）。ここで
    読み、1 箇所で書き込むことで、その主張を検証可能な形に保つ。

    **見出しも caption も置かない（§5.3.1、項目 30）。** widget のラベル自体が
    「顔の中心線を表示」「映像上の HUD の位置」と何をするかを述べており、
    区画名を重ねると同じことを 2 度書くことになる。補足はすべて `help` にある。
    呼び出し元（`render_camera_section`）が `st.popover` の中でこれを呼ぶため、
    ポップオーバー自身が枠を持つ——ここで枠付きの囲みを重ねて描かない。
    """
    midline_column, demographic_column = st.columns(2)
    with midline_column:
        show_midline = st.toggle(
            t("顔の中心線を表示"),
            value=True,
            key="camera_midline_enabled",
            help=t("採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。"),
        )
    with demographic_column:
        # この pipeline が実際に保持している worker を条件にする。worker が
        # 無いのに機能が動いていると示すことがないようにするためである。
        show_demographics = st.toggle(
            t("年齢・性別の参考値を表示"),
            value=True,
            key=DEMOGRAPHIC_ENABLED_STATE_KEY,
            disabled=not pipeline.has_demographic_worker,
            help=t(
                "18〜75歳向けモデル推定の参考年齢・参考性別を最大1Hzで独立推論します。"
                "左右差の計算とは分離しており、失敗してもカメラ映像は止まりません。"
            ),
        )
        if not pipeline.has_demographic_worker:
            st.caption(t("年齢・性別モデルが未配置のため使用できません。"))
    pipeline.set_midline_enabled(show_midline)
    pipeline.set_demographic_enabled(show_demographics)

    segmenter_path = configured_segmenter_path()
    if segmenter_status(segmenter_path) is InstallStatus.READY:
        background_mode = st.radio(
            t("背景を置き換える（表示専用）"),
            options=list(_BACKGROUND_MODE_LABELS),
            format_func=_background_mode_label,
            key="camera_background_mode",
            horizontal=True,
            help=t("人物以外を単色にします。左右差・年齢性別の判定には使いません。"),
        )
        pipeline.set_background_mode(background_mode)
    else:
        st.caption(t("背景置換モデルが未配置のため、この機能は使用できません。"))

    st.divider()

    # 見出しは `render_smoothing_control` が自分で出す（`SMOOTHING_HEADING`）。
    # FR-040: panel は pipeline を知らない。選ばれた値をここで適用する。
    # `set_smoothing_tau_seconds` は同じ値なら何もしないので、5 Hz の再実行で
    # EMA が作り直されることはない（AC-SMOOTH-01）。
    choice = render_smoothing_control(
        tau_seconds=pipeline.smoothing_tau_seconds,
        analysis_interval_ms=pipeline.analysis_interval_ms,
        hold_seconds=pipeline.display_hold_seconds,
    )
    pipeline.set_smoothing_tau_seconds(choice.tau_seconds)
    pipeline.set_display_hold_seconds(choice.hold_seconds)

    st.divider()

    _render_hud_position_control()

    st.divider()

    _render_session_controls(pipeline)


def _hud_position_label(position: str) -> str:
    """`HUD_POSITION_OPTIONS`（`ui.hud`）の値（識別子・CSS 引き当て鍵）は動かさず、
    表示だけを翻訳する（ADR-0010 (c)）。

    訳は 3d 担当が `translations/_hud.py` へ登録済み（"左上" → "Top left" 等）。
    `t()` はリテラルか module-level 名しか受け取れないため（`tests/ui/test_i18n.py`
    の AST 走査）、`lambda position: t(position)` のように引数をそのまま渡す形は
    使えない——`_background_mode_label` と同じく分岐で明示する。
    """
    if position == "左上":
        return t("左上")
    if position == "右上":
        return t("右上")
    if position == "左下":
        return t("左下")
    if position == "右下":
        return t("右下")
    if position == "非表示":
        return t("非表示")
    return position


def _render_hud_position_control() -> None:
    """FR-032 のプリセット widget。表示設定ポップオーバーの中で描画する。

    `st.popover` は既定の `on_change="ignore"` のもとで開閉に関わらず中身を
    毎回実行するため、この widget は常に存在し、値も保たれる（項目 29）。
    項目26（H3）で映像の隣の見出しパネルとその配置プリセット（右・左・下）を
    削除したため、ここで描く radio は映像上 HUD の位置だけになる。
    """
    st.radio(
        t("映像上の HUD の位置"),
        options=HUD_POSITION_OPTIONS,
        format_func=_hud_position_label,
        key=HUD_POSITION_STATE_KEY,
        horizontal=True,
        help=t(
            "映像へ重ねる HUD の吹き出し"
            "（状態・平滑化時間・全体の対称度・全体のスコア・部位別4つ・"
            "参考年齢・参考性別・折れ線グラフ）の位置です。映像自体は加工しません。"
            "HUD は掴んで動かせます。プリセットを選び直すと元の角へ戻ります。"
        ),
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
    """停止・リセットの操作子。メイン画面から到達できる位置に置く（§5.1）。

    STOP ボタンの案内は置かない（項目 30）——映像の中に見えている。
    """
    controls = st.columns(2)
    if controls[0].button(t("結果・履歴・平滑化をリセット"), key="reset_analysis"):
        pipeline.reset_analysis()
        st.rerun()
    if controls[1].button(t("推論を停止してモデルを解放"), key="release_landmarker"):
        release_session_pipeline()
        st.rerun()


def render_camera_section() -> None:
    """リアルタイムのランドマーク表示を描画する。同意の後にのみ呼ぶこと。

    §5.1: 映像とその見出しを最初に置き、その位置を保つ。表示設定は映像直下の
    ポップオーバーへ畳み、詳細数値・参考年齢・エクスポートは下のタブへ入れ、
    一度に高さを占めるのが 1 つだけになるようにする。これにより表示が画面外へ
    押し出されない。タブの切り替え・ポップオーバーの開閉はどちらも表示専用で、
    測定値を変えず WebRTC の通信も中断しない（AC-LAYOUT-01）。
    """
    model_path = resolve_ready_landmarker()
    if model_path is None:
        return

    try:
        pipeline = _session_pipeline(model_path)
    except (LandmarkDetectionError, MetricConfigError) as error:
        # FR-043: `domain` / `adapters` の例外文言は UI 境界で `translate_message`
        # （`ui/i18n.py`）を通す。静的な文言だけが `translations/_errors.py` の
        # 表に載っており訳される（動的な文言は既知の穴として日本語のまま残る）。
        st.error(translate_message(str(error)))
        return

    fullscreen = st.toggle(
        t("映像を全画面表示"),
        key=FULLSCREEN_STATE_KEY,
        help=t(
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
            st.button(t("✕ 全画面を終了"), key="exit_fullscreen", on_click=_exit_fullscreen)

    # 項目26（H3）: 「ライブ解析」パネルの削除に伴い、映像はもう `st.columns`
    # で分割せず、画面幅いっぱいを使う。`key="camera_preview_frame"` は HUD の
    # CSS（`.st-key-camera_preview_frame`）が参照するため変えない。
    with st.container(border=True, key="camera_preview_frame"):
        context = webrtc_streamer(
            key=WEBRTC_KEY,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=LOOPBACK_RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            video_frame_callback=build_video_frame_callback(pipeline),
            video_html_attrs=VIDEO_HTML_ATTRS,
            async_processing=True,
        )

    # §14.3e/§5.1（項目 29、2026-08-29）: 表示設定は映像の直下へ畳んで置く。
    # **`playing` の分岐より前**に置くのは、開始前と停止中にも操作できることを
    # §5.1 が求めるためであり、擬似全画面の early return より前に置くのは、
    # 全画面中も widget を描画し続けて値を保つためである（全画面中はトリガーが
    # 映像 iframe に覆われて押せないが、値は失われない）。`on_change` は既定の
    # `"ignore"` のままにする——`"rerun"` や callback を渡すと開閉状態を追跡し、
    # 閉じている間は中身を実行しなくなるので、設定 widget が消えて pipeline へ
    # の `set_*` も走らなくなる。
    with st.popover(t("⚙️ 表示設定")):
        _render_display_controls(pipeline)

    playing = context.state.playing
    if not playing:
        # 不変条件 3: 明示的な同意の後にだけカメラ許可を求める導線を保つ。
        st.info(t("`START` を押すと、ブラウザがカメラの許可を求めます。"))
    else:
        # 項目26（H3）: 映像に隣接して数値を出すのは映像上の HUD だけである。この
        # fragment は payload の publish だけを行い、それ自体は何も描画しない。
        _render_hud_fragment(pipeline)

        if fullscreen:
            # 映像がビューポートを覆っている間、その下の要素へは到達できない。
            # 誰も見られないパネルを 5 Hz で再描画し続けないよう、まるごと省く。
            return

    # §5.1（2026-08-27）: **`STOP` の後もタブの名前・順序・数は変わらない。**
    # 以前は停止中だけ別のタブを組んでおり、押した瞬間に詳細結果・参考年齢が
    # 画面から消えていた。ここで変えてよいのは「各タブが何を読むか」だけである
    # ——停止中はライブの fragment ではなく、そのセッションで最後に解析した
    # フレームを読む区画へ差し替える。
    def _detail_tab() -> None:
        if playing:
            _render_detail_panel(pipeline)
        else:
            _render_stopped_detail_panel(pipeline)

    def _demographic_tab() -> None:
        # §5.1（2026-08-26）: Image と同じ位置・同じ名前の独立区画。左右差とは
        # 別の結果として扱う（不変条件 5）。
        if playing:
            _render_demographic_fragment(pipeline)
        else:
            _render_stopped_demographic_panel(pipeline)

    # §5.1, AC-LAYOUT-01: 映像の下はすべて画面幅いっぱいで一度に 1 つだけ表示し、
    # 上の表示がスクロールで手の届かない位置へ行かないようにする。**名前・順序・
    # アイコンは `ui.result_tabs` が Image と共有する**（2026-08-27）——ここで
    # 直接タブを組み立てていた間、共通のはずのラベルが実際にずれていた。
    # Camera は「左右対称との比較」を持たない（項目 33、FR-034、AC-SYMM-01）。
    render_result_tabs(
        ResultTabs(
            detail=_detail_tab,
            demographic=_demographic_tab,
            # レポートはセッション全体を対象とするので、再生状態で分岐しない。
            # 記録がまだ無い開始前は、パネル側が `empty_notice` で説明する。
            report=lambda: _render_export_panel(pipeline),
        )
    )
