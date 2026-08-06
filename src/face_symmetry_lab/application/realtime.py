"""リアルタイムのランドマーク use case。

1 インスタンスが 1 セッションに対応する。``process`` は WebRTC の受信スレッド
で動き、``snapshot`` は Streamlit の fragment が読む。両者が共有するのは lock
の内側の不変 snapshot だけであり、1 フレームの失敗が通信を止めることはない
（§14.3, §14.4, NFR-017）。

このモジュールは調整役に徹する。実際の仕事は責務ごとの協調オブジェクトが持つ。

* `AnalysisCadence` — 解析周期と表示キャッシュの TTL
* `PreviewCompositor` — 描画・背景置換・表示キャッシュ
* `CaptureBuffer` — §14.3g の一発取り込み
* `DemographicRelay` — 年齢・性別 worker との受け渡し
* `AutoTuneController` — §14.3c/d の再調整判断
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.analysis_cadence import (
    DEFAULT_ANALYSIS_INTERVAL_MS,
    DEFAULT_DISPLAY_TTL_MS,
    AnalysisCadence,
)
from face_symmetry_lab.application.analyzer import AnalysisOutcome, FrameAnalyzer
from face_symmetry_lab.application.auto_tune_controller import AutoTuneController, DisplayTuning
from face_symmetry_lab.application.demographic_relay import DemographicRelay
from face_symmetry_lab.application.frame_capture import CaptureBuffer, CapturedFrame
from face_symmetry_lab.application.ports import (
    DemographicSampler,
    LandmarkProvider,
    SegmentationProvider,
)
from face_symmetry_lab.application.preview_compositor import PreviewCompositor
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.domain.auto_tuning import (
    recommend_age_interval_ms,
    recommend_history_capacity,
)
from face_symmetry_lab.domain.export import SessionExport
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.landmarks import LandmarkFrame, TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.overlay import (
    DEFAULT_MIDLINE_COLOR,
    DEFAULT_POINT_COLOR,
    DEFAULT_POINT_RADIUS,
)
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy
from face_symmetry_lab.runtime.frame_scheduler import FrameScheduler

#: UI が復旧手段を提示すべき連続失敗回数（§14.4）。
ERROR_CIRCUIT_THRESHOLD = 3

#: 周期と TTL の既定値は `AnalysisCadence` が正本として持つ。ここから再輸出
#: するのは、既存の呼び出し側が `application.realtime` 越しに参照している
#: ためである。
__all__ = [
    "DEFAULT_ANALYSIS_INTERVAL_MS",
    "DEFAULT_DISPLAY_TTL_MS",
    "ERROR_CIRCUIT_THRESHOLD",
    "CapturedFrame",
    "PipelineSnapshot",
    "RealtimeLandmarkPipeline",
]


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """最新の解析済みフレームの不変ビュー。"""

    frame_id: int = 0
    status: TrackingStatus = TrackingStatus.NO_FACE
    face_count: int = 0
    landmark_count: int = 0
    #: cadence 制御が意図的に表示専用のまま通したフレーム数。
    throttled_frames: int = 0
    #: 1 枠しかない worker が埋まっていて弾かれたフレーム数。
    dropped_frames: int = 0
    consecutive_errors: int = 0
    processing_ms: float = 0.0
    #: §14.3f: `processing_ms` を段階別に分けた NFR-002 の診断値であり、
    #: 完全な内訳ではない（オーバーレイ描画、自動調整の記帳、recorder への
    #: 書き込みは含まない）。その段階がこのフレームで走らなかったときは 0.0
    #: になる（背景置換が OFF のときの `segmentation_ms` など）。
    landmark_ms: float = 0.0
    segmentation_ms: float = 0.0
    scoring_ms: float = 0.0
    #: analyzer が接続され、かつそのフレームが採点されたときだけ入る。
    analysis: AnalysisOutcome | None = None
    #: §14.3, FR-022〜024: `DemographicSampler` が別周期で生成する。
    demographic_estimate: DemographicEstimate | None = None

    @property
    def needs_recovery(self) -> bool:
        return self.consecutive_errors >= ERROR_CIRCUIT_THRESHOLD


class RealtimeLandmarkPipeline:
    """遅延を招くキューを作らずに、最新のプレビューフレームだけを抽出する。"""

    def __init__(
        self,
        *,
        landmarker: LandmarkProvider,
        analyzer: FrameAnalyzer | None = None,
        recorder: SessionRecorder | None = None,
        scheduler: FrameScheduler | None = None,
        segmenter: SegmentationProvider | None = None,
        demographic_worker: DemographicSampler | None = None,
        clock: Callable[[], float] = time.monotonic,
        analysis_interval_ms: int = DEFAULT_ANALYSIS_INTERVAL_MS,
        display_ttl_ms: int = DEFAULT_DISPLAY_TTL_MS,
        midline_indices: tuple[int, ...] = (),
        midline_enabled: bool = True,
        midline_color: tuple[int, int, int] = DEFAULT_MIDLINE_COLOR,
        point_color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
        point_radius: int = DEFAULT_POINT_RADIUS,
        brightness_gain: float = 1.0,
        background_mode: BackgroundMode = BackgroundMode.OFF,
        auto_tune_enabled: bool = False,
    ) -> None:
        self._landmarker = landmarker
        self._analyzer = analyzer
        self._recorder = recorder
        self.scheduler = scheduler if scheduler is not None else FrameScheduler()
        self._clock = clock
        self._started_at = clock()
        self._cadence = AnalysisCadence(
            interval_ms=analysis_interval_ms,
            display_ttl_ms=display_ttl_ms,
            clock=clock,
            started_at=self._started_at,
        )
        self._compositor = PreviewCompositor(
            segmenter=segmenter,
            clock=clock,
            midline_indices=midline_indices,
            midline_enabled=midline_enabled,
            midline_color=midline_color,
            point_color=point_color,
            point_radius=point_radius,
            brightness_gain=brightness_gain,
            background_mode=background_mode,
        )
        self._capture = CaptureBuffer()
        self._demographics = DemographicRelay(demographic_worker)
        #: §14.3c/d: 再調整の窓と間引きをすべて所有する。触るのは `_analyse`
        #: だけで、そこは `FrameScheduler` が直列化しているため、自前の lock を
        #: 必要としない。
        self._auto_tune = AutoTuneController(
            enabled=auto_tune_enabled,
            started_at=self._started_at,
        )
        self._frame_id = 0
        self._last_timestamp_ms = -1
        self._snapshot = PipelineSnapshot()
        self._closed = False
        self._state_lock = threading.Lock()

    # --- フレーム処理 ---------------------------------------------------------

    def process(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """1 フレームを解析し、ブラウザへ返す画像を返す。"""
        painted, _ = self.process_with_analysis(rgb_image)
        return painted

    def process_with_analysis(
        self,
        rgb_image: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], AnalysisOutcome | None]:
        """このプレビューフレーム自身に属する解析だけを返す。

        FR-013: 返すフレームへ描くのはランドマークと正中線だけである（約 1ms）。
        数値パネルを意図的に描かないのは、1 フレームあたり 18.6ms を要して遅延
        として表面化したためである。数値はすべて `snapshot` 経由で UI へ届く。
        描く幾何が無く背景置換も OFF のときは `rgb_image` 自身が返るので、
        呼び出し側は再エンコードを完全に省ける。

        §14.3b: 明るさ補正はランドマーク検出器へ渡すフレームにしか届かない
        （`_analyse` の内側）。そのため AI の検出補助は表示に現れず、
        `apply_brightness_gain` は毎フレームではなく解析周期（約 8Hz）で走る。
        """
        if self._closed:
            return rgb_image, None
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            self._compositor.clear_caches()
            self._record_error()
            return rgb_image, None

        if not self._cadence.admit():
            return self._draw_cached(rgb_image), None

        with self.scheduler.slot() as acquired:
            if not acquired:
                return self._draw_cached(rgb_image), None
            return self._analyse(rgb_image)

    def _draw_cached(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """間引いたフレームを、TTL の内側で直前の解析結果から描き直す。"""
        return self._compositor.draw_cached(
            rgb_image,
            ttl_seconds=self._cadence.display_ttl_seconds,
        )

    def _analyse(
        self,
        rgb_image: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], AnalysisOutcome | None]:
        started = self._clock()
        self._frame_id += 1
        timestamp_ms = max(
            int((started - self._started_at) * 1000),
            self._last_timestamp_ms + 1,
        )
        self._last_timestamp_ms = timestamp_ms

        landmark_ms = 0.0
        segmentation_ms = 0.0
        try:
            # §14.3b: 明るくするのは検出器向けだけで、表示に返すのは生の
            # `rgb_image` である。
            detection_input = self._compositor.brighten_for_detection(rgb_image)
            frame = self._landmarker.detect_video(
                detection_input,
                frame_id=self._frame_id,
                timestamp_ms=timestamp_ms,
            )
            landmark_ms = (self._clock() - started) * 1000.0
            self._compositor.remember_geometry(frame, at=started)
            # §14.3c: 計測は生フレームに対して行い、影響するのは *次の* フレーム
            # だけなので、描画との前後関係は自由に選べる。
            self._apply_display_tuning(self._auto_tune.display_tuning(rgb_image, frame, at=started))
            self._capture.store(rgb_image, frame)
            painted = self._compositor.paint(rgb_image, frame)
            painted, segmentation_ms = self._compositor.apply_background(
                rgb_image, painted, timestamp_ms, at=started
            )
            self._demographics.submit(rgb_image, frame)
        except Exception:
            # 1 フレームの失敗が通信を止めてはならない（NFR-017）。
            self._compositor.clear_geometry_cache()
            self._record_error(
                frame_id=self._frame_id,
                elapsed_ms=timestamp_ms,
                processing_ms=(self._clock() - started) * 1000.0,
            )
            return rgb_image, None

        scoring_ms = 0.0
        analysis: AnalysisOutcome | None = None
        if self._analyzer is not None:
            scoring_started = self._clock()
            try:
                analysis = self._analyzer.analyze(
                    frame,
                    at=started,
                    rgb_image=rgb_image,
                )
            except Exception:
                # 採点がプレビューを道連れにしてはならない。
                analysis = None
            scoring_ms = (self._clock() - scoring_started) * 1000.0

        demographic_sample = self._demographics.collect(timestamp_ms)

        finished = self._clock()
        processing_ms = (finished - started) * 1000.0
        self._record_to_session(
            frame,
            analysis,
            demographic_sample,
            elapsed_ms=timestamp_ms,
            processing_ms=processing_ms,
        )

        self._apply_cadence_tuning(
            self._auto_tune.cadence_tuning(
                processing_ms,
                at=finished,
                current_interval_ms=self.analysis_interval_ms,
            )
        )

        with self._state_lock:
            self._snapshot = PipelineSnapshot(
                frame_id=frame.frame_id,
                status=frame.status,
                face_count=frame.face_count,
                landmark_count=len(frame.points),
                throttled_frames=self._cadence.throttled_frames,
                dropped_frames=self.scheduler.dropped_frames,
                consecutive_errors=0,
                processing_ms=processing_ms,
                landmark_ms=landmark_ms,
                segmentation_ms=segmentation_ms,
                scoring_ms=scoring_ms,
                analysis=analysis,
                demographic_estimate=(
                    demographic_sample[0] if demographic_sample is not None else None
                ),
            )
        return painted, analysis

    def _record_to_session(
        self,
        frame: LandmarkFrame,
        analysis: AnalysisOutcome | None,
        demographic_sample: tuple[DemographicEstimate, int] | None,
        *,
        elapsed_ms: int,
        processing_ms: float,
    ) -> None:
        """エクスポート用の履歴へ 1 フレーム分を書く。

        記帳はカメラを生かし続けることより優先度が低いので、失敗しても握り潰す。
        """
        if self._recorder is None:
            return
        try:
            if analysis is None:
                self._recorder.append_error(
                    frame_id=frame.frame_id,
                    elapsed_ms=elapsed_ms,
                    processing_ms=processing_ms,
                )
                return
            self._recorder.append_analysis(
                frame,
                analysis,
                elapsed_ms=elapsed_ms,
                processing_ms=processing_ms,
            )
            if demographic_sample is not None:
                estimate, observed_elapsed_ms = demographic_sample
                self._recorder.attach_demographic_estimate(
                    estimate,
                    observed_elapsed_ms=observed_elapsed_ms,
                )
        except Exception:
            pass

    # --- 自動調整の適用 -------------------------------------------------------

    def _apply_display_tuning(self, tuning: DisplayTuning) -> None:
        """§14.3c の明るさ・ズーム推奨があれば取り込む。"""
        if tuning.brightness_gain is not None:
            self._compositor.set_brightness_gain(tuning.brightness_gain)
        if tuning.zoom_policy is not None:
            self._update_zoom_policy(tuning.zoom_policy)

    def _update_zoom_policy(self, policy: TrackingZoomPolicy) -> None:
        """ズーム方針を landmarker へ渡す。追跡していない実装なら何もしない。

        `LandmarkProvider` はズーム対応を要求しない（§14.3a は素の provider を
        包む）ため、`update_policy` を持たない landmarker はエラーではなく
        no-op として扱う。
        """
        update_zoom_policy = getattr(self._landmarker, "update_policy", None)
        if callable(update_zoom_policy):
            update_zoom_policy(policy)

    def _apply_cadence_tuning(self, new_interval_ms: int | None) -> None:
        """§14.3d の cadence 推奨があれば取り込む。

        `analysis_interval_ms`/`age_interval_ms`/`history_capacity` は採点に
        影響しない（§15.2）ため、config_revision・区間・EMA には一切触れない。
        動かすのはサンプリング周期、年齢 worker 自身の周期、そして recorder の
        非破壊的なリサイズだけである。
        """
        if new_interval_ms is None:
            return
        self._cadence.set_interval_ms(new_interval_ms)
        self._cadence.widen_display_ttl()
        if self._demographics.available:
            try:
                age_interval_ms = recommend_age_interval_ms(new_interval_ms)
            except Exception:
                pass
            else:
                self._demographics.set_interval_ms(age_interval_ms)
        if self._recorder is not None:
            try:
                capacity = recommend_history_capacity(new_interval_ms)
            except Exception:
                pass
            else:
                self._recorder.resize_history_capacity(capacity)

    # --- §14.3g 取り込み ------------------------------------------------------

    def request_capture(self) -> None:
        """§14.3g: 次に解析へ成功した単一顔フレームを保持する。"""
        self._capture.request()

    def take_capture(self) -> CapturedFrame | None:
        """保持したフレームを消費する。準備できていなければ None。"""
        return self._capture.take()

    @property
    def capture_pending(self) -> bool:
        """要求がまだ使えるフレームを待っている状態かどうか。"""
        return self._capture.pending

    def clear_capture(self) -> None:
        """保留中の要求と保持済みフレームを両方破棄する（§4.4, §10.4）。"""
        self._capture.clear()

    # --- 状態 -----------------------------------------------------------------

    def _record_error(
        self,
        *,
        frame_id: int | None = None,
        elapsed_ms: int | None = None,
        processing_ms: float | None = None,
    ) -> None:
        with self._state_lock:
            self._snapshot = replace(
                self._snapshot,
                status=TrackingStatus.ERROR,
                face_count=0,
                landmark_count=0,
                consecutive_errors=self._snapshot.consecutive_errors + 1,
                analysis=None,
            )
        if (
            self._recorder is not None
            and frame_id is not None
            and elapsed_ms is not None
            and processing_ms is not None
        ):
            with suppress(Exception):
                self._recorder.append_error(
                    frame_id=frame_id,
                    elapsed_ms=elapsed_ms,
                    processing_ms=processing_ms,
                )

    def snapshot(self) -> PipelineSnapshot:
        """UI スレッド向けに最新の不変結果を返す。"""
        throttled_frames = self._cadence.throttled_frames
        with self._state_lock:
            return replace(
                self._snapshot,
                throttled_frames=throttled_frames,
                dropped_frames=self.scheduler.dropped_frames,
            )

    def reset_analysis(self) -> None:
        """通信を止めずに、スコア・平滑化・ヒステリシスを破棄する。"""
        if self._analyzer is not None:
            self._analyzer.reset()
        if self._recorder is not None:
            self._recorder.reset()
        self.scheduler.reset()
        self._compositor.clear_caches()
        self._capture.clear()
        self._demographics.reset()
        self._cadence.reset()
        with self._state_lock:
            self._snapshot = PipelineSnapshot()

    def apply_advanced_settings(
        self,
        *,
        thresholds: QualityThresholds,
        tau_seconds: float,
        zoom_policy: TrackingZoomPolicy,
        history_capacity: int | None = None,
        age_interval_ms: int | None = None,
        analysis_interval_ms: int | None = None,
    ) -> None:
        """新しい Advanced 設定をまとめて適用する（§15.2）。

        これらの値に振る舞いが依存するすべての構成要素へ届く。品質ゲートと EMA
        は作り直され（古い連続数・平滑値は捨てられる）、追跡ズームの方針は
        まだ有効な直前の顔位置に対してその場で差し替えられ、recorder は再計算
        した設定 fingerprint とともに新しい区間を開始する。これにより、変更の
        前後のエクスポート記録が混同されることはない。``history_capacity`` は
        セッションバッファをリサイズし、``age_interval_ms`` と
        ``analysis_interval_ms`` は 2 つの独立したサンプリング周期を retune する。
        ``None`` のものは現在値を保つ。

        指標 bundle はこの設定に含まれない。§8.6 が調整可能な係数を残さなかった
        ためであり、この経路は明示的な利用者操作からのみ到達し、自動調整からは
        決して到達しない（AC-AUTOTUNE-02）。
        """
        if self._analyzer is not None:
            self._analyzer.update(thresholds=thresholds, tau_seconds=tau_seconds)
        self._update_zoom_policy(zoom_policy)
        if self._recorder is not None:
            self._recorder.reset(
                thresholds=thresholds,
                tau_seconds=tau_seconds,
                history_capacity=history_capacity,
                analysis_interval_ms=analysis_interval_ms,
            )
        if analysis_interval_ms is not None:
            self._cadence.set_interval_ms(analysis_interval_ms)
        if age_interval_ms is not None:
            self._demographics.set_interval_ms(age_interval_ms)
        self._compositor.clear_caches()
        self._capture.clear()
        self._demographics.reset()

    # --- 表示専用の設定 -------------------------------------------------------

    def set_midline_enabled(self, enabled: bool) -> None:
        """表示専用の中心線の設定をスレッドセーフに更新する。"""
        self._compositor.set_midline_enabled(enabled)

    def set_background_mode(self, mode: BackgroundMode) -> None:
        """表示専用の背景置換をスレッドセーフに更新する。"""
        self._compositor.set_background_mode(mode)

    def set_demographic_enabled(self, enabled: bool) -> None:
        """年齢・性別推論の表示可否を切り替える（§15.1、既定は ON）。"""
        self._demographics.set_enabled(enabled)

    # --- 読み取り専用のプロパティ ---------------------------------------------

    @property
    def brightness_gain(self) -> float:
        """検出器の入力にだけ適用されている現在の補正値（§14.3b）。"""
        return self._compositor.brightness_gain

    @property
    def zoom_policy(self) -> TrackingZoomPolicy | None:
        """landmarker が有効にしているズーム方針。追跡しない実装なら None。"""
        return getattr(self._landmarker, "policy", None)

    @property
    def analysis_interval_ms(self) -> int:
        """現在有効な解析周期（§15.2、§14.3d の自動調整の読み戻し）。"""
        return self._cadence.interval_ms

    @property
    def display_ttl_ms(self) -> int:
        """現在有効な幾何・mask キャッシュの TTL（§1.2(17) の読み戻し）。"""
        return self._cadence.display_ttl_ms

    @property
    def metric_config(self) -> MetricConfig | None:
        """analyzer が固定している bundle。analyzer 未接続なら None。"""
        return self._analyzer.config if self._analyzer is not None else None

    @property
    def has_demographic_worker(self) -> bool:
        """このセッションで年齢・性別推論が使えるかどうか。"""
        return self._demographics.available

    def export_bundle(self) -> SessionExport | None:
        """ライブ履歴を露出せずに、不変のエクスポート snapshot を返す。"""
        if self._recorder is None:
            return None
        return self._recorder.export_bundle()

    def close(self) -> None:
        """landmark runtime を解放し、フレームの受け付けを止める。"""
        if self._closed:
            return
        self._closed = True
        self._capture.clear()
        self._landmarker.close()
        self._compositor.close()
        self._demographics.close()
