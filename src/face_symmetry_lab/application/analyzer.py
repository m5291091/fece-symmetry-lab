"""1 フレーム分の解析の調整役（§14.2 `application.analyzer`）。

順序は「品質ゲート → 左右差距離 → 平滑化」である。analyzer は Streamlit の
状態を持たず、何も描画せず、モデルを取得することもない。1 インスタンスが
1 セッションに対応する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.quality import QualityGate, QualityThresholds
from face_symmetry_lab.domain.capture_quality import exposure_warnings
from face_symmetry_lab.domain.frame_adjustments import FrameAdjustmentError
from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    to_pixel_array,
    to_pixel_array_3d,
)
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig
from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.smoothing import (
    DEFAULT_TAU_SECONDS,
    TimeBasedEma,
    validate_tau_seconds,
)
from face_symmetry_lab.domain.symmetry import (
    SymmetryError,
    SymmetryResult,
    compute_symmetry,
)
from face_symmetry_lab.domain.symmetry_3d import (
    Symmetry3DError,
    Symmetry3DResult,
    compute_symmetry_3d,
)

#: §8.8: 顔がこの時間だけ不在になったら、フィルタをリセットする。
FACE_LOST_RESET_SECONDS: float = 0.5


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """フレーム 1 枚を解析した結果。不変である。"""

    frame_id: int
    state: AnalysisState
    symmetry: SymmetryResult | None
    #: §8.7 の EMA を掛けた総合 *距離*。部位別の距離は生値のままである。
    overall_distance_smooth: float | None
    pose: FacePose | None
    face_area_ratio: float | None
    metric_version: str
    landmark_map_version: str
    #: §8.5.1 / §8.6.5 の主指標（`symmetry-3d/1.0.0`, FR-036）。表示は
    #: `ui.readings.resolve_overall_distance` / `resolve_part_distance_3d` が、
    #: エクスポートは `application.session` が `distance3d_*_raw` 列としてここから
    #: 読む。
    symmetry_3d: Symmetry3DResult | None = None
    #: `symmetry_3d.overall.distance`（`d_3d`）へ §8.7 と同じ時間ベース EMA を
    #: 独立に掛けたもの。`overall_distance_smooth` とは別の EMA インスタンスで
    #: あり、値も共有しない。
    overall_distance_3d_smooth: float | None = None
    #: §8.7 改訂（H1）: `symmetry_3d` の部位別距離 `d_3d,p` へ、`d_3d` と同じ
    #: tau・同じ式の EMA を部位ごとに独立して掛けたもの。キーは
    #: `domain.metrics.PART_NAMES`。`symmetry_3d` 自体が欠測のフレームは
    #: `None`（コンテナごと欠測）。`symmetry_3d` は成立していても特定の部位だけ
    #: 欠測のフレームでは、その部位のキーだけ `None` になり、EMA も更新しない
    #: （0 で埋めない、不変条件 11）。表示は §8.6.3 のとおりこの平滑値を主表示
    #: する。エクスポートへは接続しない（`distance3d_*_raw` は生値のまま）。
    part_distances_3d_smooth: dict[str, float | None] | None = None

    @property
    def overall_distance_raw(self) -> float | None:
        return self.symmetry.overall if self.symmetry is not None else None


def _new_part_emas(tau_seconds: float) -> dict[str, TimeBasedEma]:
    """§8.7 改訂（H1）: 部位ごとに独立した `TimeBasedEma` の集合を作る。

    `d_3d` 用の EMA と同じ `tau`・同じ式であり、新しい自由パラメータは導入しない
    （§8.6 の原則）。`__init__` と `update()`（Advanced 設定の差し替え）の両方が
    「新しく作ってから差し替える」という同じ規約に従うため、ここへ集約する。
    """
    return {name: TimeBasedEma(tau_seconds=tau_seconds) for name in PART_NAMES}


class FrameAnalyzer:
    """ランドマークフレーム 1 枚を、状態・距離・平滑化した総合値へ変換する。"""

    def __init__(
        self,
        *,
        config: MetricConfig,
        config_3d: MetricConfig,
        thresholds: QualityThresholds | None = None,
        tau_seconds: float = DEFAULT_TAU_SECONDS,
    ) -> None:
        self._config = config
        self._config_3d = config_3d
        self._gate = QualityGate(thresholds)
        # §8.7 改訂（H1）: 部位ごとに独立したインスタンス。`d_3d` 用の `_ema_3d`
        # とも値を共有しない（`_replace_emas` が 3 種類をまとめて作る）。
        self._replace_emas(tau_seconds)
        self._lost_since: float | None = None

    @property
    def config(self) -> MetricConfig:
        return self._config

    @property
    def config_3d(self) -> MetricConfig:
        return self._config_3d

    @property
    def thresholds(self) -> QualityThresholds:
        return self._gate.thresholds

    @property
    def tau_seconds(self) -> float:
        """EMA が現在有効にしている時定数（§14.3d の自動調整の読み戻し）。"""
        return self._ema.tau_seconds

    def update(
        self,
        *,
        thresholds: QualityThresholds,
        tau_seconds: float,
    ) -> None:
        """閾値と tau をまとめて差し替え、ゲートと EMA の履歴をリセットする（§15.2）。

        Advanced 設定の契約に従い、新しく作ってから差し替える。変更をまたいで
        古いゲートの連続数や平滑値が漏れてはならないためである。指標 bundle は
        意図的に差し替え不可にしている。§8.6 が調整可能な係数を残さなかったので、
        セッションの生存期間中は固定である。
        """
        self._gate = QualityGate(thresholds)
        self._replace_emas(tau_seconds)
        self._lost_since = None

    def set_tau_seconds(self, tau_seconds: float) -> None:
        """FR-040: 平滑化の時定数だけを差し替える（§8.7、ADR-0007 (b)）。

        `update()` との違いは、**品質ゲートに触れないこと**である。`tau` は表示の
        時間応答だけを決め、採点条件を動かさない。`update()` を流用して
        `QualityGate` ごと作り直すと、平滑化時間を選び直すたびに `Status` が
        `HOLD` へ戻り、`valid_frames_to_ok` 分のフレームのあいだ数値が `—` になる
        ——値を選び比べるという操作そのものが成立しなくなる。

        EMA は §8.8 のとおり作り直す。古い時定数で均した `ema_(t-1)` は新しい
        `tau` の定義（直近 `tau` 秒あたりの平均）を満たさないため、区間をまたい
        で持ち越さない。値域の検証は `domain.smoothing` が持ち、範囲外は clamp
        せず例外にする（呼び出し側が直前の設定を保つ）。
        """
        self._replace_emas(validate_tau_seconds(tau_seconds))

    def _replace_emas(self, tau_seconds: float) -> None:
        """全体・3D 総合・部位別の EMA を、同じ `tau` で作り直す。

        3 種類すべてを 1 か所で作る。`__init__`・`update()`・`set_tau_seconds()`
        の 3 経路が別々に列挙していると、EMA を 1 本足したときに更新漏れが起き、
        その 1 本だけ古い時定数のまま動き続ける。
        """
        self._ema = TimeBasedEma(tau_seconds=tau_seconds)
        self._ema_3d = TimeBasedEma(tau_seconds=tau_seconds)
        self._ema_3d_parts = _new_part_emas(tau_seconds)

    def reset(self) -> None:
        """履歴・平滑化・ヒステリシスを破棄する（§4.4, §8.8）。"""
        self._gate.reset()
        self._ema.reset()
        self._ema_3d.reset()
        # §8.7 改訂（H1）: 部位別 EMA も全体値とまったく同じタイミングでリセット
        # する。片方だけ漏れると、リセット後に古い部位別の値が新しい区間へ
        # 混ざる（W3 と同じ注意）。
        for part_ema in self._ema_3d_parts.values():
            part_ema.reset()
        self._lost_since = None

    def analyze(
        self,
        frame: LandmarkFrame,
        *,
        at: float,
        rgb_image: NDArray[np.uint8] | None = None,
    ) -> AnalysisOutcome:
        state = self._gate.update(frame)
        self._track_face_loss(state.status, at)
        quality_ok = state.status is Status.OK

        symmetry: SymmetryResult | None = None
        if quality_ok:
            try:
                symmetry = compute_symmetry(to_pixel_array(frame), self._config)
            except (SymmetryError, LandmarkError):
                # 指標の失敗は、このフレームの失敗であって通信の失敗ではない。
                self._gate.reset()
                self._ema.reset()
                state = AnalysisState(
                    status=Status.ERROR,
                    reason=ReasonCode.LOW_QUALITY,
                    reason_codes=(ReasonCode.LOW_QUALITY,),
                )
                symmetry = None

        smooth: float | None = None
        if symmetry is not None and symmetry.overall is not None:
            smooth = self._ema.update(symmetry.overall, at=at)

        # symmetry-3d/1.0.0（§8.5.1, §8.6.5, FR-036）は主指標であり、表示と
        # エクスポートはここから読む。それでも 2D の失敗処理（Status.ERROR にして
        # ゲートと EMA をリセットする）をここへコピーしない —— 3D の失敗はこの
        # フレームの 3D 側だけを欠測にし、state・品質ゲート・2D の距離・EMA の
        # いずれも変えない（不変条件 12 の失敗分離）。`quality_ok` は品質ゲート
        # 直後の判定であり、2D の算出が後から失敗して `state` が ERROR に変わっ
        # ても、3D はそれに引きずられない。
        symmetry_3d: Symmetry3DResult | None = None
        if quality_ok:
            try:
                symmetry_3d = compute_symmetry_3d(to_pixel_array_3d(frame), self._config_3d)
            except (Symmetry3DError, LandmarkError):
                symmetry_3d = None

        smooth_3d: float | None = None
        if symmetry_3d is not None and symmetry_3d.overall is not None:
            smooth_3d = self._ema_3d.update(symmetry_3d.overall.distance, at=at)

        # §8.7 改訂（H1）: `d_3d,p` にも部位ごとに独立した同じ時間ベース EMA を
        # 適用する。`symmetry_3d` 自体が欠測のフレームはコンテナごと欠測にする
        # （`overall_distance_3d_smooth` と同じ「更新しない・0 で埋めない」規約）。
        # `symmetry_3d` が成立していても特定の部位だけ欠測の場合は、その部位の
        # EMA だけ更新をスキップし、キーは `None` を返す（不変条件 11）。他の
        # 部位の EMA は通常どおり動き続ける。
        smooth_3d_parts: dict[str, float | None] | None = None
        if symmetry_3d is not None:
            smooth_3d_parts = {}
            for name in PART_NAMES:
                raw_part = symmetry_3d.distance(name)
                smooth_3d_parts[name] = (
                    None if raw_part is None else self._ema_3d_parts[name].update(raw_part, at=at)
                )

        # §6.2/§6.3/§8.2: 非ブロッキック照度警告。採点・品質ゲート・検出には
        # 影響しない。品質ゲートの状態に関わらず算出する（NO_FACE でも「暗い」
        # と言えるように）。失敗分離（不変条件 12）のため try/except で包み、
        # 失敗時は警告なし（`()`）。`FrameAdjustmentError` は `mean_luminance`
        # が不正 shape 等で送出する例外で、`exposure_warnings` が外へ伝播させる。
        warning_codes: tuple[ReasonCode, ...] = ()
        if rgb_image is not None:
            try:
                warning_codes = exposure_warnings(rgb_image)
            except FrameAdjustmentError:
                warning_codes = ()  # 失敗分離（不変条件 12）: 照度判定の失敗は分析を止めない
        if warning_codes:
            state = state.model_copy(update={"warning_codes": warning_codes})

        return AnalysisOutcome(
            frame_id=frame.frame_id,
            state=state,
            symmetry=symmetry,
            overall_distance_smooth=smooth,
            pose=frame.pose,
            face_area_ratio=frame.face_area_ratio,
            metric_version=self._config.metric_version,
            landmark_map_version=self._config.landmark_map_version,
            symmetry_3d=symmetry_3d,
            overall_distance_3d_smooth=smooth_3d,
            part_distances_3d_smooth=smooth_3d_parts,
        )

    def _track_face_loss(self, status: Status, at: float) -> None:
        if status not in {Status.NO_FACE, Status.MULTI_FACE}:
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = at
        elif at - self._lost_since >= FACE_LOST_RESET_SECONDS:
            self._ema.reset()
            self._ema_3d.reset()
            # §8.7 改訂（H1）: 顔喪失リセットも部位別 EMA まで含めて全体値と同じ
            # ように扱う。
            for part_ema in self._ema_3d_parts.values():
                part_ema.reset()
