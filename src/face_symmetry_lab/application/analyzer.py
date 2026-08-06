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
from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    to_pixel_array,
)
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.skin_quality import (
    MEDIAPIPE_CHEEK_PAIRS,
    SkinQualityError,
    SkinQualityResult,
    compute_skin_quality,
)
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS, TimeBasedEma
from face_symmetry_lab.domain.symmetry import (
    SymmetryError,
    SymmetryResult,
    compute_symmetry,
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
    skin_quality: SkinQualityResult | None = None

    @property
    def overall_distance_raw(self) -> float | None:
        return self.symmetry.overall if self.symmetry is not None else None


class FrameAnalyzer:
    """ランドマークフレーム 1 枚を、状態・距離・平滑化した総合値へ変換する。"""

    def __init__(
        self,
        *,
        config: MetricConfig,
        thresholds: QualityThresholds | None = None,
        tau_seconds: float = DEFAULT_TAU_SECONDS,
        skin_pairs: tuple[tuple[int, int], ...] = MEDIAPIPE_CHEEK_PAIRS,
    ) -> None:
        self._config = config
        self._gate = QualityGate(thresholds)
        self._ema = TimeBasedEma(tau_seconds=tau_seconds)
        self._lost_since: float | None = None
        self._skin_pairs = skin_pairs

    @property
    def config(self) -> MetricConfig:
        return self._config

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
        self._ema = TimeBasedEma(tau_seconds=tau_seconds)
        self._lost_since = None

    def reset(self) -> None:
        """履歴・平滑化・ヒステリシスを破棄する（§4.4, §8.8）。"""
        self._gate.reset()
        self._ema.reset()
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

        symmetry: SymmetryResult | None = None
        if state.status is Status.OK:
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

        skin_quality: SkinQualityResult | None = None
        if state.status is Status.OK and rgb_image is not None:
            try:
                skin_quality = compute_skin_quality(
                    rgb_image,
                    frame,
                    pairs=self._skin_pairs,
                )
            except SkinQualityError:
                # 撮影条件で補正した参考値は独立している（§8.6.1）。
                skin_quality = None

        return AnalysisOutcome(
            frame_id=frame.frame_id,
            state=state,
            symmetry=symmetry,
            overall_distance_smooth=smooth,
            pose=frame.pose,
            face_area_ratio=frame.face_area_ratio,
            metric_version=self._config.metric_version,
            landmark_map_version=self._config.landmark_map_version,
            skin_quality=skin_quality,
        )

    def _track_face_loss(self, status: Status, at: float) -> None:
        if status not in {Status.NO_FACE, Status.MULTI_FACE}:
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = at
        elif at - self._lost_since >= FACE_LOST_RESET_SECONDS:
            self._ema.reset()
