"""一発の対称化 use case（§14.3g, FR-034）。

`application.still_image` と同様、定義上まったく状態を持たない。EMA も履歴も
timestamp も recorder も無い。1 回の呼び出しが 1 フレームについての 1 つの問い
に答えるだけなので、リアルタイムの周期に加わることはなく、実行中セッションの
平滑化や比較区間を乱すこともない。

`RealtimeLandmarkPipeline._analyse` へ意図的に組み込んでいない。オーバーレイは
全解像度の再描画であり 8 Hz のループに居場所が無いうえ、この機能が本来求められて
いるのは静止した結果だからである。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.ports import PiecewiseWarper
from face_symmetry_lab.domain.landmarks import LandmarkError, LandmarkFrame, to_pixel_array
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.overlay import (
    DEFAULT_DEVIATION_COLOR,
    DEFAULT_POINT_COLOR,
    DEFAULT_POINT_RADIUS,
    DEFAULT_SYMMETRIZED_COLOR,
    draw_symmetry_deviation,
)
from face_symmetry_lab.domain.symmetrization import (
    SymmetrizationError,
    SymmetrizationResult,
    symmetrize_landmarks,
)
from face_symmetry_lab.domain.warp_mesh import WarpMeshError, build_warp_mesh


@dataclass(frozen=True, slots=True)
class SymmetrizationView:
    """1 回の対称化と、それを表示するための画像。"""

    result: SymmetrizationResult
    overlay: NDArray[np.uint8]
    #: §8.6.4 の変形画像。要求されなかったか生成できなかったときは None。
    #: これが無くても `overlay` は無効にならない（fail-soft）。
    warped: NDArray[np.uint8] | None = None
    #: 要求されたのに `warped` が無い理由。UI が黙って 2 枚のうち 1 枚だけを
    #: 見せるのではなく、説明できるようにするためである。
    warp_unavailable_reason: str | None = None

    @property
    def symmetrized_point_count(self) -> int:
        return len(self.result.symmetrized_indices)

    @property
    def mean_deviation(self) -> float | None:
        return self.result.mean_deviation

    @property
    def max_deviation(self) -> float | None:
        return self.result.max_deviation


def symmetrize_frame(
    rgb_image: NDArray[np.uint8],
    frame: LandmarkFrame,
    *,
    config: MetricConfig,
    warper: PiecewiseWarper | None = None,
    measured_color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
    symmetrized_color: tuple[int, int, int] = DEFAULT_SYMMETRIZED_COLOR,
    deviation_color: tuple[int, int, int] = DEFAULT_DEVIATION_COLOR,
    radius: int = DEFAULT_POINT_RADIUS,
) -> SymmetrizationView:
    """フレーム 1 枚のランドマークを対称化し、ズレのオーバーレイを描く。

    フレームがそもそもこの構成を担えないときは、部分的な表示を返さず
    `SymmetrizationError` を送出する。複数顔や寸法の無いフレームには折り返す
    べき正中線が 1 本に定まらず、2 人が写った画像に 1 人分の折り返しを描けば
    別人に帰属させてしまうためである（FR-008）。

    ``warper`` を渡すと §8.6.4 の変形画像も生成する。これは任意であり fail-soft
    である。主たる結果は点のオーバーレイなので、変位ガードがメッシュを拒否した
    場合や画像化に失敗した場合も、パネル全体を落とさずに理由を記録して残す。
    """
    if not frame.has_single_face:
        raise SymmetrizationError("単一顔のフレームだけを対称化できます。")
    try:
        points = to_pixel_array(frame)
    except LandmarkError as error:
        raise SymmetrizationError(str(error)) from error

    result = symmetrize_landmarks(points, config)
    try:
        overlay = draw_symmetry_deviation(
            rgb_image,
            measured=result.measured,
            symmetrized=result.symmetrized,
            indices=result.symmetrized_indices,
            measured_color=measured_color,
            symmetrized_color=symmetrized_color,
            deviation_color=deviation_color,
            radius=radius,
        )
    except LandmarkError as error:
        raise SymmetrizationError(str(error)) from error

    warped, reason = _warp(rgb_image, result, warper)
    return SymmetrizationView(
        result=result,
        overlay=overlay,
        warped=warped,
        warp_unavailable_reason=reason,
    )


def _warp(
    rgb_image: NDArray[np.uint8],
    result: SymmetrizationResult,
    warper: PiecewiseWarper | None,
) -> tuple[NDArray[np.uint8] | None, str | None]:
    """§8.6.4 の変形画像、または (None, 理由) を返す。例外は送出しない。

    メッシュはこの同じ呼び出しが生成した対称化点から構築する。したがって変形は
    独自の幾何を一切追加せず、画素の転送だけを行う。
    """
    if warper is None:
        return None, None
    height, width = rgb_image.shape[:2]
    try:
        mesh = build_warp_mesh(
            result.measured,
            result.symmetrized,
            width=width,
            height=height,
            interocular_px=result.interocular_px,
        )
    except WarpMeshError as error:
        return None, str(error)
    try:
        return warper.warp(rgb_image, mesh), None
    except Exception as error:
        # 画像化の失敗は、この画像の失敗であって解析の失敗ではない。
        return None, f"画像を生成できませんでした: {error}"
