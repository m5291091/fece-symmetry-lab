"""対称化画像の変形のための純粋な幾何（§8.6.4, FR-035）。

ここにあるのは、区分変形のうち画素に触れずに判断できる部分である。どの点が
参加し、どの点を固定し、いつ歪みが大きすぎて実行する価値が無いか、を決める。
Delaunay 三角形分割と三角形ごとの画素転送は `cv2` を必要とするため、
`application.ports.PiecewiseWarper` の背後の `adapters.imaging` に置く。

メッシュは画像の外周を固定する。ランドマークは顔を覆うが、耳・髪・首・背景は
覆わない。固定アンカーが無いと、それらの領域は三角形分割から完全に外れて穴に
なるか、たまたま届いた顔の三角形に引きずられるかのどちらかになる。四隅と辺の
中点を置くことで分割が敷き詰める枠ができ、アンカーと動いたランドマークの間は
滑らかに補間される。

ここには較正された指標パラメータは 1 つも無い。`MAX_WARP_DISPLACEMENT_RATIO`
は明るさのクランプと同種の破綻ガードであり、いつ拒否するかを決めるだけで、
顔の点数を決めるものではない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: §8.6.4: いずれかの点が両眼間隔のこの割合を超えて移動する場合は変形を拒否
#: する。実際の非対称はこれよりはるかに小さい（§8.6 の距離は通常は数パーセント）
#: ので、これが働くのは検出が破綻したときだけである。その状態で変形すると、
#: 対称化された画像ではなく人物ですらない画像ができてしまう。
MAX_WARP_DISPLACEMENT_RATIO: Final[float] = 0.25

#: これ（px²）より細い三角形は、安定したアフィン写像を定義できない。
MIN_TRIANGLE_AREA_PX: Final[float] = 1e-3

#: 四隅と辺の中点（§8.6.4）。
BORDER_ANCHOR_COUNT: Final[int] = 8


class WarpMeshError(ValueError):
    """要求された変形を構築できない、または実行が安全でない。"""


@dataclass(frozen=True, slots=True)
class WarpMesh:
    """対応付けられた変形元・変形先の点集合。外周アンカーを含む。

    `source[i]` が `target[i]` へ変形される。末尾 `BORDER_ANCHOR_COUNT` 件は
    固定された画像外周であり、両者で同一である。そのため `target` を三角形分割
    する利用側はフレーム全体を覆える。
    """

    source: NDArray[np.float64]
    target: NDArray[np.float64]
    width: int
    height: int
    #: 先頭から何件が外周ではなくランドマーク由来か。
    landmark_count: int

    @property
    def anchor_indices(self) -> tuple[int, ...]:
        """固定された外周点の index。"""
        return tuple(range(self.landmark_count, len(self.source)))

    @property
    def max_displacement_px(self) -> float:
        """いずれかの点が移動する最大距離（画素）。"""
        if len(self.source) == 0:
            return 0.0
        return float(np.max(np.linalg.norm(self.target - self.source, axis=1)))


def border_anchors(width: int, height: int) -> NDArray[np.float64]:
    """画像の四隅と 4 辺の中点。

    変形元の座標系で返す。呼び出し側は同じ配列を変形先にも使い、それによって
    これらの点が固定される（§8.6.4）。
    """
    if width <= 0 or height <= 0:
        raise WarpMeshError("画像サイズは正の値である必要があります。")
    right = float(width - 1)
    bottom = float(height - 1)
    middle_x = right / 2.0
    middle_y = bottom / 2.0
    return np.array(
        [
            [0.0, 0.0],
            [middle_x, 0.0],
            [right, 0.0],
            [right, middle_y],
            [right, bottom],
            [middle_x, bottom],
            [0.0, bottom],
            [0.0, middle_y],
        ],
        dtype=np.float64,
    )


def triangle_area(points: NDArray[np.float64]) -> float:
    """(3, 2) 配列で与えられた三角形 1 つの面積の絶対値。"""
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (3, 2):
        raise WarpMeshError("三角形は (3, 2) の配列である必要があります。")
    first = array[1] - array[0]
    second = array[2] - array[0]
    return float(abs(first[0] * second[1] - first[1] * second[0]) / 2.0)


def is_degenerate_triangle(points: NDArray[np.float64]) -> bool:
    """三角形が細すぎて安定したアフィン変換を定義できないかどうか。"""
    return triangle_area(points) < MIN_TRIANGLE_AREA_PX


def build_warp_mesh(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    width: int,
    height: int,
    interocular_px: float,
    max_displacement_ratio: float = MAX_WARP_DISPLACEMENT_RATIO,
) -> WarpMesh:
    """実測点と対称化点を対応付け、画像の外周を固定する。

    両方の集合をフレーム内へクランプする。三角形分割は外接矩形の外側の頂点を
    受け付けられず、また端から 1 画素外にあるランドマークは検出の副産物で
    あって失敗の理由ではないためである。

    変位ガードが働いたときは `WarpMeshError` を送出する。その先では区分写像が
    自分自身へ折り重なり、出力はもはや顔ではなくなる。壊れた結果を見せるより、
    拒否する方が誠実である。
    """
    source_array = _as_points(source, "変形元")
    target_array = _as_points(target, "変形先")
    if source_array.shape != target_array.shape:
        raise WarpMeshError("変形元と変形先の点数が一致しません。")
    if len(source_array) == 0:
        raise WarpMeshError("変形にはランドマークが必要です。")
    if width <= 0 or height <= 0:
        raise WarpMeshError("画像サイズは正の値である必要があります。")
    if not np.isfinite(interocular_px) or interocular_px <= 0:
        raise WarpMeshError("両眼間隔は正の有限値である必要があります。")
    if not np.isfinite(max_displacement_ratio) or max_displacement_ratio <= 0:
        raise WarpMeshError("変位の上限比は正の有限値である必要があります。")

    displacement = float(np.max(np.linalg.norm(target_array - source_array, axis=1)))
    limit = max_displacement_ratio * interocular_px
    if displacement > limit:
        raise WarpMeshError(
            "変形量が大きすぎるため画像を生成しません"
            f"（最大 {displacement:.1f}px > 上限 {limit:.1f}px）。"
        )

    clamped_source = _clamp(source_array, width, height)
    clamped_target = _clamp(target_array, width, height)
    anchors = border_anchors(width, height)
    return WarpMesh(
        source=np.vstack([clamped_source, anchors]),
        target=np.vstack([clamped_target, anchors]),
        width=width,
        height=height,
        landmark_count=len(clamped_source),
    )


def _as_points(points: NDArray[np.float64], label: str) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise WarpMeshError(f"{label}の座標は (N, 2) の配列である必要があります。")
    if not np.isfinite(array).all():
        raise WarpMeshError(f"{label}の座標は有限値である必要があります。")
    return array


def _clamp(points: NDArray[np.float64], width: int, height: int) -> NDArray[np.float64]:
    clamped = points.copy()
    np.clip(clamped[:, 0], 0.0, float(width - 1), out=clamped[:, 0])
    np.clip(clamped[:, 1], 0.0, float(height - 1), out=clamped[:, 1])
    return clamped
