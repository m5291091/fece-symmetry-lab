"""対称化したランドマーク位置と、そのズレ（§8.6.2, FR-034）。

顔を正中線で折り返すと、版付きの対応 `(l, r)` ごとに、左右が一致していた場合
に両側が占めるはずの位置が 1 つ求まる。

    mid_l = (left_l + mirror(right_r)) / 2
    mid_r = mirror(mid_l)

正中点は同じ式を自分自身へ適用したもの `(p + mirror(p)) / 2` であり、これは
推定した直線への直交射影そのものである。したがって 1 つの規則で両方の場合を
覆える。対応が定義されていないランドマーク（478 点の大半）は、推測せずに実測
位置のまま残す。

ここで新しい指標を導入することはない。実測点から対称化位置までの距離は、§8.5
の `||left_i - mirror(right_i)||` のちょうど半分である。つまりこのモジュールは
既存の測定値を *見せる* 手段であって、2 つ目の測定ではない。ここには順位付けも
採点も補正も無い。「対称化」は幾何的な構成を指す語であり、目標でも改善でもない
（§5.3, §10.7）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.geometry import (
    GeometryError,
    Midline,
    fit_midline,
    normalization_transform,
    reflect_across_midline,
)
from face_symmetry_lab.domain.metrics import MetricConfig


class SymmetrizationError(ValueError):
    """入力をそもそも対称化できない。"""


@dataclass(frozen=True, slots=True)
class SymmetrizationResult:
    """顔 1 つ分の実測位置と対称化位置。座標系は入力と同じ。"""

    #: 入力の点をそのまま保持する。呼び出し側が 1 つの結果から両方を描ける。
    measured: NDArray[np.float64]
    #: `measured` と同じ形。`symmetrized_indices` に含まれない index はそのまま
    #: 引き継ぎ、補間は行わない。
    symmetrized: NDArray[np.float64]
    #: 実際に動いた index。使用できたペアと正中点の集合である。
    symmetrized_indices: tuple[int, ...]
    #: index -> 実測位置と対称化位置の距離。単位は両眼間隔（§8.3）なので、
    #: §8.5 の距離と直接比較できる。
    deviations: dict[int, float]
    #: 推定した正中線。推定を行った正規化フレームの座標で持つ。
    midline: Midline
    #: 入力画素での両眼間隔。§8.3 の正規化スケールである。§8.6.4 の変位ガードが
    #: ズレを画素へ戻すために必要とする。
    interocular_px: float
    metric_version: str
    landmark_map_version: str

    @property
    def mean_deviation(self) -> float | None:
        """正規化ズレの平均。対称化した点が無いときは None。"""
        if not self.deviations:
            return None
        return float(np.mean(list(self.deviations.values())))

    @property
    def max_deviation(self) -> float | None:
        """正規化ズレの最大値。対称化した点が無いときは None。"""
        if not self.deviations:
            return None
        return float(max(self.deviations.values()))


def symmetrize_landmarks(
    points: NDArray[np.float64],
    config: MetricConfig,
) -> SymmetrizationResult:
    """顔 1 つ分のランドマークに対応する、対称化された点群を構築する。"""
    try:
        transform = normalization_transform(
            points,
            left_eye_corners=config.left_eye_corners,
            right_eye_corners=config.right_eye_corners,
        )
        normalized = transform.apply(points)
        available = len(normalized)
        midline_indices = tuple(index for index in config.midline_indices if index < available)
        midline = fit_midline(normalized, midline_indices)
    except GeometryError as error:
        raise SymmetrizationError(str(error)) from error

    mirrored = reflect_across_midline(normalized, midline)
    symmetrized = normalized.copy()
    changed: list[int] = []

    usable = [
        (left, right) for left, right in config.all_pairs if left < available and right < available
    ]
    if usable:
        lefts = np.fromiter((left for left, _ in usable), dtype=np.int64, count=len(usable))
        rights = np.fromiter((right for _, right in usable), dtype=np.int64, count=len(usable))
        midpoints = (normalized[lefts] + mirrored[rights]) / 2.0
        symmetrized[lefts] = midpoints
        # 右側は構成上つねに左側の鏡像である。これにより結果が推定した直線に
        # 対して厳密に対称になる。
        symmetrized[rights] = reflect_across_midline(midpoints, midline)
        changed.extend(int(index) for index in lefts)
        changed.extend(int(index) for index in rights)

    if midline_indices:
        selected = np.asarray(midline_indices, dtype=np.int64)
        # `(p + mirror(p)) / 2` は、その直線への直交射影である。
        symmetrized[selected] = (normalized[selected] + mirrored[selected]) / 2.0
        changed.extend(int(index) for index in selected)

    changed.sort()
    offsets = symmetrized[changed] - normalized[changed] if changed else np.empty((0, 2))
    deviations = {
        index: float(value)
        for index, value in zip(changed, np.linalg.norm(offsets, axis=1), strict=True)
    }

    return SymmetrizationResult(
        measured=np.asarray(points, dtype=np.float64),
        symmetrized=transform.invert(symmetrized),
        symmetrized_indices=tuple(changed),
        deviations=deviations,
        midline=midline,
        interocular_px=transform.scale,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
