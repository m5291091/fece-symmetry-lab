"""Symmetry distances (§8.5, §8.6).

    d_p   = sqrt(mean_{i in pairs(p)}(||left_i - mirror(right_i)||^2))
    d_all = sqrt(mean_{i in every part's pairs pooled}(same))

いずれも両眼間隔を単位とする正規化 RMS 距離であり、§8.3 がその間隔を 1 にする。
`100 × exp(-k_p × d_p)` という可読化変換も、`Σ(w_p × score_p)` という混合も
もう存在しない。感度係数は固定評価セットに対して凍結されたことが一度も無く、
部位の重みには経験的な根拠が無かった。表示される数値を形作らせたままにせず、
どちらの恣意的なパラメータも取り除いた。

`d_all` は `d_p` から合成しない。まとめた 1 つの対応集合を 1 回だけ折りたたむ
ため、どの点もちょうど 1 票ずつ持ち、部位の重要度という判断が総合値へ入らない。
部位別の値は並列に得られる純粋な診断出力（「ズレはどこに集中しているか」）で
あり、総合値へは決して寄与しない。

使用できるペアが足りない部位は欠測であって 0 ではない。ある部位が欠測でも総合値
は欠測に *ならない*。両者は独立した出力であり、まとめた折りたたみは残っている
ペアの上で定義され続ける（§8.5, §8.6）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.geometry import (
    GeometryError,
    fit_midline,
    normalize_landmarks,
    reflect_across_midline,
)
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig
from face_symmetry_lab.domain.models import SymmetryDistance


class SymmetryError(ValueError):
    """入力からそもそも左右差距離を求められない。"""


@dataclass(frozen=True, slots=True)
class SymmetryResult:
    """部位別の正規化 RMS 距離と、それとは独立にまとめて求めた総合値。"""

    distances: dict[str, float | None]
    overall: float | None
    metric_version: str
    landmark_map_version: str

    def distance(self, name: str) -> float | None:
        return self.distances.get(name)

    @property
    def is_complete(self) -> bool:
        return self.overall is not None and all(
            self.distances.get(name) is not None for name in PART_NAMES
        )

    def to_distance(self) -> SymmetryDistance | None:
        """共有 DTO へ変換する。欠測値が 1 つでもあれば None を返す。"""
        if not self.is_complete:
            return None
        return SymmetryDistance(
            contour=float(self.distances["contour"]),  # type: ignore[arg-type]
            eyes=float(self.distances["eyes"]),  # type: ignore[arg-type]
            nose=float(self.distances["nose"]),  # type: ignore[arg-type]
            mouth=float(self.distances["mouth"]),  # type: ignore[arg-type]
            overall=float(self.overall),  # type: ignore[arg-type]
        )


def _pair_distance(
    normalized: NDArray[np.float64],
    mirrored: NDArray[np.float64],
    pairs: tuple[tuple[int, int], ...],
    available: int,
) -> tuple[float, int]:
    """使用できたペアに対する RMS 距離と、その本数を返す。"""
    usable = [(left, right) for left, right in pairs if left < available and right < available]
    if not usable:
        return 0.0, 0
    left_points = normalized[[left for left, _ in usable]]
    mirrored_right = mirrored[[right for _, right in usable]]
    squared = np.sum((left_points - mirrored_right) ** 2, axis=1)
    return float(np.sqrt(float(np.mean(squared)))), len(usable)


def compute_symmetry(
    points: NDArray[np.float64],
    config: MetricConfig,
) -> SymmetryResult:
    """正規化した顔 1 つ分の左右差を測る。"""
    try:
        normalized = normalize_landmarks(
            points,
            left_eye_corners=config.left_eye_corners,
            right_eye_corners=config.right_eye_corners,
        )
        available = len(normalized)
        midline_indices = tuple(index for index in config.midline_indices if index < available)
        midline = fit_midline(normalized, midline_indices)
    except GeometryError as error:
        raise SymmetryError(str(error)) from error

    mirrored = reflect_across_midline(normalized, midline)

    distances: dict[str, float | None] = {}
    for name in PART_NAMES:
        part = config.parts[name]
        distance, used = _pair_distance(normalized, mirrored, part.pairs, available)
        distances[name] = distance if used >= part.min_pairs else None

    # §8.6: まとめたペアの上で 1 回だけ折りたたむ。上の 4 つの値から組み立て
    # ないので、ある部位が欠測になっても総合値は定義され続ける。
    pooled, pooled_used = _pair_distance(normalized, mirrored, config.all_pairs, available)
    overall = pooled if pooled_used >= config.overall_min_pairs else None

    return SymmetryResult(
        distances=distances,
        overall=overall,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
