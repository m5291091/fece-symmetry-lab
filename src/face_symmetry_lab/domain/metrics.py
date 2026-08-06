"""版付きの指標設定（§8.6, §15.2）。

ランドマークの対応表は指標の版の一部である。ここでは実行時に何も導出しない。
値は凍結された設定 bundle に由来し、結果を常に再現できるようにするためである。

bundle は感度係数も部位の重みも持たない。§8.6 が `100 × exp(-k_p × d_p)` の
変換と `Σ(w_p × score_p)` の混合を距離そのものへ置き換えたため、bundle が固定
すべき自由パラメータは残っていない。これは、採点を自動調整できるものがもはや
存在しない理由でもある（FR-030）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

PART_NAMES: Final[tuple[str, ...]] = ("contour", "eyes", "nose", "mouth")

#: 直線の推定が意味を持つには、正中点が最低 3 点必要である（§8.4）。
MIN_MIDLINE_POINTS: Final[int] = 3


class MetricConfigError(ValueError):
    """指標設定が不正である。"""


@dataclass(frozen=True, slots=True)
class PartMap:
    """顔の部位 1 つ分の、左右ランドマークのペア。"""

    pairs: tuple[tuple[int, int], ...]
    min_pairs: int


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """完全に版付けされた、不変の指標 bundle 1 件。"""

    metric_version: str
    landmark_map_version: str
    landmark_count: int
    left_eye_corners: tuple[int, int]
    right_eye_corners: tuple[int, int]
    midline_indices: tuple[int, ...]
    parts: Mapping[str, PartMap]

    @property
    def all_pairs(self) -> tuple[tuple[int, int], ...]:
        """全部位のペアを 1 つにまとめた集合。§8.6 の総合距離に使う。

        `build_metric_config` は同じランドマークの二重使用を拒否するので、
        まとめても点が二重に数えられることはない。どの部位が定義したもので
        あっても、各対応は総合の折りたたみでちょうど 1 票を持つ。
        """
        return tuple(pair for name in PART_NAMES for pair in self.parts[name].pairs)

    @property
    def overall_min_pairs(self) -> int:
        """総合距離が成立するために §8.6 が必要とする、まとめたペアの本数。

        独立した数値ではなく、部位ごとの最小値の総和である。それらはすでに
        「この顔を測るのに十分な対応がある」という版付きの宣言であり、再利用
        することで bundle に未較正の閾値をもう 1 つ増やさずに済む。
        """
        return sum(self.parts[name].min_pairs for name in PART_NAMES)


def _validate_indices(indices: tuple[int, ...], landmark_count: int, label: str) -> None:
    for index in indices:
        if not 0 <= index < landmark_count:
            raise MetricConfigError(f"{label} の index が範囲外です: {index}")


def build_metric_config(
    *,
    metric_version: str,
    landmark_map_version: str,
    landmark_count: int,
    left_eye_corners: tuple[int, int],
    right_eye_corners: tuple[int, int],
    midline_indices: tuple[int, ...],
    parts: Mapping[str, PartMap],
) -> MetricConfig:
    """指標 bundle を検証する。不正な対応表が距離計算へ届いてはならない。"""
    if landmark_count <= 0:
        raise MetricConfigError("landmark_count は正の値である必要があります。")

    missing = [name for name in PART_NAMES if name not in parts]
    if missing:
        raise MetricConfigError(f"部位の定義が不足しています: {', '.join(missing)}")

    if len(midline_indices) < MIN_MIDLINE_POINTS:
        raise MetricConfigError(f"正中点は {MIN_MIDLINE_POINTS} 点以上必要です。")
    _validate_indices(midline_indices, landmark_count, "正中点")

    for label, corners in (("左眼", left_eye_corners), ("右眼", right_eye_corners)):
        if corners[0] == corners[1]:
            raise MetricConfigError(f"{label}の目頭・目尻は別の index である必要があります。")
        _validate_indices(corners, landmark_count, f"{label}")

    seen: set[int] = set()
    for name in PART_NAMES:
        part = parts[name]
        if part.min_pairs < 1:
            raise MetricConfigError(f"{name} の min_pairs は 1 以上である必要があります。")
        if part.min_pairs > len(part.pairs):
            raise MetricConfigError(f"{name} の min_pairs が定義済みペア数を超えています。")
        for left, right in part.pairs:
            if left == right:
                raise MetricConfigError(f"{name} に自己対応のペアがあります: {left}")
            _validate_indices((left, right), landmark_count, name)
            if left in seen or right in seen:
                raise MetricConfigError(f"{name} に重複した landmark があります: {left}, {right}")
            seen.add(left)
            seen.add(right)

    return MetricConfig(
        metric_version=metric_version,
        landmark_map_version=landmark_map_version,
        landmark_count=landmark_count,
        left_eye_corners=left_eye_corners,
        right_eye_corners=right_eye_corners,
        midline_indices=tuple(midline_indices),
        parts={name: parts[name] for name in PART_NAMES},
    )
