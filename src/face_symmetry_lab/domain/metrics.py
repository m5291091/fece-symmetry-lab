"""Versioned metric configuration (§8.6, §15.2).

The landmark correspondence map is part of the metric version. Nothing here is
derived at runtime: the values come from a frozen config bundle so a result can
always be reproduced.

The bundle carries no sensitivity coefficient and no part weight. §8.6 replaced
the `100 × exp(-k_p × d_p)` transform and the `Σ(w_p × score_p)` blend with the
raw distances themselves, so there is no free parameter left for a bundle to
pin — which is also why nothing may auto-tune scoring any more (FR-030).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

PART_NAMES: Final[tuple[str, ...]] = ("contour", "eyes", "nose", "mouth")

#: A line fit needs at least three midline points to be meaningful (§8.4).
MIN_MIDLINE_POINTS: Final[int] = 3


class MetricConfigError(ValueError):
    """Invalid metric configuration."""


@dataclass(frozen=True, slots=True)
class PartMap:
    """Left/right landmark pairs for one facial part."""

    pairs: tuple[tuple[int, int], ...]
    min_pairs: int


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """One immutable, fully versioned metric bundle."""

    metric_version: str
    landmark_map_version: str
    landmark_count: int
    left_eye_corners: tuple[int, int]
    right_eye_corners: tuple[int, int]
    midline_indices: tuple[int, ...]
    parts: Mapping[str, PartMap]

    @property
    def all_pairs(self) -> tuple[tuple[int, int], ...]:
        """Every part's pairs as one pooled set, for §8.6's overall distance.

        `build_metric_config` rejects a landmark used twice, so pooling can
        never double-count a point: each correspondence carries exactly one
        vote in the overall fold, whichever part defined it.
        """
        return tuple(pair for name in PART_NAMES for pair in self.parts[name].pairs)

    @property
    def overall_min_pairs(self) -> int:
        """How many pooled pairs §8.6 needs before an overall distance exists.

        The sum of the per-part minimums rather than a separate number: those
        are already the versioned statement of "enough correspondences to
        measure this face", and reusing them keeps the bundle free of one more
        uncalibrated threshold.
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
    """Validate a metric bundle. An invalid map must never reach a distance."""
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
