"""Symmetry distances (§8.5, §8.6).

    d_p   = sqrt(mean_{i in pairs(p)}(||left_i - mirror(right_i)||^2))
    d_all = sqrt(mean_{i in every part's pairs pooled}(same))

Both are normalized RMS distances in units of the interocular distance, which
§8.3 scales to 1. There is no `100 × exp(-k_p × d_p)` readability transform and
no `Σ(w_p × score_p)` blend any more: the sensitivity coefficients were never
frozen against a fixed evaluation set and the part weights had no empirical
basis, so both arbitrary parameters were removed rather than left to shape a
displayed number.

`d_all` is not composed from `d_p`. It folds one pooled set of correspondences
once, so every point carries exactly one vote and no part-importance judgement
enters the total. The per-part values are a parallel, purely diagnostic output
("where is the deviation concentrated?") and never feed the overall one.

A part with too few usable pairs is missing, never zero. A missing part does
*not* make the overall value missing — the two are independent outputs, and the
pooled fold stays well defined on whatever pairs remain (§8.5, §8.6).
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
    """The input cannot produce a symmetry distance at all."""


@dataclass(frozen=True, slots=True)
class SymmetryResult:
    """Per-part normalized RMS distances plus the independent pooled total."""

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
        """Convert to the shared DTO, or None when any value is missing."""
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
    """RMS distance over the usable pairs, and how many there were."""
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
    """Measure the left/right difference of one normalized face."""
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

    # §8.6: folded once over the pooled pairs, not assembled from the four
    # values above, so this stays defined even when a part has gone missing.
    pooled, pooled_used = _pair_distance(normalized, mirrored, config.all_pairs, available)
    overall = pooled if pooled_used >= config.overall_min_pairs else None

    return SymmetryResult(
        distances=distances,
        overall=overall,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
