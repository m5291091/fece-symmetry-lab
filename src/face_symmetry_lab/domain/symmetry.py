"""Symmetry scoring (§8.5, §8.6).

    d_p     = sqrt(mean(||left_i - mirror(right_i)||^2))
    score_p = clamp(100 * exp(-k_p * d_p), 0, 100)
    overall = sum(w_p * score_p)

A part with too few usable pairs is missing, never zero, and a missing part
makes the overall score missing too (§8.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.geometry import (
    GeometryError,
    fit_midline,
    normalize_landmarks,
    reflect_across_midline,
)
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig
from face_symmetry_lab.domain.models import SymmetryScore


class SymmetryError(ValueError):
    """The input cannot produce a symmetry score at all."""


@dataclass(frozen=True, slots=True)
class SymmetryResult:
    """Per-part scores with their raw normalized distances."""

    scores: dict[str, float | None]
    distances: dict[str, float | None]
    overall: float | None
    metric_version: str
    landmark_map_version: str

    def part(self, name: str) -> float | None:
        return self.scores.get(name)

    def distance(self, name: str) -> float | None:
        return self.distances.get(name)

    @property
    def is_complete(self) -> bool:
        return self.overall is not None and all(
            self.scores.get(name) is not None for name in PART_NAMES
        )

    def to_score(self) -> SymmetryScore | None:
        """Convert to the shared DTO, or None when any part is missing."""
        if not self.is_complete:
            return None
        return SymmetryScore(
            contour=float(self.scores["contour"]),  # type: ignore[arg-type]
            eyes=float(self.scores["eyes"]),  # type: ignore[arg-type]
            nose=float(self.scores["nose"]),  # type: ignore[arg-type]
            mouth=float(self.scores["mouth"]),  # type: ignore[arg-type]
            overall=float(self.overall),  # type: ignore[arg-type]
        )


def score_from_distance(distance: float, sensitivity: float) -> float:
    """Map a normalized RMS distance to 0-100. Purely a readability transform."""
    if not isfinite(distance) or distance < 0:
        raise SymmetryError("距離が有限の非負値ではありません。")
    return float(min(100.0, max(0.0, 100.0 * exp(-sensitivity * distance))))


def _part_distance(
    normalized: NDArray[np.float64],
    mirrored: NDArray[np.float64],
    pairs: tuple[tuple[int, int], ...],
    available: int,
) -> tuple[float, int]:
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
    """Score the left/right difference of one normalized face."""
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

    scores: dict[str, float | None] = {}
    distances: dict[str, float | None] = {}
    for name in PART_NAMES:
        part = config.parts[name]
        distance, used = _part_distance(normalized, mirrored, part.pairs, available)
        if used < part.min_pairs:
            scores[name] = None
            distances[name] = None
            continue
        distances[name] = distance
        scores[name] = score_from_distance(distance, config.sensitivity[name])

    overall: float | None = None
    if all(scores[name] is not None for name in PART_NAMES):
        total = sum(config.weights[name] * float(scores[name] or 0.0) for name in PART_NAMES)
        overall = float(min(100.0, max(0.0, total)))

    return SymmetryResult(
        scores=scores,
        distances=distances,
        overall=overall,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
