"""Symmetrized landmark positions and their deviations (§8.6.2, FR-034).

Folding a face across its midline gives, for every versioned correspondence
`(l, r)`, one position both sides would occupy if the two halves matched:

    mid_l = (left_l + mirror(right_r)) / 2
    mid_r = mirror(mid_l)

A midline point is the same formula applied to itself, `(p + mirror(p)) / 2`,
which is exactly its orthogonal projection onto the fitted line — so one rule
covers both cases. Landmarks with no defined correspondence (the majority of
the 478) are left at their measured position rather than guessed at.

This introduces no new metric. The distance from a measured point to its
symmetrized position is exactly half `||left_i - mirror(right_i)||` from §8.5,
so this module is a way of *showing* the existing measurement, not a second
one. Nothing here ranks, scores, or corrects a face: "symmetrized" names a
geometric construction, never a target or an improvement (§5.3, §10.7).
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
    """The input cannot be symmetrized at all."""


@dataclass(frozen=True, slots=True)
class SymmetrizationResult:
    """Measured and symmetrized positions for one face, in source coordinates."""

    #: The input points, unchanged, so a caller draws both from one result.
    measured: NDArray[np.float64]
    #: Same shape as `measured`. Indices outside `symmetrized_indices` are
    #: carried over untouched, never interpolated.
    symmetrized: NDArray[np.float64]
    #: Which indices actually moved: the usable pairs plus the midline points.
    symmetrized_indices: tuple[int, ...]
    #: index -> distance between measured and symmetrized, in units of the
    #: interocular distance (§8.3), so it is comparable with §8.5's distances.
    deviations: dict[int, float]
    #: The fitted midline, in the normalized frame it was fitted in.
    midline: Midline
    metric_version: str
    landmark_map_version: str

    @property
    def mean_deviation(self) -> float | None:
        """Mean normalized deviation, or None when nothing was symmetrized."""
        if not self.deviations:
            return None
        return float(np.mean(list(self.deviations.values())))

    @property
    def max_deviation(self) -> float | None:
        """Largest normalized deviation, or None when nothing was symmetrized."""
        if not self.deviations:
            return None
        return float(max(self.deviations.values()))


def symmetrize_landmarks(
    points: NDArray[np.float64],
    config: MetricConfig,
) -> SymmetrizationResult:
    """Build the symmetrized counterpart of one face's landmarks."""
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
        # The right side is the left side's mirror by construction, which is
        # what makes the result exactly symmetric about the fitted line.
        symmetrized[rights] = reflect_across_midline(midpoints, midline)
        changed.extend(int(index) for index in lefts)
        changed.extend(int(index) for index in rights)

    if midline_indices:
        selected = np.asarray(midline_indices, dtype=np.int64)
        # `(p + mirror(p)) / 2` is the orthogonal projection onto the line.
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
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
