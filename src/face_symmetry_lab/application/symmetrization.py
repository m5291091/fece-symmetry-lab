"""One-shot symmetrization use case (§14.3g, FR-034).

Stateless by definition, like `application.still_image`: no EMA, no history,
no timestamp, no recorder. One call answers one question about one frame, so
this never participates in the realtime cadence and cannot perturb a running
session's smoothing or comparison segment.

It is deliberately not wired into `RealtimeLandmarkPipeline._analyse`: the
overlay is a full-resolution repaint that has no place in an 8 Hz loop, and a
still result is what the feature is actually for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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


@dataclass(frozen=True, slots=True)
class SymmetrizationView:
    """One symmetrization and the image to render for it."""

    result: SymmetrizationResult
    overlay: NDArray[np.uint8]

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
    measured_color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
    symmetrized_color: tuple[int, int, int] = DEFAULT_SYMMETRIZED_COLOR,
    deviation_color: tuple[int, int, int] = DEFAULT_DEVIATION_COLOR,
    radius: int = DEFAULT_POINT_RADIUS,
) -> SymmetrizationView:
    """Symmetrize one frame's landmarks and paint the deviation overlay.

    Raises `SymmetrizationError` rather than returning a partial view when the
    frame cannot carry the construction at all — a multi-face or size-less
    frame has no single midline to fold about, and drawing one face's fold on
    an image holding two would attribute it to the wrong person (FR-008).
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
    return SymmetrizationView(result=result, overlay=overlay)
