"""Still-image landmark use case.

Stateless by definition (§8.7): no EMA, no history, no timestamp. Multi-face
and no-face inputs return an un-annotated preview so the UI can never present
one face's landmarks as "the" result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.ports import ImageLandmarkProvider
from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.overlay import (
    DEFAULT_POINT_COLOR,
    DEFAULT_POINT_RADIUS,
    draw_landmark_points,
)


@dataclass(frozen=True, slots=True)
class StillImageLandmarks:
    """One still-image detection and the preview to render for it."""

    frame: LandmarkFrame
    preview: NDArray[np.uint8]

    @property
    def landmark_count(self) -> int:
        return len(self.frame.points)


def detect_still_image_landmarks(
    provider: ImageLandmarkProvider,
    rgb_image: NDArray[np.uint8],
    *,
    color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
    radius: int = DEFAULT_POINT_RADIUS,
) -> StillImageLandmarks:
    """Detect landmarks in one image and return an annotated copy."""
    frame = provider.detect(rgb_image)
    preview = draw_landmark_points(rgb_image, frame, color=color, radius=radius)
    return StillImageLandmarks(frame=frame, preview=preview)
