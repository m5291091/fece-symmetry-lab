from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.models import AgeEstimate


class AgeEstimator(Protocol):
    """Boundary implemented by local age-estimation runtimes."""

    def estimate(self, rgb_image: NDArray[np.uint8]) -> AgeEstimate: ...


class LandmarkProvider(Protocol):
    """Boundary implemented by local landmark runtimes running on video."""

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame: ...

    def close(self) -> None: ...


class ImageLandmarkProvider(Protocol):
    """Boundary implemented by local landmark runtimes running on still images."""

    def detect(self, rgb_image: NDArray[np.uint8], *, frame_id: int = 0) -> LandmarkFrame: ...

    def close(self) -> None: ...
