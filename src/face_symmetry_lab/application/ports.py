from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.models import DemographicEstimate


class DemographicEstimator(Protocol):
    """Boundary implemented by local age/gender runtimes."""

    def estimate(self, rgb_image: NDArray[np.uint8]) -> DemographicEstimate: ...


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


class SegmentationProvider(Protocol):
    """Boundary implemented by local person/background segmentation runtimes."""

    def segment(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        timestamp_ms: int,
    ) -> NDArray[np.float32]: ...

    def close(self) -> None: ...
