from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.warp_mesh import WarpMesh


class DemographicEstimator(Protocol):
    """Boundary implemented by local age/gender runtimes."""

    def estimate(self, rgb_image: NDArray[np.uint8]) -> DemographicEstimate: ...


class DemographicSampler(Protocol):
    """Boundary for off-critical-path age/gender sampling (§14.3).

    The realtime pipeline only ever offers a crop and reads whatever the
    sampler has finished; it never waits for an inference. ``latest`` pairs
    the estimate with a version token that changes only on a fresh result,
    which is what distinguishes a new observation from a redisplay.
    """

    def is_due(self) -> bool: ...

    def submit(self, rgb_crop: NDArray[np.uint8]) -> object: ...

    def latest(self) -> tuple[DemographicEstimate, int] | None: ...

    def set_interval_ms(self, interval_ms: int) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


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


class PiecewiseWarper(Protocol):
    """Boundary for §8.6.4's pixel warp, implemented with a local imaging library.

    The mesh already carries every geometric decision (which points move,
    which are pinned, whether the displacement is safe), so an implementation
    only has to triangulate and transfer pixels.
    """

    def warp(self, rgb_image: NDArray[np.uint8], mesh: WarpMesh) -> NDArray[np.uint8]: ...


class SegmentationProvider(Protocol):
    """Boundary implemented by local person/background segmentation runtimes."""

    def segment(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        timestamp_ms: int,
    ) -> NDArray[np.float32]: ...

    def close(self) -> None: ...
