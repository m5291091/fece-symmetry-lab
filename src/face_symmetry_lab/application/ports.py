from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.models import AgeEstimate


class AgeEstimator(Protocol):
    """Boundary implemented by local age-estimation runtimes."""

    def estimate(self, rgb_image: NDArray[np.uint8]) -> AgeEstimate: ...
