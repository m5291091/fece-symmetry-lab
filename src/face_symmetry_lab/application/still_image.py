"""Still-image landmark use case.

Stateless by definition (§8.7): no EMA, no history, no timestamp. Multi-face
and no-face inputs return an un-annotated preview so the UI can never present
one face's landmarks as "the" result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.analyzer import AnalysisOutcome, FrameAnalyzer
from face_symmetry_lab.application.ports import ImageLandmarkProvider
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.overlay import (
    DEFAULT_POINT_COLOR,
    DEFAULT_POINT_RADIUS,
    draw_face_midline,
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
    show_midline: bool = False,
    midline_indices: tuple[int, ...] = (),
) -> StillImageLandmarks:
    """Detect landmarks in one image and return an annotated copy."""
    frame = provider.detect(rgb_image)
    preview = draw_landmark_points(rgb_image, frame, color=color, radius=radius)
    if show_midline:
        preview = draw_face_midline(
            preview,
            frame,
            midline_indices=midline_indices,
        )
    return StillImageLandmarks(frame=frame, preview=preview)


#: A still image cannot build a stability streak, so one valid frame is enough.
STILL_IMAGE_THRESHOLDS = QualityThresholds(valid_frames_to_ok=1)


@dataclass(frozen=True, slots=True)
class StillImageAnalysis:
    """Landmarks plus the symmetry outcome for one uploaded photo."""

    landmarks: StillImageLandmarks
    outcome: AnalysisOutcome

    @property
    def preview(self) -> NDArray[np.uint8]:
        return self.landmarks.preview


def analyze_still_image(
    provider: ImageLandmarkProvider,
    rgb_image: NDArray[np.uint8],
    *,
    config: MetricConfig,
    thresholds: QualityThresholds | None = None,
    color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
    radius: int = DEFAULT_POINT_RADIUS,
    show_midline: bool = True,
) -> StillImageAnalysis:
    """Detect, draw, and score one photo. Stateless: no smoothing is applied."""
    landmarks = detect_still_image_landmarks(
        provider,
        rgb_image,
        color=color,
        radius=radius,
        show_midline=show_midline,
        midline_indices=config.midline_indices,
    )
    analyzer = FrameAnalyzer(
        config=config,
        thresholds=thresholds or STILL_IMAGE_THRESHOLDS,
    )
    outcome = analyzer.analyze(landmarks.frame, at=0.0, rgb_image=rgb_image)
    return StillImageAnalysis(landmarks=landmarks, outcome=outcome)
