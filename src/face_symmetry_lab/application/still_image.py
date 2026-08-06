"""静止画のランドマーク use case。

定義上まったく状態を持たない（§8.7）。EMA も履歴も timestamp も無い。複数顔や
顔なしの入力に対しては注記の無いプレビューを返す。UI が 1 人分のランドマークを
「その」結果として提示できないようにするためである。
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
    """静止画 1 枚の検出結果と、それを表示するためのプレビュー。"""

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
    """画像 1 枚のランドマークを検出し、注記を描き込んだ複製を返す。"""
    frame = provider.detect(rgb_image)
    preview = draw_landmark_points(rgb_image, frame, color=color, radius=radius)
    if show_midline:
        preview = draw_face_midline(
            preview,
            frame,
            midline_indices=midline_indices,
        )
    return StillImageLandmarks(frame=frame, preview=preview)


#: 静止画では安定性の連続数を作れないので、有効なフレーム 1 枚で十分である。
STILL_IMAGE_THRESHOLDS = QualityThresholds(valid_frames_to_ok=1)


@dataclass(frozen=True, slots=True)
class StillImageAnalysis:
    """アップロードされた写真 1 枚の、ランドマークと左右差の結果。"""

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
    """写真 1 枚を検出・描画・採点する。状態を持たず、平滑化は適用しない。"""
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
