"""MediaPipe Image Segmenter adapter for background replacement (§14.3b).

The adapter owns the MediaPipe boundary only: it converts an RGB array into
a MediaPipe image, runs ``segment_for_video`` with a strictly increasing
timestamp, and returns a foreground confidence mask. It never composites a
background and never fetches a model. The mask is display-only input for
``domain.frame_adjustments.composite_background``; it never reaches the
landmark detector or the quality gate.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class SegmentationError(RuntimeError):
    """Recoverable failure at the segmentation model boundary."""


class SegmentationFrameOrderError(SegmentationError):
    """A frame arrived with a non-increasing timestamp (§14.4)."""


class _Detector(Protocol):
    def segment_for_video(self, image: Any, timestamp_ms: int) -> Any: ...

    def close(self) -> None: ...


DetectorFactory = Callable[[Path], Any]
ImageFactory = Callable[[NDArray[np.uint8]], Any]


def _create_detector(model_path: Path) -> Any:
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            ImageSegmenter,
            ImageSegmenterOptions,
            RunningMode,
        )
    except ImportError as error:
        raise SegmentationError("MediaPipe を読み込めません。") from error

    try:
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        return ImageSegmenter.create_from_options(options)
    except Exception as error:
        raise SegmentationError("セグメンテーションモデルを初期化できません。") from error


def _default_detector_factory(model_path: Path) -> Any:
    return _create_detector(model_path)


def _default_image_factory(rgb_image: NDArray[np.uint8]) -> Any:
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_mask(result: Any, width: int, height: int) -> NDArray[np.float32]:
    masks = list(getattr(result, "confidence_masks", []) or [])
    if len(masks) != 1:
        raise SegmentationError("セグメンテーション出力が不正です。")
    # `numpy_view()` is a live view into MediaPipe/TFLite-owned memory that the
    # delegate can reuse for a later call; copy immediately so this adapter
    # never hands the caller a buffer that can mutate out from under it.
    mask = np.array(masks[0].numpy_view(), dtype=np.float32, copy=True)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != (height, width) or not np.all(np.isfinite(mask)):
        raise SegmentationError("セグメンテーション出力の形状または値が不正です。")
    return mask


class MediaPipeSegmenter:
    """Local Image Segmenter adapter running in ``RunningMode.VIDEO``."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_version: str,
        detector_factory: DetectorFactory = _default_detector_factory,
        image_factory: ImageFactory = _default_image_factory,
    ) -> None:
        if not model_path.is_file():
            raise SegmentationError("セグメンテーションモデルが見つかりません。")
        self._model_version = model_version
        self._model_sha256 = _file_sha256(model_path)
        self._image_factory = image_factory
        self._detector: Any | None = detector_factory(model_path)
        self._last_timestamp_ms = -1
        self._warm_up()

    def _warm_up(self) -> None:
        """Discard one inference before any real frame (§21.1).

        The XNNPACK delegate's very first inference after construction was
        observed to intermittently return uninitialized (finite but
        nonsensical) values on this runtime, stabilizing from the second
        call onward. Absorbing that here means every call through
        ``segment`` is reliable, instead of leaking a one-in-a-few-calls
        garbage frame into the displayed composite.
        """
        detector = self._detector
        if detector is None:
            return
        try:
            dummy = self._image_factory(np.zeros((2, 2, 3), dtype=np.uint8))
            detector.segment_for_video(dummy, -1)
        except Exception as error:
            raise SegmentationError("セグメンテーションモデルを初期化できません。") from error

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def segment(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        timestamp_ms: int,
    ) -> NDArray[np.float32]:
        detector = self._detector
        if detector is None:
            raise SegmentationError("セグメンテーション推論は停止済みです。")
        if rgb_image.dtype != np.uint8 or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise SegmentationError("セグメンテーションには uint8 の RGB 画像が必要です。")
        if timestamp_ms <= self._last_timestamp_ms:
            raise SegmentationFrameOrderError("フレームの時刻が単調増加していません。")

        self._last_timestamp_ms = timestamp_ms
        try:
            result = detector.segment_for_video(self._image_factory(rgb_image), timestamp_ms)
        except Exception as error:
            raise SegmentationError("セグメンテーションを実行できませんでした。") from error

        height, width = rgb_image.shape[:2]
        return _to_mask(result, width, height)

    def close(self) -> None:
        detector, self._detector = self._detector, None
        if detector is not None:
            detector.close()
