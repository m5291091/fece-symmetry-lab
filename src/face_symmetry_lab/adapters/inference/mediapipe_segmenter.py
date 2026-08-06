"""背景置換のための MediaPipe Image Segmenter adapter（§14.3b）。

この adapter が担うのは MediaPipe との境界だけである。RGB 配列を MediaPipe の
画像へ変換し、狭義単調増加する timestamp とともに ``segment_for_video`` を実行し、
前景の確信度 mask を返す。背景の合成は行わず、モデルを取得することもない。
mask は ``domain.frame_adjustments.composite_background`` への表示専用の入力で
あり、ランドマーク検出器にも品質ゲートにも届かない。
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.runtime_errors import init_failure_detail


class SegmentationError(RuntimeError):
    """segmentation モデル境界で発生した、回復可能な失敗。"""


class SegmentationFrameOrderError(SegmentationError):
    """timestamp が増加していないフレームが届いた（§14.4）。"""


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
        raise SegmentationError(
            f"セグメンテーションモデルを初期化できません。{init_failure_detail(error)}"
        ) from error


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
    # `numpy_view()` は MediaPipe/TFLite が所有するメモリへの生きたビューで
    # あり、delegate が後続の呼び出しで再利用しうる。直ちに複製することで、
    # 呼び出し側の背後で書き換わるバッファを渡さないようにする。
    mask = np.array(masks[0].numpy_view(), dtype=np.float32, copy=True)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != (height, width) or not np.all(np.isfinite(mask)):
        raise SegmentationError("セグメンテーション出力の形状または値が不正です。")
    return mask


class MediaPipeSegmenter:
    """``RunningMode.VIDEO`` で動作するローカル Image Segmenter adapter。"""

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
        """実フレームの前に推論を 1 回捨てる（§21.1）。

        この runtime では、構築直後の XNNPACK delegate の最初の推論が、
        初期化されていない（有限だが無意味な）値を断続的に返すことが観測された。
        2 回目以降は安定する。ここで吸収しておくことで ``segment`` 経由の呼び
        出しがすべて信頼できるものになり、数回に 1 回のごみフレームが表示用の
        合成へ漏れ出すのを防げる。
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
