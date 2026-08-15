from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.reference_stats import ReferenceDataset
from face_symmetry_lab.domain.warp_mesh import WarpMesh


class DemographicEstimator(Protocol):
    """ローカルの年齢・性別 runtime が実装する境界。"""

    def estimate(self, rgb_image: NDArray[np.uint8]) -> DemographicEstimate: ...


class DemographicSampler(Protocol):
    """クリティカルパス外で年齢・性別を標本化するための境界（§14.3）。

    リアルタイム pipeline は切り出しを提示し、sampler が終えたものを読むだけで
    あり、推論を待つことはない。``latest`` は推定と版トークンを対にして返す。
    このトークンは新しい結果が出たときだけ変わり、それが新規観測と再表示を
    区別する手がかりになる。
    """

    def is_due(self) -> bool: ...

    def submit(self, rgb_crop: NDArray[np.uint8]) -> object: ...

    def latest(self) -> tuple[DemographicEstimate, int] | None: ...

    def set_interval_ms(self, interval_ms: int) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class LandmarkProvider(Protocol):
    """映像上で動作するローカルの landmark runtime が実装する境界。"""

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame: ...

    def close(self) -> None: ...


class ImageLandmarkProvider(Protocol):
    """静止画上で動作するローカルの landmark runtime が実装する境界。"""

    def detect(self, rgb_image: NDArray[np.uint8], *, frame_id: int = 0) -> LandmarkFrame: ...

    def close(self) -> None: ...


class PiecewiseWarper(Protocol):
    """§8.6.4 の画素変形の境界。ローカルの画像ライブラリで実装する。

    どの点が動くか、どれが固定か、変位が安全かという幾何的な判断はすべて
    メッシュがすでに持っている。そのため実装は三角形分割と画素の転送だけを
    行えばよい。
    """

    def warp(self, rgb_image: NDArray[np.uint8], mesh: WarpMesh) -> NDArray[np.uint8]: ...


class SegmentationProvider(Protocol):
    """ローカルの人物・背景 segmentation runtime が実装する境界。"""

    def segment(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        timestamp_ms: int,
    ) -> NDArray[np.float32]: ...

    def close(self) -> None: ...


class ReferenceCellRepository(Protocol):
    """§9.5 の凍結参照アセット（`configs/reference/`）を読む境界（FR-037）。"""

    def load(self) -> ReferenceDataset: ...
