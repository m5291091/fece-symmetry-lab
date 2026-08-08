#!/usr/bin/env python3
"""実写の連続フレームでノイズ床と左右バイアスを測る（docs のテストA・テストB）。

`symmetry_metric_probe.py` と `symmetry_level3_probe.py` は、合成顔と文献値で
指標の感度・分解能を推定した。本スクリプトはその推定を実写で確かめる。実測が
無いと以降の対策が効いたかどうかを測れないので、docs はこの 2 つのテストを
対策より先に置いている。

    uv run python scripts/measure_metric_noise.py --video take01.mp4
    uv run python scripts/measure_metric_noise.py --frames frames/ --fps 5
    uv run python scripts/measure_metric_noise.py --self-test     # 実写不要の健全性確認
    uv run python scripts/measure_metric_noise.py --video a.mp4 --section B --json a.json

    A: 鏡像テスト   d(I) と d(mirror(I)) の差 = 検出器の左右バイアス
    B: 再現性テスト d の分布と、§8.3 正規化座標の landmark ごとのばらつき
    C: 予測         座標を EMA したとき d がどこまで下がるか（対策10 の事前見積り）
    D: 3D 化        3D object symmetry が姿勢依存を消せるか（対策11 の実写判定）
    E: 既知量注入   実写へ既知の非対称を入れ、パイプラインのゲインを測る（docs テストC）
    F: 3D 鏡像      テストA を d_3d に対して行い、3D で左右バイアスがどれだけ残るかを測る
    G: 残差の分解   Procrustes 残差を xy 成分と z 成分へ分け、z が実質 2.5D かを見る
    H: 3D 姿勢層別  yaw を固定した部分集合で d_3d のノイズ床と z の再現性を測り直す
    I: 3 方式比較   現行2D / 2D Procrustes / 3D Procrustes を同じ材料で並べる
    J: 部位別ゲイン テストE の領域変位場を口・目・輪郭・鼻へ当て、応答の部位依存を見る
    K: 3D 座標平均  テストC の座標 EMA を 3D 正規化座標へ広げ、d_3d の床を測り直す

A〜D は前回の測定と比較するため、出力形式も数値の定義も変えていない。E 以降は
「d_3d は安定だが感度もあるのか」を確かめるために後から足した診断である。I〜K は
「3D 化が要るのか、Procrustes だけで足りるのか」を分離するために足した。A〜H の
数値と出力形式は I〜K を足しても変えていない。

`src/` は読むだけで一切変更しない。画像と landmark 座標は JSON へ書き出さず、
統計量だけを残す（AGENTS.md 不変条件 1）。変形画像はメモリ上だけで扱い、
ディスクへは書かない。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from math import ceil, exp, isfinite, log, sqrt
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.adapters.imaging.piecewise_warp import (
    OpenCvPiecewiseWarper,
    PiecewiseWarpError,
)
from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    DetectorFactory,
    MediaPipeFaceLandmarker,
    MediaPipeImageLandmarker,
    _create_detector,
)
from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.adapters.io.model_store import (
    DEFAULT_MODELS_DIR,
    InstallStatus,
    ModelStore,
    find_asset,
    load_manifest,
)
from face_symmetry_lab.domain.geometry import (
    GeometryError,
    normalization_transform,
    normalize_landmarks,
)
from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    to_pixel_array,
)
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.smoothing import TimeBasedEma
from face_symmetry_lab.domain.symmetry import SymmetryError, compute_symmetry
from face_symmetry_lab.domain.warp_mesh import (
    MAX_WARP_DISPLACEMENT_RATIO,
    WarpMeshError,
    build_warp_mesh,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.symmetry_level3_probe import FACE3D, distance_3d, yaw_matrix  # noqa: E402

CONFIG: MetricConfig = load_metric_config()

#: 成人の両眼間距離のおよその値。d は「両眼間距離 = 1」で正規化されているので、
#: これを掛けると mm 換算になる。実測 IPD があればそちらを使うこと。
IPD_MM: float = 63.0

#: Prog Orthod 2015 の顔全体 RMS 偏差。この 2 つの幅が「測りたい信号」。
NORMAL_D: float = 1.52 / IPD_MM
CLINICAL_D: float = 2.85 / IPD_MM
SIGNAL_SPAN: float = CLINICAL_D - NORMAL_D

#: MediaPipe Face Mesh 論文 (arXiv:1907.06724) の IOD 正規化 MAD。§8.3 の
#: 正規化スケールと論文の IOD の定義が同じなので、そのまま比較できる。
MEDIAPIPE_FULL_MAD: float = 0.0396
HUMAN_ANNOTATOR_MAD: float = 0.0256

DEFAULT_TAUS: tuple[float, ...] = (0.4, 1.0, 2.0, 4.0, 8.0)
DEFAULT_MAX_FRAMES: int = 200
DEFAULT_FPS: float = 5.0
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})

#: 統計の見積りが不安定になる下限。下回ったら結論を書かずに注意を出す。
MIN_USEFUL_FRAMES: int = 30

#: アプリの解析間隔（`application/analysis_cadence.py` の 5Hz）。撮影 fps に
#: 依存する tau を、アプリで設定すべき値へ読み替えるために使う。
APP_DELTA_SECONDS: float = 0.2

#: 姿勢で層別するときの、yaw 中央値からの許容幅。狭いほうを先に試す。
YAW_BAND_DEGREES: tuple[float, ...] = (0.5, 1.0)

#: 3D 化の判定に使う、docs level3 §4 の z 精度の閾値（両眼間距離比）。
Z_ERROR_BUDGET: float = 0.05

#: テストE で注入する既知の非対称（mm）。0.0 は恒等変形の対照であり、
#: 「非対称を入れずに変形と再標本化だけを通したとき d がどれだけ動くか」を測る。
#: この対照より小さい変化しか出ない δ は、ゲインを読める水準に達していない。
INJECTION_DELTAS_MM: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)

#: ゲインがこれを超えたら想定外（何かが二重に効いている）として印を付ける。
GAIN_UPPER_ALARM: float = 1.20

#: セクション I が比べる 3 方式。key は JSON、値は表の見出し。
METHOD_LABELS: dict[str, str] = {
    "current_2d": "① 現行 2D",
    "procrustes_2d": "② 2D Procrustes",
    "procrustes_3d": "③ 3D Procrustes",
}
METHOD_KEYS: tuple[str, ...] = tuple(METHOD_LABELS)

#: セクション J が注入する部位。key は `MetricConfig.parts` の名前。
PART_LABELS: dict[str, str] = {
    "mouth": "口",
    "eyes": "目",
    "contour": "輪郭",
    "nose": "鼻",
}

#: 「② は ③ に近い」と読むための、① と ③ の間での相対位置の下限。1.0 が ③、
#: 0.0 が ① である。較正値ではなく、表を読むための目盛りにすぎない。
NEAR_3D_POSITION: float = 0.8
#: 同じ目盛りで「② は ① 寄り」と読む上限。
NEAR_2D_POSITION: float = 0.2

JSON_SCHEMA: str = "measure-metric-noise/1.3.0"


# --------------------------------------------------------------------------
# 鏡像で index の意味がどう入れ替わるか（テストA の前提）
# --------------------------------------------------------------------------
def mirror_permutation(config: MetricConfig) -> dict[int, int]:
    """鏡像で landmark index がどう入れ替わるかの写像 sigma。

    凍結 bundle のペアは「鏡映後の相互最近傍が、4 通りの描画で一致したもの」
    として導出されている（`scripts/derive_landmark_map.py`）。つまりペア表その
    ものが sigma である。正中点は自分自身へ写る。ここで新しく推測はしない。
    """
    sigma: dict[int, int] = {}
    for left, right in config.all_pairs:
        sigma[left] = right
        sigma[right] = left
    for index in config.midline_indices:
        sigma[index] = index
    return sigma


def validate_permutation(config: MetricConfig, sigma: dict[int, int]) -> None:
    """sigma が鏡像写像として成立していることを、使う前に確かめる。

    ここが崩れている bundle で鏡像テストを回すと、左右バイアスではなく対応表の
    誤りを測ってしまう。黙って進めずに落とす。
    """
    if any(sigma[sigma[index]] != index for index in sigma):
        raise SystemExit("bundle のペア表が対合になっていません。鏡像テストを実行できません。")
    if {sigma[index] for index in config.left_eye_corners} != set(config.right_eye_corners):
        raise SystemExit("眼アンカーが左右で対応していません。鏡像テストを実行できません。")
    paired = {index for pair in config.all_pairs for index in pair}
    if set(config.midline_indices) & paired:
        raise SystemExit("正中点がペアにも現れています。鏡像テストを実行できません。")


MIRROR_PERMUTATION: dict[int, int] = mirror_permutation(CONFIG)
MEASURED_INDICES: tuple[int, ...] = tuple(
    sorted({index for pair in CONFIG.all_pairs for index in pair})
)
SIGMA_INDICES: tuple[int, ...] = tuple(sorted(MIRROR_PERMUTATION))

MIRROR_NOTE = """
  鏡像でも d の index の並びは変えない。凍結 bundle のペア表は「鏡映後の相互
  最近傍」で導出されており、鏡像で index の意味が入れ替わる写像 sigma そのもの
  だからである。sigma は次を満たす。

      sigma(左眼アンカー) = 右眼アンカー、sigma(正中点) = 正中点、
      sigma(ペアの片側)   = ペアの相手

  このとき正規化・正中線あてはめ・鏡映のすべてが x 反転付きで対応し、鏡映が
  対合であることから、幾何学的に正しい検出器なら d(I) と d(mirror(I)) は
  厳密に一致する。したがって d の比較で index を入れ替えては *ならない*。
  入れ替えると別の量を測ることになる。

  一方 landmark ごとの比較には入れ替えが要る。§8.3 の正規化座標で

      remap(i) = ( -N_mirror[sigma(i)].x , N_mirror[sigma(i)].y )

  としたものが N_original[i] と一致するはずで、その差が左右バイアスである。
  d は「片側だけを一律にズラす」種類のバイアスには構造的に反応しない（左右を
  入れ替えても RMS が変わらない）ので、landmark 側の数字のほうが感度が高い。
"""


# --------------------------------------------------------------------------
# 検出結果の器
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Detection:
    """フレーム 1 枚の検出結果。座標は §8.3 の正規化前である。

    `points3d` は MediaPipe が返す z を含む (N, 3) 画素相当座標である。z は
    adapter の DTO に無いので（`domain/landmarks.py` の LandmarkFrame は xy
    だけ）、テストD 以降だけがこの経路を使う。取れなければ None。
    `mirrored3d` は鏡像側の同じもので、テストF が使う。
    """

    index: int
    points: NDArray[np.float64]
    mirrored: NDArray[np.float64] | None
    pose: FacePose | None
    points3d: NDArray[np.float64] | None = None
    mirrored3d: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class Capture:
    """1 回の測定で集めた検出結果と、その撮影条件。"""

    detections: tuple[Detection, ...]
    frames_seen: int
    delta_seconds: float
    source_kind: str
    mode: str
    model_version: str
    model_sha256: str
    #: 実際に構築した MediaPipe の RunningMode 名と adapter クラス名。--mode が
    #: 検出器へ届いたことを、出力を見るだけで確かめられるようにしてある。
    running_mode: str = "—"
    detector_class: str = "—"

    @property
    def fps(self) -> float:
        return 1.0 / self.delta_seconds if self.delta_seconds > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Series:
    """§8.3 で正規化した座標と d の時系列。テストB と予測が共有する。

    `normalized3d` と `distances3d` はテストD 用で、z を取れなかったフレームは
    NaN で埋めてある。こうしておくと `poses` などとの添字対応が崩れない。

    `distances2dp` はセクション I の方式② で、`normalized` から計算する。z を
    使わないので、2D 側が取れたフレームでは必ず値を持つ。
    """

    times: NDArray[np.float64]
    distances: NDArray[np.float64]
    normalized: NDArray[np.float64]
    poses: tuple[FacePose | None, ...]
    normalized3d: NDArray[np.float64]
    distances3d: NDArray[np.float64]
    distances2dp: NDArray[np.float64]

    @property
    def count(self) -> int:
        return len(self.times)


# --------------------------------------------------------------------------
# フレームの読み出し
# --------------------------------------------------------------------------
def _read_image(path: Path) -> NDArray[np.uint8] | None:
    """1 枚読む。非 ASCII のパスでも読めるよう imdecode を使う。"""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        return None
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    return np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def iter_directory_frames(directory: Path, limit: int) -> Iterator[NDArray[np.uint8]]:
    """ディレクトリ内の画像を、ファイル名順に流す。"""
    if not directory.is_dir():
        raise SystemExit(f"ディレクトリが見つかりません: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"画像が見つかりません: {directory}")
    for path in paths[:limit]:
        image = _read_image(path)
        if image is None:
            print(f"  読み込めないので飛ばします: {path.name}")
            continue
        yield image


def video_fps(path: Path, fallback: float) -> float:
    """動画の fps。取れない容器では fallback を使う。"""
    capture = cv2.VideoCapture(str(path))
    try:
        value = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return value if isfinite(value) and value > 0.0 else fallback


def iter_video_frames(path: Path, limit: int) -> Iterator[NDArray[np.uint8]]:
    """動画の先頭から limit 枚を流す。フレーム間隔を一定に保つため間引かない。"""
    if not path.is_file():
        raise SystemExit(f"動画が見つかりません: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"動画を開けません: {path}")
    try:
        for _ in range(limit):
            ok, frame = capture.read()
            if not ok:
                return
            yield np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    finally:
        capture.release()


# --------------------------------------------------------------------------
# 検出（既存 adapter をそのまま使う）
# --------------------------------------------------------------------------
#: 検出モードと、MediaPipe の RunningMode・adapter クラスの対応。--mode が
#: どの検出器に届くかはこの表だけで決まるので、self-test がここを直接検査する。
RUNNING_MODE_BY_MODE: dict[str, str] = {"image": "IMAGE", "video": "VIDEO"}


def _result_points_3d(result: Any) -> NDArray[np.float64] | None:
    """MediaPipe の生 result から、正規化 xyz を (N, 3) で取り出す。"""
    faces = list(getattr(result, "face_landmarks", []) or [])
    if len(faces) != 1:
        return None
    return np.asarray(
        [(float(point.x), float(point.y), float(point.z)) for point in faces[0]],
        dtype=np.float64,
    )


class _DepthTap:
    """MediaPipe の生 result を横取りして z を控える。

    domain の `LandmarkFrame` は xy しか持たない。src/ を一切変えずに z を読む
    ため、adapter が公開している `detector_factory` の差し込み口を使い、実
    detector を包む。生成には adapter 自身の `_create_detector` を使うので、
    graph も `FaceLandmarkerOptions` もアプリ本体とまったく同じである（ここで
    option を書き写すと、本体が変わったときに黙って乖離する）。
    """

    def __init__(self) -> None:
        self.points: NDArray[np.float64] | None = None

    def factory(self, running_mode: str) -> DetectorFactory:
        def build(model_path: Path) -> Any:
            return _TappedDetector(_create_detector(model_path, running_mode), self)

        return build


class _TappedDetector:
    """実 detector をそのまま通し、通りがけに z を `_DepthTap` へ渡すだけの包み。"""

    def __init__(self, inner: Any, tap: _DepthTap) -> None:
        self._inner = inner
        self._tap = tap

    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any:
        return self._record(self._inner.detect_for_video(image, timestamp_ms))

    def detect(self, image: Any) -> Any:
        return self._record(self._inner.detect(image))

    def _record(self, result: Any) -> Any:
        self._tap.points = _result_points_3d(result)
        return result

    def close(self) -> None:
        self._inner.close()


class LandmarkStream:
    """1 系列分の検出。IMAGE / VIDEO モードの差をここだけに閉じ込める。

    静止した被写体のノイズ床を測る目的では IMAGE モードが既定である。VIDEO
    モードはアプリ本体が使うモードなので、両方を測れるようにしてある。

    注意: MediaPipe Tasks 0.10.35 の FaceLandmarker は、VIDEO モードでもフレーム
    間の状態を持たない。同じ 1 枚を「11 枚流した後の detector」と「新品の
    detector」へ与えると結果が bit 単位で一致し、IMAGE モードの結果とも一致する
    （take01.mov 600 枚で確認）。つまり追跡による誤差の相関は発生しない。
    `running_mode` を出力へ出しているのは、「両モードの数値が同じ」ことを
    --mode の配線不良と取り違えないためである。
    """

    def __init__(
        self,
        *,
        model_path: Path,
        model_version: str,
        mode: str,
        detector_factory: DetectorFactory | None = None,
    ) -> None:
        if mode not in RUNNING_MODE_BY_MODE:
            raise SystemExit(f"未知の検出モードです: {mode}")
        self._mode = mode
        self._tap = _DepthTap()
        factory = detector_factory or self._tap.factory(RUNNING_MODE_BY_MODE[mode])
        if mode == "video":
            self._video: MediaPipeFaceLandmarker | None = MediaPipeFaceLandmarker(
                model_path=model_path,
                model_version=model_version,
                detector_factory=factory,
            )
            self._image: MediaPipeImageLandmarker | None = None
        else:
            self._video = None
            self._image = MediaPipeImageLandmarker(
                model_path=model_path,
                model_version=model_version,
                detector_factory=factory,
            )

    @property
    def model_sha256(self) -> str:
        detector = self._video or self._image
        assert detector is not None
        return detector.model_sha256

    @property
    def running_mode(self) -> str:
        """実際に構築した MediaPipe の RunningMode 名。--mode の配線の証拠。"""
        return "VIDEO" if self._video is not None else "IMAGE"

    @property
    def detector_class(self) -> str:
        detector = self._video or self._image
        assert detector is not None
        return type(detector).__name__

    @property
    def last_points_3d(self) -> NDArray[np.float64] | None:
        """直前の検出の正規化 xyz。z を取れなかったフレームでは None。"""
        return self._tap.points

    def detect(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        if self._video is not None:
            return self._video.detect_video(
                rgb_image,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )
        assert self._image is not None
        return self._image.detect(rgb_image, frame_id=frame_id)

    def close(self) -> None:
        if self._video is not None:
            self._video.close()
        if self._image is not None:
            self._image.close()


def _frame_points(frame: LandmarkFrame) -> NDArray[np.float64] | None:
    """単一顔のフレームだけを画素座標へ。それ以外は測定に使わない。"""
    if not frame.has_single_face:
        return None
    try:
        return to_pixel_array(frame)
    except LandmarkError:
        return None


def resolve_model(explicit: Path | None) -> tuple[Path, str]:
    """モデルの場所と版を決める。既定は manifest 検証済みの `models/`。"""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"モデルが見つかりません: {explicit}")
        return explicit, "local-override"
    asset = find_asset(load_manifest(), "face_landmarker")
    store = ModelStore(models_dir=DEFAULT_MODELS_DIR)
    if store.status(asset) is not InstallStatus.READY:
        raise SystemExit(
            "検証済みモデルが必要です: uv run python scripts/bootstrap_models.py --accept-manifest"
        )
    return store.path_for(asset), asset.version


def capture_detections(
    frames: Iterator[NDArray[np.uint8]],
    *,
    model_path: Path,
    model_version: str,
    mode: str,
    with_mirror: bool,
    delta_seconds: float,
    source_kind: str,
) -> Capture:
    """全フレームを検出する。保持するのは landmark 配列だけで、画像は残さない。"""
    original = LandmarkStream(model_path=model_path, model_version=model_version, mode=mode)
    mirror = (
        LandmarkStream(model_path=model_path, model_version=model_version, mode=mode)
        if with_mirror
        else None
    )
    detections: list[Detection] = []
    seen = 0
    try:
        for index, image in enumerate(frames):
            seen += 1
            # VIDEO モードは狭義単調増加を要求するので、下限を index で押さえる。
            timestamp_ms = max(index, round(index * delta_seconds * 1000.0))
            frame = original.detect(image, frame_id=index, timestamp_ms=timestamp_ms)
            points = _frame_points(frame)
            if points is None:
                continue

            height, width = image.shape[:2]
            mirrored: NDArray[np.float64] | None = None
            mirrored3d: NDArray[np.float64] | None = None
            if mirror is not None:
                flipped = np.ascontiguousarray(image[:, ::-1])
                mirrored = _frame_points(
                    mirror.detect(flipped, frame_id=index, timestamp_ms=timestamp_ms)
                )
                mirrored3d = to_pixel_3d(mirror.last_points_3d, width=width, height=height)
            detections.append(
                Detection(
                    index=index,
                    points=points,
                    mirrored=mirrored,
                    pose=frame.pose,
                    points3d=to_pixel_3d(original.last_points_3d, width=width, height=height),
                    mirrored3d=mirrored3d,
                )
            )
            if seen % 25 == 0:
                print(f"  {seen} 枚処理", flush=True)
    finally:
        running_mode = original.running_mode
        detector_class = original.detector_class
        original.close()
        if mirror is not None:
            mirror.close()

    return Capture(
        detections=tuple(detections),
        frames_seen=seen,
        delta_seconds=delta_seconds,
        source_kind=source_kind,
        mode=mode,
        model_version=model_version,
        model_sha256=original.model_sha256,
        running_mode=running_mode,
        detector_class=detector_class,
    )


# --------------------------------------------------------------------------
# 指標と統計
# --------------------------------------------------------------------------
def distance_of(points: NDArray[np.float64]) -> float | None:
    """§8.5 の総合距離 d。求められない入力では None。"""
    try:
        result = compute_symmetry(points, CONFIG)
    except SymmetryError:
        return None
    return result.overall


def normalized_of(points: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """§8.3 の正規化座標。単位は両眼間距離。"""
    try:
        return normalize_landmarks(
            points,
            left_eye_corners=CONFIG.left_eye_corners,
            right_eye_corners=CONFIG.right_eye_corners,
        )
    except GeometryError:
        return None


# --------------------------------------------------------------------------
# 3D 座標の単位あわせ（テストD）
# --------------------------------------------------------------------------
Z_SCALE_NOTE = """
  z の扱い（この節の数値がどの単位で出ているか）

  1. MediaPipe の landmark は x が幅、y が高さで正規化されている。z は
     「x とおよそ同じスケール」と規定されているので、画素相当へ戻すときは
     x と同じ係数 —— 画像の *幅* —— を掛ける。y だけ高さを掛ける。
     これで 3 軸が同じ長さの単位に揃い、既存の `to_pixel_array` の xy とも一致する。

  2. そのうえで §8.3 と同じ相似変換を 3D へ広げる。原点は両眼アンカーの中点
     （z も同じ 4 点から取る）、回転は roll 補正で z 軸まわり、スケールは
     x/y/z へ *同一の係数* を掛ける。回転と平行移動は 3D の剛体変換なので、
     Procrustes 重ね合わせの残差 d_3d を変えない。効くのはスケールだけである。

  3. そのスケールに **3D の両眼間距離** を使う。§8.3 の 2D 投影距離を使うと、
     yaw で両眼間が cos(yaw) に縮むぶん d_3d が 1/cos(yaw) 倍に膨らみ、
     いちばん測りたい「d_3d と |yaw| の相関」へ姿勢が裏口から戻ってくる。
     yaw = 0 では両者は一致する。差が気になる場合のために、2D 投影スケール
     での値も 3) に併記してある。
"""


def to_pixel_3d(
    raw: NDArray[np.float64] | None,
    *,
    width: int,
    height: int,
) -> NDArray[np.float64] | None:
    """MediaPipe の正規化 xyz を、3 軸そろった画素相当座標へ（上記 1.）。"""
    if raw is None or raw.size == 0:
        return None
    return np.asarray(raw, dtype=np.float64) * np.array([width, height, width], dtype=np.float64)


def _eye_anchor_depth(points3d: NDArray[np.float64]) -> float:
    """両眼アンカーの中点の z。§8.3 の `center` を z 方向へ広げたもの。"""
    left = float(points3d[list(CONFIG.left_eye_corners), 2].mean())
    right = float(points3d[list(CONFIG.right_eye_corners), 2].mean())
    return (left + right) / 2.0


def normalized_3d_of(
    points3d: NDArray[np.float64],
    *,
    scale_from_3d: bool = True,
) -> NDArray[np.float64] | None:
    """§8.3 の相似変換を 3D へ広げる（上記 2. と 3.）。単位は両眼間距離。"""
    if points3d.ndim != 2 or points3d.shape[1] != 3 or not np.isfinite(points3d).all():
        return None
    flat = points3d[:, :2]
    try:
        transform = normalization_transform(
            flat,
            left_eye_corners=CONFIG.left_eye_corners,
            right_eye_corners=CONFIG.right_eye_corners,
        )
    except GeometryError:
        return None

    scale = transform.scale
    if scale_from_3d:
        left = points3d[list(CONFIG.left_eye_corners)].mean(axis=0)
        right = points3d[list(CONFIG.right_eye_corners)].mean(axis=0)
        scale = float(np.linalg.norm(right - left))
    if scale <= 0.0 or not isfinite(scale):
        return None

    plane = ((flat - transform.center) @ transform.rotation.T) / scale
    depth = (points3d[:, 2] - _eye_anchor_depth(points3d)) / scale
    return np.column_stack([plane, depth])


def distance_3d_of(points3d: NDArray[np.float64]) -> float | None:
    """3D object symmetry の残差 RMS。`symmetry_level3_probe.distance_3d` そのもの。"""
    if not np.isfinite(points3d).all():
        return None
    value = distance_3d(points3d)
    return value if isfinite(value) else None


def distance_2d_procrustes_of(normalized: NDArray[np.float64]) -> float | None:
    """方式② —— z を捨て、xy だけで object symmetry を解いた残差 RMS。

    `distance_3d` の手順（鏡映 → sigma でペア名入替 → 並進と回転で Procrustes
    重ね合わせ → 残差 RMS）は次元に依存しない。(N, 2) をそのまま渡せば 2D の
    object symmetry になるので、新しい数学は入れていない。

    入力は §8.3 の正規化座標である。単位は 2D 投影の両眼間距離なので、yaw が
    あると 1/cos(yaw) 倍に膨らむ。この項は z 無しでは原理的に消せず、それこそが
    方式② と方式③ を分ける量である。ここで 3D の両眼間距離を使って補正しては
    *ならない*。補正すると「z を使わない方式」ではなくなる。
    """
    if normalized.ndim != 2 or normalized.shape[1] < 2:
        return None
    plane = np.ascontiguousarray(normalized[:, :2], dtype=np.float64)
    if not np.isfinite(plane).all():
        return None
    value = distance_3d(plane)
    return value if isfinite(value) else None


def procrustes_residual(points3d: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """`distance_3d` と同一手順で重ね合わせ、残差ベクトルを (M, 3) で返す。

    テストD の d_3d はこの残差の RMS そのものである（d_3d² = 残差の二乗平均）。
    テストG が xy と z へ分けて読むために、スカラーへ潰す前の形が要る。
    行の並びは `MEASURED_INDICES` と同じで、`distance_3d` が使う点集合に一致する。

    `symmetry_level3_probe.distance_3d` を書き換えずに手順を写しているので、
    両者が一致していることを self-test が毎回確かめる（`_check_residual_matches`）。
    """
    if points3d.ndim != 2 or points3d.shape[1] != 3 or not np.isfinite(points3d).all():
        return None
    used = list(MEASURED_INDICES)
    position = {landmark: order for order, landmark in enumerate(used)}
    original = points3d[used]

    mirrored = original.copy()
    mirrored[:, 0] *= -1.0
    swap = np.arange(len(used))
    for left, right in CONFIG.all_pairs:
        swap[position[left]], swap[position[right]] = position[right], position[left]
    mirrored = mirrored[swap]

    centred = original - original.mean(axis=0)
    mirrored_centred = mirrored - mirrored.mean(axis=0)
    left_vectors, _, right_vectors = np.linalg.svd(mirrored_centred.T @ centred)
    rotation = left_vectors @ right_vectors
    if np.linalg.det(rotation) < 0:
        left_vectors[:, -1] *= -1.0
        rotation = left_vectors @ right_vectors
    aligned = mirrored_centred @ rotation
    return np.asarray(centred - aligned, dtype=np.float64)


# --------------------------------------------------------------------------
# 既知量の注入（テストE）
# --------------------------------------------------------------------------
def part_injection_indices(normalized: NDArray[np.float64], part: str) -> tuple[int, ...]:
    """指定した部位のペアのうち、正規化フレームで x が小さい側の index。

    「片側だけ」を厳密に守るため、ペア表の第 1 要素ではなく実際の位置で選ぶ。
    ペア表の左右の並びが顔のどちら側かは bundle の導出手順に依存するので、
    ここで幾何から決めておけば、bundle が変わっても片側性は崩れない。
    """
    chosen: list[int] = []
    for left, right in CONFIG.parts[part].pairs:
        chosen.append(left if normalized[left][0] < normalized[right][0] else right)
    return tuple(sorted(chosen))


def mouth_injection_indices(normalized: NDArray[np.float64]) -> tuple[int, ...]:
    """mouth ペアの片側。テストE が既定で使う注入部位。"""
    return part_injection_indices(normalized, "mouth")


#: 領域場の広がり（両眼間距離比）。値の根拠は `injection_weights` の docstring。
REGION_SIGMA_IOD: float = 0.12

#: 折り返しの判定に使う、各点の近傍数。メッシュの三角形分割を持ち出さずに
#: 変位場の局所勾配を見積もるためのもの。
FOLD_NEIGHBOURS: int = 4


def injection_weights(
    normalized: NDArray[np.float64],
    indices: tuple[int, ...],
    *,
    sigma_iod: float,
) -> NDArray[np.float64]:
    """注入する変位の、landmark ごとの重み（0〜1）。

    `sigma_iod` が 0 なら指定した点だけが 1 で、他は 0 になる（docs テストC が
    書いている「mouth ペアの片側だけ」の場）。正の値を与えると、指定点からの
    距離に対する Gauss 関数

        w(j) = exp(-(dist(j) / sigma)^2),  dist(j) = min_i ||n_j - n_i||

    になり、近傍の点も一緒に動くので区分アフィンが折り返さない。

    sigma = 0.12 は 2 つの制約の間で決めてある。Gauss の勾配の最大は
    sqrt(2/e)/sigma なので、8mm（0.127 IOD）を折り返さずに入れるには
    sigma > 0.109 が要る。一方 mouth ペアの最短間隔は 0.176 IOD なので、
    sigma を広げるほど相手側も同じ向きへ動いて左右差が薄まる。0.12 では相手側
    の重みが exp(-(0.176/0.12)^2) = 0.12 に留まる。
    """
    weights = np.zeros(len(normalized), dtype=np.float64)
    if sigma_iod <= 0.0:
        weights[list(indices)] = 1.0
        return weights
    gaps = np.linalg.norm(
        normalized[:, None, :2] - normalized[None, list(indices), :2], axis=2
    ).min(axis=1)
    return np.asarray(np.exp(-((gaps / sigma_iod) ** 2)), dtype=np.float64)


def inject_field(
    normalized: NDArray[np.float64],
    weights: NDArray[np.float64],
    delta_iod: float,
) -> NDArray[np.float64]:
    """正規化フレームの x 方向へ、重み付きの変位 delta_iod * w を加える。

    2D (N, 2) と 3D (N, 3) の両方に同じ規則で適用する。正規化フレームの x 軸は
    両眼アンカーを結ぶ向きなので、これは顔に固定した「横」方向である。
    """
    shifted = np.array(normalized, dtype=np.float64, copy=True)
    shifted[:, 0] += delta_iod * weights
    return shifted


def fold_threshold_mm(
    normalized: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """変位場が三角形を裏返し始める δ（mm）。

    区分アフィン変形は landmark の間を線形に引き伸ばす。隣り合う 2 点の相対
    変位がその 2 点の間隔を超えると、三角形は裏返る。裏返った三角形は退化判定を
    すり抜けて逆向きのアフィン写像で転送されるので、その領域は「δ だけずれた顔」
    ではなく壊れた画素になる。

    三角形分割を持ち出さずに、各点とその近傍 `FOLD_NEIGHBOURS` 点との間で
    |Δw| / 距離 の最大値を取り、その逆数を閾値とする。
    """
    plane = normalized[:, :2]
    distances = np.linalg.norm(plane[:, None, :] - plane[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1)[:, :FOLD_NEIGHBOURS]
    rows = np.arange(len(plane))[:, None]
    gaps = distances[rows, order]
    slopes = np.abs(weights[:, None] - weights[order]) / gaps
    steepest = float(slopes.max())
    return float("inf") if steepest <= 0.0 else IPD_MM / steepest


def summarize(values: NDArray[np.float64]) -> dict[str, float]:
    """1 本の分布の要約。単位は入力のまま（多くは両眼間距離）。"""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        return {"count": 0.0}
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def spread_json(stats: dict[str, float]) -> dict[str, Any]:
    """長さの分布を、両眼間距離比と mm 換算の両方で残す。"""
    keys = ("mean", "std", "median", "p90", "min", "max")
    if not stats.get("count"):
        return {"count": 0}
    return {
        "count": int(stats["count"]),
        "iod": {key: stats[key] for key in keys},
        "mm": {key: stats[key] * IPD_MM for key in keys},
    }


def build_series(capture: Capture) -> Series:
    """d と正規化座標をそろえた時系列にする。どちらも取れないフレームは落とす。

    3D 側は z を取れたフレームだけが値を持ち、残りは NaN で埋める。2D 側の
    採否と添字を共有させることで、姿勢との対応付けが崩れないようにしている。
    """
    times: list[float] = []
    distances: list[float] = []
    normals: list[NDArray[np.float64]] = []
    poses: list[FacePose | None] = []
    normals3d: list[NDArray[np.float64]] = []
    distances3d: list[float] = []
    distances2dp: list[float] = []
    for item in capture.detections:
        distance = distance_of(item.points)
        normal = normalized_of(item.points)
        if distance is None or normal is None:
            continue
        times.append(item.index * capture.delta_seconds)
        distances.append(distance)
        normals.append(normal)
        poses.append(item.pose)
        plane = distance_2d_procrustes_of(normal)
        distances2dp.append(float("nan") if plane is None else plane)

        solid = normalized_3d_of(item.points3d) if item.points3d is not None else None
        solid_distance = distance_3d_of(solid) if solid is not None else None
        if solid is None or solid_distance is None:
            normals3d.append(np.full((len(normal), 3), np.nan, dtype=np.float64))
            distances3d.append(float("nan"))
        else:
            normals3d.append(solid)
            distances3d.append(solid_distance)
    if not normals:
        return Series(
            times=np.zeros(0, dtype=np.float64),
            distances=np.zeros(0, dtype=np.float64),
            normalized=np.zeros((0, 0, 2), dtype=np.float64),
            poses=(),
            normalized3d=np.zeros((0, 0, 3), dtype=np.float64),
            distances3d=np.zeros(0, dtype=np.float64),
            distances2dp=np.zeros(0, dtype=np.float64),
        )
    return Series(
        times=np.asarray(times, dtype=np.float64),
        distances=np.asarray(distances, dtype=np.float64),
        normalized=np.stack(normals),
        poses=tuple(poses),
        normalized3d=np.stack(normals3d),
        distances3d=np.asarray(distances3d, dtype=np.float64),
        distances2dp=np.asarray(distances2dp, dtype=np.float64),
    )


def landmark_spread(normalized: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    """landmark ごとの位置のばらつき。平均位置まわりの 2D 距離で測る。

    第 1 要素が RMS（標準偏差の 2D 版）、第 2 要素が平均絶対偏差である。後者は
    MediaPipe 論文の「IOD 正規化 MAD」と同種の量なので、3.96% と直接比較できる。
    """
    centre = normalized.mean(axis=0)
    radial = np.linalg.norm(normalized - centre, axis=2)
    rms = np.sqrt((radial**2).mean(axis=0))
    mad = radial.mean(axis=0)
    return np.asarray(rms, dtype=np.float64), np.asarray(mad, dtype=np.float64)


def landmark_spread_3d(
    normalized3d: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """3D 正規化座標での、landmark ごとの位置のばらつき。

    第 1 要素が 3 軸まとめた RMS、第 2 要素が z だけの RMS である。セクション K
    は前者で平滑化の効きを見て、後者で「下がっているのは z か xy か」を見る。
    """
    centre = normalized3d.mean(axis=0)
    radial = np.linalg.norm(normalized3d - centre, axis=2)
    rms = np.sqrt((radial**2).mean(axis=0))
    depth = np.sqrt(((normalized3d[:, :, 2] - centre[:, 2]) ** 2).mean(axis=0))
    return np.asarray(rms, dtype=np.float64), np.asarray(depth, dtype=np.float64)


# --------------------------------------------------------------------------
# 平滑化（domain/smoothing.py と同じ式）
# --------------------------------------------------------------------------
def ema_array(
    values: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    tau_seconds: float,
    max_gap_seconds: float,
) -> NDArray[np.float64]:
    """`TimeBasedEma` と同じ式を、任意 shape の配列へ要素ごとに適用する。

        alpha_t = 1 - exp(-delta_t / tau)
        ema_t   = alpha_t * raw_t + (1 - alpha_t) * ema_(t-1)

    現行実装がスカラー（d）にしか適用できないので、座標側の効果を測るために
    ここで同じ式を配列へ広げている。判断は何も足していない。
    """
    smoothed = np.empty_like(values)
    current = values[0]
    smoothed[0] = current
    for step in range(1, len(values)):
        delta = float(times[step] - times[step - 1])
        if delta < 0.0 or delta >= max_gap_seconds:
            current = values[step]
        else:
            alpha = 1.0 - exp(-delta / tau_seconds) if delta > 0.0 else 0.0
            current = alpha * values[step] + (1.0 - alpha) * current
        smoothed[step] = current
    return smoothed


def app_tau_for(alpha: float, *, delta_seconds: float = APP_DELTA_SECONDS) -> float:
    """同じ alpha を、アプリの解析間隔で得るための時定数。

        alpha = 1 - exp(-delta_t / tau)  ⇔  tau = -delta_t / ln(1 - alpha)

    平滑化の効きを決めるのは 1 ステップあたりの alpha であって tau ではない。
    撮影 fps と アプリの解析 cadence は違うので、表 C の tau をそのまま
    `TimeBasedEma` へ設定すると効きが変わる。この換算を挟むと、実測した alpha
    をそのままアプリの設定値へ持ち込める。
    """
    if not 0.0 < alpha < 1.0:
        return float("inf") if alpha <= 0.0 else 0.0
    return -delta_seconds / log(1.0 - alpha)


def ema_scalar(
    values: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    tau_seconds: float,
    max_gap_seconds: float,
) -> NDArray[np.float64]:
    """現行実装（`domain.smoothing.TimeBasedEma`）そのものを通した d の系列。"""
    ema = TimeBasedEma(tau_seconds=tau_seconds, max_gap_seconds=max_gap_seconds)
    return np.asarray(
        [ema.update(float(value), at=float(at)) for value, at in zip(values, times, strict=True)],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# 出力の体裁
# --------------------------------------------------------------------------
def heading(letter: str, title: str) -> None:
    print()
    print("=" * 100)
    print(f"[{letter}] {title}")
    print("=" * 100)


def both_units(value: float) -> str:
    """両眼間距離比と mm 換算を 1 つの欄にまとめる。"""
    return f"{value:8.5f} ({value * IPD_MM:6.3f}mm)"


def percent(distance: float) -> str:
    """UI の `format_symmetry` と同じ読み替え。"""
    return f"{min(100.0, max(0.0, (1.0 - distance) * 100.0)):6.2f}%"


def print_spread(label: str, stats: dict[str, float]) -> None:
    if not stats.get("count"):
        print(f"  {label:<24} —")
        return
    print(
        f"  {label:<24} 中央値 {both_units(stats['median'])}   "
        f"p90 {both_units(stats['p90'])}   最大 {both_units(stats['max'])}"
    )


def print_landmark_spread_table(normalized: NDArray[np.float64]) -> dict[str, Any]:
    """landmark ごとのばらつきの表を出し、同じ内容を JSON 用に返す。

    テストB の 2) と、姿勢で層別した 5) が同じ表を使うので、書式をここに一本化
    してある。両者を並べて読むのが目的なので、行の並びも桁もそろえる。
    """
    rms, mad = landmark_spread(normalized)
    available = normalized.shape[1]
    groups: tuple[tuple[str, list[int]], ...] = (
        (f"計測に使う {len(MEASURED_INDICES)} 点", [i for i in MEASURED_INDICES if i < available]),
        (
            f"正中線 {len(CONFIG.midline_indices)} 点",
            [i for i in CONFIG.midline_indices if i < available],
        ),
        (f"全 {available} 点", list(range(available))),
    )
    print(f"     {'対象':<18} {'RMS 中央値':>20} {'RMS p90':>20} {'MAD 中央値':>20}")
    print("     " + "-" * 80)
    spread_report: dict[str, Any] = {}
    for label, group in groups:
        if not group:
            continue
        print(
            f"     {label:<18} {both_units(float(np.median(rms[group]))):>20} "
            f"{both_units(float(np.percentile(rms[group], 90))):>20} "
            f"{both_units(float(np.median(mad[group]))):>20}"
        )
        spread_report[label] = {
            "rms": spread_json(summarize(rms[group])),
            "mad": spread_json(summarize(mad[group])),
        }
    return spread_report


# --------------------------------------------------------------------------
# テストA — 鏡像テスト
# --------------------------------------------------------------------------
def section_a(capture: Capture, report: dict[str, Any]) -> None:
    heading("A", "鏡像テスト — 検出器の左右バイアス")
    print(MIRROR_NOTE)

    usable = [item for item in capture.detections if item.mirrored is not None]
    if not usable:
        print(
            "  鏡像側を検出できたフレームがありません（--section A を外すと B/C は実行できます）。"
        )
        report["mirror"] = {"frames": 0}
        return

    indices = list(SIGMA_INDICES)
    partners = [MIRROR_PERMUTATION[index] for index in indices]
    flip = np.array([-1.0, 1.0], dtype=np.float64)

    original_d: list[float] = []
    mirror_d: list[float] = []
    bias_frames: list[NDArray[np.float64]] = []

    for item in usable:
        assert item.mirrored is not None
        left = distance_of(item.points)
        right = distance_of(item.mirrored)
        normal_original = normalized_of(item.points)
        normal_mirror = normalized_of(item.mirrored)
        if left is None or right is None or normal_original is None or normal_mirror is None:
            continue
        original_d.append(left)
        mirror_d.append(right)
        remapped = normal_mirror[partners] * flip
        bias_frames.append(remapped - normal_original[indices])

    if not bias_frames:
        print("  比較できたフレームがありません。")
        report["mirror"] = {"frames": 0}
        return

    original_array = np.asarray(original_d, dtype=np.float64)
    mirror_array = np.asarray(mirror_d, dtype=np.float64)
    signed = mirror_array - original_array
    absolute = np.abs(signed)

    stacked = np.stack(bias_frames)
    per_landmark_bias = np.linalg.norm(stacked, axis=2)
    systematic = np.linalg.norm(stacked.mean(axis=0), axis=1)

    print(f"  比較できたフレーム: {len(original_array)} / {len(capture.detections)}")
    print()
    print("  1) d の水準（この 2 つが離れていたら、指標そのものが左右で違う値を出している）")
    print(
        f"     d(I)          平均 {original_array.mean():.5f} "
        f"({original_array.mean() * IPD_MM:.3f}mm, {percent(float(original_array.mean()))})"
    )
    print(
        f"     d(mirror(I))  平均 {mirror_array.mean():.5f} "
        f"({mirror_array.mean() * IPD_MM:.3f}mm, {percent(float(mirror_array.mean()))})"
    )
    print()
    print("  2) |d(I) - d(mirror(I))| — 幾何学的に正しい検出器なら 0 になる量")
    print_spread("差の大きさ", summarize(absolute))
    print(
        f"  {'符号付き平均':<24} {both_units(float(signed.mean()))}  "
        "（正なら鏡像側が大きい、つまり一方向の系統バイアス）"
    )
    print()
    print("  3) landmark ごとの左右バイアス（sigma で index を入れ替えたうえでの差）")
    print_spread("フレームごとの差", summarize(per_landmark_bias))
    print_spread("うち系統成分", summarize(systematic))
    print()
    print(
        f"  参考: 「対称な成人」と「非対称な成人」を分ける差 = {SIGNAL_SPAN:.5f} "
        f"({SIGNAL_SPAN * IPD_MM:.2f}mm)"
    )
    print("  → 3) の系統成分がこの幅と同程度以上なら、表示値の一部は顔の形ではなく")
    print("    検出器の左右非対称でできている。2) が 0 に近くても 3) は大きくなりうる。")

    report["mirror"] = {
        "frames": len(original_array),
        "distance_original": spread_json(summarize(original_array)),
        "distance_mirrored": spread_json(summarize(mirror_array)),
        "distance_absolute_difference": spread_json(summarize(absolute)),
        "distance_signed_difference_mean": {
            "iod": float(signed.mean()),
            "mm": float(signed.mean()) * IPD_MM,
        },
        "landmark_bias": spread_json(summarize(per_landmark_bias)),
        "landmark_bias_systematic": spread_json(summarize(systematic)),
        "landmark_count": len(indices),
    }


# --------------------------------------------------------------------------
# テストB — 再現性テスト
# --------------------------------------------------------------------------
def section_b(capture: Capture, series: Series, report: dict[str, Any]) -> None:
    heading("B", "再現性テスト — ノイズ床")
    if series.count == 0:
        print("  測定できたフレームがありません。")
        report["repeatability"] = {"frames": 0}
        return

    distances = series.distances
    print(
        f"  検出できたフレーム: {series.count} / {capture.frames_seen}"
        f"  （検出率 {series.count / max(1, capture.frames_seen):.1%}）"
    )
    print(
        f"  モード: {capture.mode} → {capture.detector_class} / RunningMode.{capture.running_mode}"
    )
    if series.count < MIN_USEFUL_FRAMES:
        print(f"  ※ {MIN_USEFUL_FRAMES} 枚未満なので、以下の統計は目安にとどまる。")
    print()

    print("  1) d の分布（フレームごとの生値）")
    print(
        f"     平均 {both_units(float(distances.mean()))}   "
        f"標準偏差 {both_units(float(distances.std(ddof=1)) if series.count > 1 else 0.0)}"
    )
    print(
        f"     最小 {both_units(float(distances.min()))}   "
        f"最大 {both_units(float(distances.max()))}"
    )
    noise_floor = float(distances.std(ddof=1)) if series.count > 1 else 0.0
    print(
        f"     表示に直すと {percent(float(distances.max()))} 〜 {percent(float(distances.min()))}"
    )
    print()
    print("     ノイズ床（d の標準偏差）と、測りたい信号の比較")
    print(f"       ノイズ床           {both_units(noise_floor)}")
    print(f"       信号全幅           {both_units(SIGNAL_SPAN)}  ← 対称な成人 ↔ 非対称な成人")
    if noise_floor > 0.0:
        print(f"       信号 ÷ ノイズ床   {SIGNAL_SPAN / noise_floor:6.2f}x")
        print("       → 1.0 を下回るなら、この指標は人と人の差を区別できていない。")
    print()

    print("  2) §8.3 正規化座標での、landmark ごとの位置のばらつき")
    print("     これが MediaPipe の実測誤差。論文の IOD 正規化 MAD 3.96% と直接比較できる。")
    print()
    spread_report = print_landmark_spread_table(series.normalized)
    print()
    print(
        f"     参考  MediaPipe Full の公称 MAD  {MEDIAPIPE_FULL_MAD:.4f} "
        f"({MEDIAPIPE_FULL_MAD * IPD_MM:.2f}mm)"
    )
    print(
        f"           人間アノテータのばらつき    {HUMAN_ANNOTATOR_MAD:.4f} "
        f"({HUMAN_ANNOTATOR_MAD * IPD_MM:.2f}mm)"
    )
    print("     ※ ここで測っているのは再現性（同じ被写体を繰り返し測ったときのばらつき）で、")
    print("       論文の MAD は正解に対する誤差である。再現性は誤差の下限にしかならない。")
    print("       また被写体の微動もここに含まれるので、4) の姿勢分布と併せて読むこと。")
    print()

    print("  3) 検出できたフレーム数")
    print(f"     {series.count} / {capture.frames_seen}")
    print()

    print("  4) 姿勢の分布（「姿勢が動いた分」と「検出誤差」を切り分ける材料）")
    pose_report: dict[str, Any] = {}
    poses = [pose for pose in series.poses if pose is not None]
    if not poses:
        print("     姿勢を取得できたフレームがありません。")
    else:
        print(f"     {'':<8} {'平均':>10} {'標準偏差':>10} {'最小':>10} {'最大':>10} {'範囲':>10}")
        print("     " + "-" * 62)
        axes: tuple[tuple[str, Callable[[FacePose], float]], ...] = (
            ("yaw", lambda pose: pose.yaw_deg),
            ("pitch", lambda pose: pose.pitch_deg),
            ("roll", lambda pose: pose.roll_deg),
        )
        for name, reader in axes:
            values = np.asarray([reader(pose) for pose in poses], dtype=np.float64)
            width = float(values.max() - values.min())
            deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            print(
                f"     {name:<8} {values.mean():9.2f}° {deviation:9.2f}° "
                f"{values.min():9.2f}° {values.max():9.2f}° {width:9.2f}°"
            )
            pose_report[name] = summarize(values)

        yaw = np.asarray([abs(pose.yaw_deg) for pose in poses], dtype=np.float64)
        paired_d = np.asarray(
            [
                distance
                for distance, pose in zip(series.distances, series.poses, strict=True)
                if pose is not None
            ],
            dtype=np.float64,
        )
        if len(yaw) > 2 and float(yaw.std()) > 0.0 and float(paired_d.std()) > 0.0:
            correlation = float(np.corrcoef(yaw, paired_d)[0, 1])
            print()
            print(f"     |yaw| と d の相関係数: {correlation:+.3f}")
            print("     → 1 に近いほど、フレーム間の d の揺れは")
            print("       検出誤差ではなく姿勢のブレで説明できる。")
            pose_report["abs_yaw_distance_correlation"] = correlation

    stable_report = _print_stable_yaw_floor(series, noise_floor)

    report["repeatability"] = {
        "frames_detected": series.count,
        "frames_seen": capture.frames_seen,
        "detection_rate": series.count / max(1, capture.frames_seen),
        "mode": capture.mode,
        "detector_class": capture.detector_class,
        "running_mode": capture.running_mode,
        "distance": spread_json(summarize(distances)),
        "noise_floor": {"iod": noise_floor, "mm": noise_floor * IPD_MM},
        "signal_span": {"iod": SIGNAL_SPAN, "mm": SIGNAL_SPAN * IPD_MM},
        "landmark_spread": spread_report,
        "pose_deg": pose_report,
        "stable_yaw": stable_report,
    }


def yaw_values(series: Series) -> NDArray[np.float64]:
    """フレームごとの yaw。姿勢を取れなかったフレームは NaN。"""
    return np.asarray(
        [float("nan") if pose is None else pose.yaw_deg for pose in series.poses],
        dtype=np.float64,
    )


def stable_yaw_mask(series: Series) -> tuple[NDArray[np.bool_], float, float] | None:
    """yaw が中央値の近くに収まっているフレームを選ぶ。

    狭い帯から順に試し、統計が成り立つ枚数（`MIN_USEFUL_FRAMES`）に届いた時点で
    確定する。どの帯でも届かなければ、いちばん広い帯を返して枚数の判断は
    呼び出し側へ委ねる。返り値は (選別 mask, 使った幅, yaw 中央値)。
    """
    yaws = yaw_values(series)
    finite = np.isfinite(yaws)
    if not bool(finite.any()):
        return None
    median = float(np.median(yaws[finite]))
    mask = finite
    band = YAW_BAND_DEGREES[-1]
    for candidate in YAW_BAND_DEGREES:
        mask = finite & (np.abs(yaws - median) <= candidate)
        band = candidate
        if int(mask.sum()) >= MIN_USEFUL_FRAMES:
            break
    return mask, band, median


def _print_stable_yaw_floor(series: Series, overall_floor: float) -> dict[str, Any]:
    """5) 姿勢で層別したノイズ床。検出器そのものの再現性へ近づける。"""
    print()
    print("  5) 姿勢で層別したノイズ床（姿勢のドリフトを除いた、検出器だけの再現性）")
    print("     1) のノイズ床は姿勢の揺れを含むので、検出器の再現性を過大評価している。")
    print("     yaw が中央値の近くに留まったフレームだけで 1) と 2) を計算し直す。")
    print()

    selection = stable_yaw_mask(series)
    if selection is None:
        print("     姿勢を取得できたフレームがないので、層別できません。")
        return {"frames": 0}

    mask, band, median = selection
    count = int(mask.sum())
    print(
        f"     yaw 中央値 {median:+.2f}° / 帯域 ±{band:.1f}° / "
        f"該当 {count} 枚（全 {series.count} 枚）"
    )
    if count < 2:
        print("     ※ 2 枚未満なので、ばらつきを計算できません。")
        return {"frames": count, "band_deg": band, "median_yaw_deg": median}
    if count < MIN_USEFUL_FRAMES:
        print(
            f"     ※ {MIN_USEFUL_FRAMES} 枚未満（いちばん広い帯 ±{YAW_BAND_DEGREES[-1]:.1f}° "
            "でも足りない）。以下は目安にとどまる。"
        )
    print()

    subset = series.distances[mask]
    floor = float(subset.std(ddof=1))
    print("     1') d の分布と、そのノイズ床")
    print(f"        平均 {both_units(float(subset.mean()))}   標準偏差 {both_units(floor)}")
    print(f"        全体のノイズ床     {both_units(overall_floor)}")
    print(f"        姿勢を除いた床     {both_units(floor)}", end="")
    if overall_floor > 0.0:
        print(f"   （全体の {floor / overall_floor:.0%}）")
    else:
        print()
    if floor > 0.0:
        print(f"        信号 ÷ この床     {SIGNAL_SPAN / floor:6.2f}x")
    print()
    print("     2') landmark ごとの位置のばらつき（同じ部分集合で）")
    subset_spread = print_landmark_spread_table(series.normalized[mask])
    print()
    print("     → 全体との差が、d の揺れのうち姿勢のドリフトで説明できる分である。")

    return {
        "frames": count,
        "band_deg": band,
        "median_yaw_deg": median,
        "distance": spread_json(summarize(subset)),
        "noise_floor": {"iod": floor, "mm": floor * IPD_MM},
        "noise_floor_ratio_to_overall": floor / overall_floor if overall_floor > 0.0 else None,
        "landmark_spread": subset_spread,
    }


# --------------------------------------------------------------------------
# 予測 — 座標平滑化の効果
# --------------------------------------------------------------------------
def _burn_in(count: int, tau_seconds: float, delta_seconds: float) -> tuple[int, bool]:
    """EMA の立ち上がりを除く枚数と、それが足りているか。

    時定数 3 本分を捨てたいが、系列の 1/3 を超えて捨てると残りが少なすぎる。
    足りない場合は捨てられるだけ捨て、定常に達していないことを呼び出し側へ返す。
    """
    if delta_seconds <= 0.0:
        return 0, False
    required = ceil(3.0 * tau_seconds / delta_seconds)
    burn = int(min(count // 3, required))
    return burn, burn >= required


def section_c(capture: Capture, series: Series, report: dict[str, Any]) -> None:
    heading("C", "予測 — 座標を平滑化すると d はどこまで下がるか（対策10 の事前見積り）")
    if series.count < 2:
        print("  平滑化を評価するにはフレームが足りません。")
        report["smoothing"] = []
        return

    delta = capture.delta_seconds
    # 立ち上がりを潰さないため、実測の間隔に対して十分広い gap を使う。現行の
    # 既定は 2.0 秒で、5fps 以上ならそのままでも影響しない。
    max_gap = max(2.0, 5.0 * delta)
    print(f"  フレーム間隔 delta_t = {delta:.4f} 秒（{capture.fps:.2f} fps）")
    if max_gap > 2.0:
        print(f"  ※ delta_t が大きいので、EMA の reset 閾値を {max_gap:.2f} 秒に広げている")
    print("""
  比べているのは「何を平滑化するか」だけである。

    座標を EMA  … §8.3 の正規化座標へ EMA。誤差は符号を持つので打ち消し合う
    d を EMA    … 現行の domain/smoothing.py。d は距離なので必ず 0 以上で、
                  検出誤差は d を押し上げる方向にしか働かない（正のバイアス）

  理論列は i.i.d. な誤差に対する EMA の定常分散比 alpha/(2-alpha) と、その平方根
  （標準偏差比）である。実測がこれより大きければ、誤差は時間的に相関している
  ——つまり姿勢のドリフトなど、平均では消えない成分が入っている。
""")
    raw_rms, _ = landmark_spread(series.normalized)
    available = series.normalized.shape[1]
    group = [index for index in MEASURED_INDICES if index < available]
    raw_reference = raw_rms[group]

    raw_mean = float(series.distances.mean())
    raw_std = float(series.distances.std(ddof=1))

    print(f"""
  tau は撮影 fps に依存するので、そのままアプリへ持ち込めない。効きを決めるのは
  alpha だけなので、同じ alpha を出すアプリ側の時定数を併記する。

      tau_app = -{APP_DELTA_SECONDS} / ln(1 - alpha)
      （アプリの解析間隔 {APP_DELTA_SECONDS} 秒 = {1.0 / APP_DELTA_SECONDS:.0f}Hz に対する時定数）

  P1 で `TimeBasedEma` へ設定すべき値は tau ではなく、この tau_app である。""")
    print(
        f"  {'tau':>6} {'tau_app':>9} {'alpha':>7} {'a/(2-a)':>9} {'理論SD比':>9}"
        f" {'実測座標SD比':>12}"
        f" | {'座標EMA d平均':>14} {'d標準偏差':>11}"
        f" | {'d EMA d平均':>13} {'d標準偏差':>11}"
    )
    print("  " + "-" * 126)
    print(
        f"  {'生値':>6} {'—':>9} {'—':>7} {'—':>9} {'—':>9} {1.0:12.3f}"
        f" | {raw_mean:14.5f} {raw_std:11.5f}"
        f" | {raw_mean:13.5f} {raw_std:11.5f}"
    )

    rows: list[dict[str, Any]] = []
    for tau in DEFAULT_TAUS:
        alpha = 1.0 - exp(-delta / tau)
        variance_ratio = alpha / (2.0 - alpha)
        theoretical = sqrt(variance_ratio)
        tau_app = app_tau_for(alpha)

        smoothed = ema_array(
            series.normalized, series.times, tau_seconds=tau, max_gap_seconds=max_gap
        )
        scalar = ema_scalar(
            series.distances, series.times, tau_seconds=tau, max_gap_seconds=max_gap
        )

        burn, warmed = _burn_in(series.count, tau, delta)
        window = slice(burn, None)
        smoothed_rms, _ = landmark_spread(smoothed[window])
        measured_ratio = float(np.median(smoothed_rms[group] / np.maximum(raw_reference, 1e-12)))

        coordinate_d = np.asarray(
            [value for value in (distance_of(frame) for frame in smoothed[window]) if value],
            dtype=np.float64,
        )
        scalar_d = scalar[window]
        if coordinate_d.size == 0:
            continue

        marker = " " if warmed else "*"
        print(
            f" {marker}{tau:6.1f} {tau_app:8.1f}s {alpha:7.3f} {variance_ratio:9.3f}"
            f" {theoretical:9.3f} {measured_ratio:12.3f}"
            f" | {coordinate_d.mean():14.5f} {coordinate_d.std(ddof=1):11.5f}"
            f" | {scalar_d.mean():13.5f} {scalar_d.std(ddof=1):11.5f}"
        )
        rows.append(
            {
                "tau_seconds": tau,
                "tau_app_seconds": tau_app,
                "app_delta_seconds": APP_DELTA_SECONDS,
                "alpha": alpha,
                "variance_ratio_theory": variance_ratio,
                "std_ratio_theory": theoretical,
                "coordinate_std_ratio_measured": measured_ratio,
                "frames_used": int(coordinate_d.size),
                "warmed_up": warmed,
                "coordinate_ema_distance": spread_json(summarize(coordinate_d)),
                "scalar_ema_distance": spread_json(summarize(scalar_d)),
            }
        )

    print()
    print(f"  （tau ごとに立ち上がり分を除外。除外は最大 {series.count // 3} 枚）")
    if any(not row["warmed_up"] for row in rows):
        print("  * その tau に対してフレームが足りず、EMA が定常に達していない行。")
        print("    フレーム数を増やすか、その行を読まないこと。")
    print("  読み方: 座標 EMA の列だけ d の *平均* が下がれば、docs level3 §3 の主張")
    print("          （座標を平滑化しないと効果がない）が実写でも成り立っている。")
    print("          d を EMA した列は、標準偏差だけ下がって平均が動かないはずである。")
    print(
        f"          tau_app は、解析 {APP_DELTA_SECONDS} 秒間隔のアプリで同じ alpha を得る時定数。"
    )

    report["smoothing"] = {
        "delta_seconds": delta,
        "app_delta_seconds": APP_DELTA_SECONDS,
        "max_gap_seconds": max_gap,
        "raw_distance": spread_json(summarize(series.distances)),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# テストD — 3D object symmetry
# --------------------------------------------------------------------------
def _correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float | None:
    """相関係数。定数列や 3 点未満では求めない。"""
    if len(left) < 3 or float(left.std()) <= 0.0 or float(right.std()) <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _flat_scale_distances(series: Series, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """§8.3 と同じ 2D 投影スケールで正規化したときの d_3d（比較用）。

    こちらは yaw で 1/cos(yaw) の倍率が残る。本文の値と並べることで、スケールの
    取り方が結論を左右していないことを確かめられる。
    """
    values: list[float] = []
    for frame3d, keep in zip(series.normalized3d, mask, strict=True):
        if not keep:
            continue
        # normalized3d は 3D 両眼間距離で割ってある。2D 投影スケールへ戻すには
        # 「3D 距離 ÷ 2D 投影距離」を掛け直せばよい。
        left = frame3d[list(CONFIG.left_eye_corners)].mean(axis=0)
        right = frame3d[list(CONFIG.right_eye_corners)].mean(axis=0)
        projected = float(np.linalg.norm((right - left)[:2]))
        value = distance_3d_of(frame3d)
        if value is None or projected <= 0.0:
            continue
        values.append(value / projected)
    return np.asarray(values, dtype=np.float64)


def section_d(capture: Capture, series: Series, report: dict[str, Any]) -> None:
    heading("D", "3D object symmetry — 姿勢を原理的に消せるか（対策11 の実写判定）")
    print(Z_SCALE_NOTE)

    mask = np.isfinite(series.distances3d)
    usable = int(mask.sum())
    if usable < 2:
        print(f"  z を取れたフレームが足りません（{usable} / {series.count}）。")
        report["object_symmetry_3d"] = {"frames": usable}
        return

    solid = series.distances3d[mask]
    flat = series.distances[mask]
    print(f"  z を取れたフレーム: {usable} / {series.count}")
    if usable < MIN_USEFUL_FRAMES:
        print(f"  ※ {MIN_USEFUL_FRAMES} 枚未満なので、以下の統計は目安にとどまる。")
    print()

    solid_std = float(solid.std(ddof=1))
    flat_std = float(flat.std(ddof=1))
    print("  1) d_3d と、現行 2D d")
    print(f"     {'':<10} {'平均':>22} {'標準偏差':>22} {'最小':>22} {'最大':>22}")
    print("     " + "-" * 102)
    for label, values in (("現行 2D d", flat), ("3D d_3d", solid)):
        print(
            f"     {label:<10} {both_units(float(values.mean())):>22} "
            f"{both_units(float(values.std(ddof=1))):>22} "
            f"{both_units(float(values.min())):>22} {both_units(float(values.max())):>22}"
        )
    print()
    print("     どちらも「両眼間距離を 1 とする残差 RMS」で、規約は同じである。")
    print("     2D はあてはめた正中線で折り、3D は最適な重ね合わせで折る点だけが違う。")
    print()

    print("  2) 信号 ÷ ノイズ床")
    print(f"     信号全幅            {both_units(SIGNAL_SPAN)}  ← 対称な成人 ↔ 非対称な成人")
    for label, deviation in (("現行 2D", flat_std), ("3D", solid_std)):
        if deviation > 0.0:
            print(
                f"     {label:<8} ノイズ床   {both_units(deviation)}   "
                f"信号 ÷ ノイズ床 {SIGNAL_SPAN / deviation:6.2f}x"
            )
    print()

    yaws = np.abs(yaw_values(series))[mask]
    pose_ok = np.isfinite(yaws)
    correlation_3d = _correlation(yaws[pose_ok], solid[pose_ok])
    correlation_2d = _correlation(yaws[pose_ok], flat[pose_ok])
    reference = _flat_scale_distances(series, mask)
    correlation_reference = (
        _correlation(yaws[pose_ok], reference[pose_ok]) if len(reference) == len(yaws) else None
    )

    print("  3) |yaw| との相関係数 ← 判定の核心")
    print(
        f"     現行 2D d            {'—' if correlation_2d is None else f'{correlation_2d:+.3f}'}"
    )
    print(
        f"     3D d_3d              {'—' if correlation_3d is None else f'{correlation_3d:+.3f}'}"
    )
    if correlation_reference is not None:
        print(
            f"     参考: 2D 投影スケール {correlation_reference:+.3f}"
            "   （z の正規化スケールを §8.3 の 2D 距離にした場合）"
        )
    print("     → 2D の相関が 3D で 0 付近まで落ちていれば、3D 化は姿勢問題を解いている。")
    print("       落ちていなければ、姿勢以外の何か（z の誤差など）が d_3d を動かしている。")
    print()

    print("  4) z 座標そのものの再現性（正規化空間、landmark ごと）")
    depths = series.normalized3d[mask][:, :, 2]
    depth_rms = np.sqrt(((depths - depths.mean(axis=0)) ** 2).mean(axis=0))
    plane_rms, _ = landmark_spread(series.normalized3d[mask][:, :, :2])
    available = depths.shape[1]
    groups: tuple[tuple[str, list[int]], ...] = (
        (f"計測に使う {len(MEASURED_INDICES)} 点", [i for i in MEASURED_INDICES if i < available]),
        (f"全 {available} 点", list(range(available))),
    )
    print(f"     {'対象':<18} {'z RMS 中央値':>20} {'z RMS p90':>20} {'xy RMS 中央値':>20}")
    print("     " + "-" * 80)
    depth_report: dict[str, Any] = {}
    for label, group in groups:
        if not group:
            continue
        print(
            f"     {label:<18} {both_units(float(np.median(depth_rms[group]))):>20} "
            f"{both_units(float(np.percentile(depth_rms[group], 90))):>20} "
            f"{both_units(float(np.median(plane_rms[group]))):>20}"
        )
        depth_report[label] = {
            "z_rms": spread_json(summarize(depth_rms[group])),
            "xy_rms": spread_json(summarize(plane_rms[group])),
        }
    measured_group = [i for i in MEASURED_INDICES if i < available]
    depth_median = float(np.median(depth_rms[measured_group])) if measured_group else float("nan")
    print()
    print(
        f"     判定閾値（docs level3 §4）: z の誤差が {Z_ERROR_BUDGET:.0%} IOD を超えると、"
        "3D 化は割に合わない。"
    )
    print("     ※ ここも再現性であって正解に対する誤差ではない。被写体の微動と姿勢の")
    print("       ドリフトを含むので、真の z 誤差の下限にしかならない。")
    print()

    print("  判定")
    verdict: list[str] = []
    if correlation_2d is not None and correlation_3d is not None:
        if abs(correlation_3d) < 0.5 * abs(correlation_2d):
            verdict.append(f"姿勢依存は落ちた（{correlation_2d:+.3f} → {correlation_3d:+.3f}）")
        else:
            verdict.append(f"姿勢依存が残っている（{correlation_2d:+.3f} → {correlation_3d:+.3f}）")
    if solid_std < flat_std:
        verdict.append(f"ノイズ床は下がった（{flat_std:.5f} → {solid_std:.5f}）")
    else:
        verdict.append(f"ノイズ床は下がらなかった（{flat_std:.5f} → {solid_std:.5f}）")
    if isfinite(depth_median):
        if depth_median > Z_ERROR_BUDGET:
            verdict.append(f"z の再現性が閾値超え（{depth_median:.4f} > {Z_ERROR_BUDGET:.2f} IOD）")
        else:
            verdict.append(f"z の再現性は閾値内（{depth_median:.4f} ≦ {Z_ERROR_BUDGET:.2f} IOD）")
    for line in verdict:
        print(f"     - {line}")

    report["object_symmetry_3d"] = {
        "frames": usable,
        "z_scale": "3d-interocular-distance",
        "distance_3d": spread_json(summarize(solid)),
        "distance_2d_same_frames": spread_json(summarize(flat)),
        "noise_floor_3d": {"iod": solid_std, "mm": solid_std * IPD_MM},
        "noise_floor_2d": {"iod": flat_std, "mm": flat_std * IPD_MM},
        "signal_to_noise_3d": SIGNAL_SPAN / solid_std if solid_std > 0.0 else None,
        "signal_to_noise_2d": SIGNAL_SPAN / flat_std if flat_std > 0.0 else None,
        "abs_yaw_correlation_3d": correlation_3d,
        "abs_yaw_correlation_2d": correlation_2d,
        "abs_yaw_correlation_3d_flat_scale": correlation_reference,
        "depth_reproducibility": depth_report,
        "z_error_budget_iod": Z_ERROR_BUDGET,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# テストE — 既知量注入でゲインを測る（docs §7 のテストC）
# --------------------------------------------------------------------------
GAIN_NOTE = """
  何を測るか

    1. yaw が中央値に近いフレームを 1 枚選び、検出して d_2d(0) と d_3d(0) を得る
    2. mouth ペアの片側だけを、正規化フレームの横方向へ既知量 δ だけずらす
    3. その変位を landmark へ *直接* 適用した d を求める（検出を通さない真値）
    4. 同じ変位を `domain/warp_mesh` + `adapters/imaging/piecewise_warp` で画像へ
       適用し、変形画像を再検出して d を求める（検出を通した値）

        ゲイン = (検出を通した増分) / (landmark へ直接適用した増分)

    ゲイン ≈ 1.0  パイプラインは注入した非対称をそのまま検出できている
    ゲイン ≪ 1.0  テンプレートが非対称を抑制している（docs §4.3 の実証）

  読むときの前提

    区分アフィン変形は、メッシュの頂点をその変形先へ *厳密に* 写す。したがって
    注入した landmark の位置にある画素は、意図どおり δ だけ動く。再検出した
    landmark がそれより小さくしか動かないなら、それは変形の不足ではなく検出器の
    応答である。ただし三角形の *内側* は線形補間なので、周囲のテクスチャは実際の
    非対称とは異なる歪み方をする。ここが本テストの原理的な限界である。

    それが成り立つのは、隣り合う点の相対変位がその間隔より小さいうちだけである。
    超えると三角形が裏返り、その領域は「δ ずれた顔」ではなく壊れた画素になる。
    変位場ごとに、裏返り始める δ を表の上に出してある。

    δ = 0 の行は恒等変形の対照である。非対称を入れずに変形と再標本化だけを通した
    ときの d の動きを示す。この対照より小さい変化しか出ない δ は読めない。

    ゲインは δ ごとに求めると暴れる。d は距離なので、注入した非対称が被写体自身の
    非対称を打ち消す向きだと、検出側の d は小さい δ でいったん下がるからである
    （真値側は打ち消しの後で必ず増えるので、比が負になる）。判定には δ をまとめて
    1 本の直線にあてはめた「回帰の傾き」を使う。
"""


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """変形画像を再検出した結果。成立しなかったときは `reason` に理由が入る。"""

    points: NDArray[np.float64] | None
    points3d: NDArray[np.float64] | None
    reason: str | None


#: (変形元の画素座標, 変形先の画素座標, 変形先の画素 3D) → 再検出の結果。
#: 実写では画像を変形して再検出する。self-test では既知の応答を返す実装を差し込み、
#: MediaPipe も画像も無しでゲインの算術だけを検証する。
InjectionProbe = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None],
    ProbeResult,
]


@dataclass(frozen=True, slots=True)
class InjectionOutcome:
    """δ を 1 つ注入したときの、真値経路と検出経路の測定結果。距離の単位は IOD。"""

    delta_mm: float
    truth_2d: float | None
    truth_3d: float | None
    detected_2d: float | None
    detected_3d: float | None
    gain_2d: float | None
    gain_3d: float | None
    #: 注入した点の、実際の横変位（中央値、mm）。意図した δ と比べる。
    realized_mm: float
    #: 直交方向（正規化 y）へ漏れた分の中央値（mm）。0 に近いはず。
    realized_y_mm: float
    #: 注入しなかった計測点が付随して動いた距離の中央値（mm）。
    leak_mm: float
    #: 方式② の真値と検出値、およびそのゲイン。セクション I と J だけが読む。
    truth_2dp: float | None = None
    detected_2dp: float | None = None
    gain_2dp: float | None = None
    #: 注入した点の横変位を、*基準フレームの正規化変換* で測ったもの（中央値、mm）。
    #: `realized_mm` は変形後の点から正規化をやり直して測るので、眼アンカー自身が
    #: 動く部位（目）では正規化フレームごと動いてしまい、応答が読めなくなる。
    #: 変形は画素空間で起きるので、基準フレームを固定して測るほうが「その点が
    #: どれだけ動いたか」に一致する。セクション J はこちらを使う。
    realized_fixed_mm: float = float("nan")
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class InjectionReport:
    """1 つの変位場ぶんの注入測定。複数フレームをまとめた中央値の場合もある。"""

    field_name: str
    sigma_iod: float
    baseline_2d: float
    baseline_3d: float | None
    injected_count: int
    #: 変位場の影響が及ぶ点の数（重み 0.05 以上）。
    affected_count: int
    #: この変位場が三角形を裏返し始める δ（mm）。
    fold_mm: float
    outcomes: tuple[InjectionOutcome, ...]
    #: まとめたフレーム数。
    frames: int = 1
    #: 方式② の基準値。セクション I と J だけが読む。
    baseline_2dp: float | None = None
    #: 変位場が §8.3 の眼アンカー 4 点へ与える重みの最大値。0 より大きいと、
    #: 注入が正規化フレームそのものを動かしている（部位「目」で起きる）。
    anchor_weight: float = 0.0
    #: 注入した部位の名前（`MetricConfig.parts` の key）。
    part: str = "mouth"


def _gain(baseline: float, truth: float | None, detected: float | None) -> float | None:
    """増分の比。真値側の増分がほぼ 0 の行（δ=0 の対照）では求めない。"""
    if truth is None or detected is None:
        return None
    denominator = truth - baseline
    if abs(denominator) < 1e-9:
        return None
    return (detected - baseline) / denominator


def measure_injection(
    *,
    baseline_points: NDArray[np.float64],
    baseline_points3d: NDArray[np.float64] | None,
    deltas_mm: tuple[float, ...],
    probe: InjectionProbe,
    field_name: str = "点のみ",
    sigma_iod: float = 0.0,
    part: str = "mouth",
) -> InjectionReport | None:
    """既知量注入の測定本体。検出の実体は `probe` に委ねる。

    座標系は入力の画素座標のままである。注入は §8.3 の正規化フレームで行い、
    同じ変換で画素へ戻すので、δ は顔に固定した「横」方向の長さになる。
    """
    transform_2d = None
    try:
        transform_2d = normalization_transform(
            baseline_points,
            left_eye_corners=CONFIG.left_eye_corners,
            right_eye_corners=CONFIG.right_eye_corners,
        )
    except GeometryError:
        return None
    normalized = transform_2d.apply(baseline_points)
    baseline_2d = distance_of(baseline_points)
    if baseline_2d is None:
        return None

    normalized3d = normalized_3d_of(baseline_points3d) if baseline_points3d is not None else None
    baseline_3d = distance_3d_of(normalized3d) if normalized3d is not None else None
    baseline_2dp = distance_2d_procrustes_of(normalized)

    indices = part_injection_indices(normalized, part)
    weights = injection_weights(normalized, indices, sigma_iod=sigma_iod)
    others = [index for index in MEASURED_INDICES if index not in indices]
    anchors = list(CONFIG.left_eye_corners + CONFIG.right_eye_corners)

    outcomes: list[InjectionOutcome] = []
    for delta_mm in deltas_mm:
        delta_iod = delta_mm / IPD_MM
        target_points = transform_2d.invert(inject_field(normalized, weights, delta_iod))
        truth_2d = distance_of(target_points)
        truth_3d = (
            distance_3d_of(inject_field(normalized3d, weights, delta_iod))
            if normalized3d is not None
            else None
        )
        # 方式② の真値は方式① と同じく、変形後の画素座標から正規化をやり直して
        # 求める。目のように眼アンカーごと動く部位でも、真値と検出値が同じ規約に
        # なる。
        target_normal = normalized_of(target_points)
        truth_2dp = None if target_normal is None else distance_2d_procrustes_of(target_normal)
        if truth_2d is None:
            continue
        # 注入は面内の横ずらしなので、画素 3D の z 列はそのままでよい。
        target_points3d = (
            np.column_stack([target_points, baseline_points3d[:, 2]])
            if baseline_points3d is not None
            else None
        )

        result = probe(baseline_points, target_points, target_points3d)
        if result.points is None:
            outcomes.append(
                InjectionOutcome(
                    delta_mm=delta_mm,
                    truth_2d=truth_2d,
                    truth_3d=truth_3d,
                    detected_2d=None,
                    detected_3d=None,
                    gain_2d=None,
                    gain_3d=None,
                    realized_mm=float("nan"),
                    realized_y_mm=float("nan"),
                    leak_mm=float("nan"),
                    realized_fixed_mm=float("nan"),
                    reason=result.reason,
                )
            )
            continue

        detected_normal = normalized_of(result.points)
        detected_normal3d = (
            normalized_3d_of(result.points3d) if result.points3d is not None else None
        )
        detected_3d = distance_3d_of(detected_normal3d) if detected_normal3d is not None else None
        detected_2dp = (
            None if detected_normal is None else distance_2d_procrustes_of(detected_normal)
        )
        if detected_normal is None:
            realized = realized_y = leak = float("nan")
        else:
            moved = detected_normal[list(indices)] - normalized[list(indices)]
            realized = float(np.median(moved[:, 0])) * IPD_MM
            realized_y = float(np.median(np.abs(moved[:, 1]))) * IPD_MM
            drift = np.linalg.norm(detected_normal[others] - normalized[others], axis=1)
            leak = float(np.median(drift)) * IPD_MM
        # 基準フレームを固定したままの応答。正規化のやり直しを挟まない。
        fixed = transform_2d.apply(result.points)[list(indices)] - normalized[list(indices)]
        realized_fixed = float(np.median(fixed[:, 0])) * IPD_MM
        detected_2d = distance_of(result.points)
        outcomes.append(
            InjectionOutcome(
                delta_mm=delta_mm,
                truth_2d=truth_2d,
                truth_3d=truth_3d,
                detected_2d=detected_2d,
                detected_3d=detected_3d,
                gain_2d=_gain(baseline_2d, truth_2d, detected_2d),
                gain_3d=(
                    None if baseline_3d is None else _gain(baseline_3d, truth_3d, detected_3d)
                ),
                realized_mm=realized,
                realized_y_mm=realized_y,
                leak_mm=leak,
                truth_2dp=truth_2dp,
                detected_2dp=detected_2dp,
                gain_2dp=(
                    None if baseline_2dp is None else _gain(baseline_2dp, truth_2dp, detected_2dp)
                ),
                realized_fixed_mm=realized_fixed,
                reason=None,
            )
        )

    return InjectionReport(
        field_name=field_name,
        sigma_iod=sigma_iod,
        baseline_2d=baseline_2d,
        baseline_3d=baseline_3d,
        injected_count=len(indices),
        affected_count=int((weights >= 0.05).sum()),
        fold_mm=fold_threshold_mm(normalized, weights),
        outcomes=tuple(outcomes),
        baseline_2dp=baseline_2dp,
        anchor_weight=float(weights[anchors].max()),
        part=part,
    )


def _median_or_none(values: list[float | None]) -> float | None:
    """None と NaN を除いた中央値。全部欠けていれば None。"""
    usable = [value for value in values if value is not None and isfinite(value)]
    return float(np.median(usable)) if usable else None


def _median_float(values: list[float]) -> float:
    usable = [value for value in values if isfinite(value)]
    return float(np.median(usable)) if usable else float("nan")


def aggregate_injection(reports: list[InjectionReport]) -> InjectionReport:
    """同じ変位場の複数フレーム測定を、δ ごとの中央値へまとめる。

    ゲインはフレームごとに求めてから中央値を取る。まとめた d から計算し直すと、
    基準値の個人差（フレームごとの d(0)）が分子と分母で食い違う。
    """
    first = reports[0]
    if len(reports) == 1:
        return first

    outcomes: list[InjectionOutcome] = []
    for position, template in enumerate(first.outcomes):
        same = [report.outcomes[position] for report in reports if position < len(report.outcomes)]
        reasons = [item.reason for item in same if item.reason is not None]
        outcomes.append(
            InjectionOutcome(
                delta_mm=template.delta_mm,
                truth_2d=_median_or_none([item.truth_2d for item in same]),
                truth_3d=_median_or_none([item.truth_3d for item in same]),
                detected_2d=_median_or_none([item.detected_2d for item in same]),
                detected_3d=_median_or_none([item.detected_3d for item in same]),
                gain_2d=_median_or_none([item.gain_2d for item in same]),
                gain_3d=_median_or_none([item.gain_3d for item in same]),
                realized_mm=_median_float([item.realized_mm for item in same]),
                realized_y_mm=_median_float([item.realized_y_mm for item in same]),
                leak_mm=_median_float([item.leak_mm for item in same]),
                truth_2dp=_median_or_none([item.truth_2dp for item in same]),
                detected_2dp=_median_or_none([item.detected_2dp for item in same]),
                gain_2dp=_median_or_none([item.gain_2dp for item in same]),
                realized_fixed_mm=_median_float([item.realized_fixed_mm for item in same]),
                reason=(
                    None if not reasons else f"{len(reasons)}/{len(same)} 枚で失敗: {reasons[0]}"
                ),
            )
        )

    return InjectionReport(
        field_name=first.field_name,
        sigma_iod=first.sigma_iod,
        baseline_2d=_median_float([report.baseline_2d for report in reports]),
        baseline_3d=_median_or_none([report.baseline_3d for report in reports]),
        injected_count=first.injected_count,
        affected_count=round(_median_float([float(item.affected_count) for item in reports])),
        fold_mm=_median_float([report.fold_mm for report in reports]),
        outcomes=tuple(outcomes),
        frames=len(reports),
        baseline_2dp=_median_or_none([report.baseline_2dp for report in reports]),
        anchor_weight=_median_float([report.anchor_weight for report in reports]),
        part=first.part,
    )


def frame_at(arguments: argparse.Namespace, position: int) -> NDArray[np.uint8] | None:
    """入力の position 番目のフレームだけを、もう一度読み出す。

    検出時に画像を保持しないので（不変条件 1）、テストE が必要とする 1 枚だけを
    読み直す。読んだ画像も変形した画像もメモリ上にしか置かない。
    """
    frames = (
        iter_video_frames(arguments.video, position + 1)
        if arguments.video is not None
        else iter_directory_frames(arguments.frames, position + 1)
    )
    for index, candidate in enumerate(frames):
        if index == position:
            return candidate
    return None


def _median_yaw_detections(capture: Capture, count: int) -> list[Detection]:
    """yaw が中央値に近い順に count 枚。姿勢の影響を最小にしたフレームを選ぶ。"""
    posed = [
        item for item in capture.detections if item.pose is not None and item.points3d is not None
    ]
    if not posed:
        posed = [item for item in capture.detections if item.pose is not None]
    if not posed:
        return []
    yaws = np.asarray([item.pose.yaw_deg for item in posed if item.pose is not None])
    median = float(np.median(yaws))
    ordered = sorted(posed, key=lambda item: abs(item.pose.yaw_deg - median) if item.pose else 1e9)
    return ordered[: max(1, count)]


@dataclass(frozen=True, slots=True)
class InjectionRequest:
    """1 枚に対して行う注入測定 1 件の指定。"""

    #: 表と JSON に出る変位場の名前。
    name: str
    #: 変位場の広がり（両眼間距離比）。0 なら指定点だけを動かす。
    sigma_iod: float
    #: 注入する部位（`MetricConfig.parts` の key）。
    part: str


#: 変位場の名前と広がり。「点のみ」は docs テストC の記述どおりの場、「領域」は
#: 区分アフィンが折り返さないよう近傍も一緒に動かした場。
INJECTION_FIELDS: tuple[tuple[str, float], ...] = (("点のみ", 0.0), ("領域", REGION_SIGMA_IOD))

#: テストE が使う指定。docs テストC のとおり、注入先は口だけである。
MOUTH_REQUESTS: tuple[InjectionRequest, ...] = tuple(
    InjectionRequest(name=name, sigma_iod=sigma, part="mouth") for name, sigma in INJECTION_FIELDS
)

#: セクション J が使う指定。テストE の領域変位場を、部位だけ変えて当てる。
PART_REQUESTS: tuple[InjectionRequest, ...] = tuple(
    InjectionRequest(name=label, sigma_iod=REGION_SIGMA_IOD, part=part)
    for part, label in PART_LABELS.items()
)


def section_e(
    arguments: argparse.Namespace,
    capture: Capture,
    report: dict[str, Any],
) -> None:
    heading("E", "既知量注入 — パイプラインのゲイン（docs §7 のテストC）")
    print(GAIN_NOTE)

    if arguments.self_test:
        print("  合成 capture には元画像が無いので実写の測定は行わない。")
        print("  ゲイン算術そのものの検証は [S] の「ゲイン」項目で行っている。")
        report["injection_gain"] = {"frames": 0, "reason": "self-test"}
        return

    chosen = _median_yaw_detections(capture, arguments.gain_frames)
    if not chosen:
        print("  姿勢を取得できたフレームがありません。")
        report["injection_gain"] = {"frames": 0, "reason": "no-pose"}
        return

    print(
        f"  変位ガード: 両眼間隔の {MAX_WARP_DISPLACEMENT_RATIO:.0%}"
        f"（IPD {IPD_MM:.0f}mm 換算で {MAX_WARP_DISPLACEMENT_RATIO * IPD_MM:.1f}mm）まで"
    )
    model_path, model_version = resolve_model(arguments.model)
    collected: dict[str, list[InjectionReport]] = {name: [] for name, _ in INJECTION_FIELDS}
    used: list[dict[str, float]] = []

    for detection in chosen:
        image = frame_at(arguments, detection.index)
        if image is None:
            print(f"  フレーム {detection.index} を読み直せませんでした。飛ばします。")
            continue
        yaw = detection.pose.yaw_deg if detection.pose is not None else float("nan")
        measured = _inject_one_frame(
            image,
            model_path=model_path,
            model_version=model_version,
            mode=arguments.mode,
        )
        if measured is None:
            print(f"  フレーム {detection.index} は基準を作れませんでした。飛ばします。")
            continue
        used.append({"frame_index": float(detection.index), "yaw_deg": yaw})
        for name, item in measured.items():
            collected[name].append(item)

    if not used:
        print("  測定できたフレームがありません。")
        report["injection_gain"] = {"frames": 0, "reason": "no-usable-frame"}
        return

    print(
        f"  対象フレーム: {len(used)} 枚（yaw 中央値の近傍から選択）"
        f" / yaw {min(item['yaw_deg'] for item in used):+.2f}°"
        f" 〜 {max(item['yaw_deg'] for item in used):+.2f}°"
    )
    if len(used) > 1:
        print("  表の各欄はフレーム間の中央値。ゲインはフレームごとに求めてから中央値を取る。")

    fields = [
        _print_injection(aggregate_injection(collected[name]))
        for name, _ in INJECTION_FIELDS
        if collected[name]
    ]
    report["injection_gain"] = {
        "frames": len(used),
        "frame_indices": [int(item["frame_index"]) for item in used],
        "yaw_deg": [item["yaw_deg"] for item in used],
        "measured_landmarks": len(MEASURED_INDICES),
        "displacement_guard_mm": MAX_WARP_DISPLACEMENT_RATIO * IPD_MM,
        "fields": fields,
        "effective_resolution": _print_effective_resolution(fields, report),
    }


def _print_effective_resolution(
    fields: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any] | None:
    """ゲインをノイズ床に織り込んだ、実効の 信号 ÷ ノイズ床。

    テストB・D の「信号 ÷ ノイズ床」は、文献の信号全幅がそのまま d に現れる
    ことを前提にしている。テストE がその前提を測るので、両者を掛け合わせて
    初めて「この指標は人と人を区別できるか」に答えられる。

        実効信号 = 信号全幅 × ゲイン
        実効 SNR = 実効信号 ÷ ノイズ床

    テストD を実行していないときは何も出さない（ノイズ床が手元に無い）。
    """
    solid = report.get("object_symmetry_3d")
    if not isinstance(solid, dict) or "noise_floor_2d" not in solid:
        return None
    usable = [item for item in fields if item.get("slope_3d") or item.get("slope_2d")]
    if not usable:
        return None
    chosen = usable[-1]

    print()
    print("  ゲインを織り込んだ 信号 ÷ ノイズ床（テストD のノイズ床と組み合わせる）")
    print(f"     変位場「{chosen['field']}」のゲインを使用")
    print(
        f"     {'':<8} {'ゲイン':>8} {'実効信号':>12} {'ノイズ床':>12} "
        f"{'公称SNR':>9} {'実効SNR':>9}"
    )
    print("     " + "-" * 66)
    rows: dict[str, Any] = {"field": chosen["field"]}
    for label, slope_key, floor_key in (
        ("2D", "slope_2d", "noise_floor_2d"),
        ("3D", "slope_3d", "noise_floor_3d"),
    ):
        slope = chosen.get(slope_key)
        floor = solid[floor_key]["mm"]
        if slope is None or floor <= 0.0:
            continue
        signal = SIGNAL_SPAN * IPD_MM
        effective = signal * slope
        print(
            f"     {label:<8} {slope:8.3f} {effective:10.3f}mm {floor:10.3f}mm "
            f"{signal / floor:8.2f}x {effective / floor:8.2f}x"
        )
        rows[label] = {
            "gain": slope,
            "effective_signal_mm": effective,
            "noise_floor_mm": floor,
            "nominal_snr": signal / floor,
            "effective_snr": effective / floor,
        }
    print()
    print("     実効 SNR が 1.0 を下回るなら、ゲインを織り込んだ後では人と人の差を")
    print("     1 フレームでは区別できない。姿勢層別（テストH）の床を使えば改善する。")
    return rows


def _inject_one_frame(
    image: NDArray[np.uint8],
    *,
    model_path: Path,
    model_version: str,
    mode: str,
    requests: tuple[InjectionRequest, ...] = MOUTH_REQUESTS,
) -> dict[str, InjectionReport] | None:
    """1 枚に対して、指定されたすべての変位場で注入測定を行う。

    detector はフレームごとに作り直す。変形画像を次々と同じ detector へ流すと、
    VIDEO モードでは前のフレームの影響が入りうるためである（本モデルは状態を
    持たないが、それに依存しないでおく）。
    """
    height, width = image.shape[:2]
    stream = LandmarkStream(model_path=model_path, model_version=model_version, mode=mode)
    warper = OpenCvPiecewiseWarper()
    counter = {"step": 0}
    try:
        baseline_points = _frame_points(stream.detect(image, frame_id=0, timestamp_ms=0))
        if baseline_points is None:
            return None
        baseline_points3d = to_pixel_3d(stream.last_points_3d, width=width, height=height)

        def probe(
            source: NDArray[np.float64],
            target: NDArray[np.float64],
            _target3d: NDArray[np.float64] | None,
        ) -> ProbeResult:
            try:
                mesh = build_warp_mesh(
                    source,
                    target,
                    width=width,
                    height=height,
                    interocular_px=float(
                        np.linalg.norm(
                            source[list(CONFIG.right_eye_corners)].mean(axis=0)
                            - source[list(CONFIG.left_eye_corners)].mean(axis=0)
                        )
                    ),
                )
            except WarpMeshError as error:
                return ProbeResult(None, None, f"変位ガードが拒否: {error}")
            try:
                warped = warper.warp(image, mesh)
            except PiecewiseWarpError as error:
                return ProbeResult(None, None, f"変形に失敗: {error}")
            counter["step"] += 1
            step = counter["step"]
            detected = _frame_points(stream.detect(warped, frame_id=step, timestamp_ms=step * 100))
            if detected is None:
                return ProbeResult(None, None, "変形画像から単一顔を検出できない")
            return ProbeResult(
                detected,
                to_pixel_3d(stream.last_points_3d, width=width, height=height),
                None,
            )

        results: dict[str, InjectionReport] = {}
        for request in requests:
            measured = measure_injection(
                baseline_points=baseline_points,
                baseline_points3d=baseline_points3d,
                deltas_mm=INJECTION_DELTAS_MM,
                probe=probe,
                field_name=request.name,
                sigma_iod=request.sigma_iod,
                part=request.part,
            )
            if measured is None:
                return None
            results[request.name] = measured
        return results
    finally:
        stream.close()


def _print_injection(measured: InjectionReport) -> dict[str, Any]:
    """テストE の表と判定を、変位場 1 つぶん出す。同じ内容を JSON 用に返す。"""
    print()
    print("  " + "=" * 96)
    print(f"  変位場: {measured.field_name}", end="")
    if measured.sigma_iod > 0.0:
        print(
            f"（Gauss、sigma = {measured.sigma_iod:.2f} IOD = {measured.sigma_iod * IPD_MM:.1f}mm）"
        )
    else:
        print("（docs テストC の記述どおり、mouth ペアの片側だけを動かす）")
    print("  " + "=" * 96)
    print(
        f"  注入する点: mouth ペアの片側 {measured.injected_count} 点"
        f" / 変位が及ぶ点（重み 0.05 以上）{measured.affected_count} 点"
        f" / 計測に使う {len(MEASURED_INDICES)} 点中"
    )
    print(
        f"  三角形が裏返り始める δ: {measured.fold_mm:.2f}mm"
        "   ← これを超えた行の変形画像は「δ ずれた顔」ではない"
    )
    print()
    print(f"  基準 d_2d = {measured.baseline_2d * IPD_MM:.3f}mm", end="")
    if measured.baseline_3d is not None:
        print(f" / 基準 d_3d = {measured.baseline_3d * IPD_MM:.3f}mm")
    else:
        print(" / 基準 d_3d = —（z を取れなかった）")
    print()
    print(
        f"  {'δ':>5} {'真値d_2d':>9} {'検出d_2d':>9} {'ゲイン2D':>9}"
        f" |{'真値d_3d':>9} {'検出d_3d':>9} {'ゲイン3D':>9}"
        f" |{'実変位':>8} {'δとの差':>8} {'実変位比':>9} {'y漏れ':>7} {'他点漏れ':>8}"
    )
    print("  " + "-" * 122)

    rows: list[dict[str, Any]] = []
    for outcome in measured.outcomes:
        ratio = outcome.realized_mm / outcome.delta_mm if outcome.delta_mm > 0.0 else float("nan")
        rows.append(
            {
                "delta_mm": outcome.delta_mm,
                "truth_2d_mm": _scaled(outcome.truth_2d),
                "truth_3d_mm": _scaled(outcome.truth_3d),
                "detected_2d_mm": _scaled(outcome.detected_2d),
                "detected_3d_mm": _scaled(outcome.detected_3d),
                "gain_2d": outcome.gain_2d,
                "gain_3d": outcome.gain_3d,
                "realized_mm": outcome.realized_mm,
                "realized_error_mm": outcome.realized_mm - outcome.delta_mm,
                "realized_ratio": ratio if isfinite(ratio) else None,
                "realized_orthogonal_mm": outcome.realized_y_mm,
                "leak_mm": outcome.leak_mm,
                "reason": outcome.reason,
            }
        )
        if outcome.reason is not None:
            print(f"  {outcome.delta_mm:5.1f} {outcome.reason}")
            continue
        print(
            f"  {outcome.delta_mm:5.1f} {_mm_or_dash(outcome.truth_2d):>9}"
            f" {_mm_or_dash(outcome.detected_2d):>9} {_ratio_or_dash(outcome.gain_2d):>9}"
            f" |{_mm_or_dash(outcome.truth_3d):>9} {_mm_or_dash(outcome.detected_3d):>9}"
            f" {_ratio_or_dash(outcome.gain_3d):>9}"
            f" |{outcome.realized_mm:8.3f} {outcome.realized_mm - outcome.delta_mm:+8.3f}"
            f" {_ratio_or_dash(ratio if isfinite(ratio) else None):>9}"
            f" {outcome.realized_y_mm:7.3f} {outcome.leak_mm:8.3f}"
        )
    print()
    print("  d は mm、δ・実変位・漏れも mm。ゲインと実変位比は無次元。")
    print("  実変位 = 変形画像から再検出した注入点の、正規化フレーム横方向の移動量（中央値）。")
    print("  実変位比 = 実変位 ÷ δ。landmark の水準でどれだけ応答したか（ゲインの前段）。")
    print("  他点漏れ = 注入しなかった計測点が付随して動いた距離（中央値）。")
    print()

    verdict = _injection_verdict(measured)
    print("  判定")
    for line in verdict:
        print(f"     - {line}")

    return {
        "field": measured.field_name,
        "sigma_iod": measured.sigma_iod,
        "frames": measured.frames,
        "injected_landmarks": measured.injected_count,
        "affected_landmarks": measured.affected_count,
        "fold_threshold_mm": measured.fold_mm,
        "baseline_2d_mm": measured.baseline_2d * IPD_MM,
        "baseline_3d_mm": _scaled(measured.baseline_3d),
        "slope_2d": _injection_slope(measured, axis="2d"),
        "slope_3d": _injection_slope(measured, axis="3d"),
        "rows": rows,
        "verdict": verdict,
    }


def _scaled(value: float | None) -> float | None:
    """両眼間距離比を mm へ。None はそのまま通す。"""
    return None if value is None else value * IPD_MM


#: 軸名 → (基準値の読み出し, 真値の読み出し, 検出値の読み出し)。方式を 1 つ足す
#: たびに `_injection_slope` の分岐が増えるのを避けるため、表にしてある。
_AXIS_READERS: dict[
    str,
    tuple[
        Callable[[InjectionReport], float | None],
        Callable[[InjectionOutcome], float | None],
        Callable[[InjectionOutcome], float | None],
    ],
] = {
    "2d": (lambda item: item.baseline_2d, lambda row: row.truth_2d, lambda row: row.detected_2d),
    "3d": (lambda item: item.baseline_3d, lambda row: row.truth_3d, lambda row: row.detected_3d),
    "2dp": (
        lambda item: item.baseline_2dp,
        lambda row: row.truth_2dp,
        lambda row: row.detected_2dp,
    ),
}


def _injection_slope(measured: InjectionReport, *, axis: str) -> float | None:
    """増分どうしの回帰の傾き（原点を通す）。δ ごとのゲインより安定する。

        slope = Σ(Δ真値 · Δ検出) / Σ(Δ真値²)

    δ ごとのゲインは、d の基準値が大きいときに分母が小さくなって暴れる。傾きは
    すべての δ をまとめて 1 本の直線にあてはめるので、その暴れを平均する。
    """
    read_baseline, read_truth, read_detected = _AXIS_READERS[axis]
    baseline = read_baseline(measured)
    if baseline is None:
        return None
    numerator = 0.0
    denominator = 0.0
    for outcome in measured.outcomes:
        if outcome.reason is not None or outcome.delta_mm > measured.fold_mm:
            continue
        truth = read_truth(outcome)
        detected = read_detected(outcome)
        if truth is None or detected is None:
            continue
        numerator += (truth - baseline) * (detected - baseline)
        denominator += (truth - baseline) ** 2
    return numerator / denominator if denominator > 0.0 else None


def _injection_verdict(measured: InjectionReport) -> list[str]:
    """ゲインの読み。想定外に大きい値と、対照より小さい変化に印を付ける。"""
    verdict: list[str] = []
    control = next(
        (
            item
            for item in measured.outcomes
            if item.delta_mm == 0.0 and item.detected_2d is not None
        ),
        None,
    )
    if control is not None and control.detected_2d is not None:
        shift = abs(control.detected_2d - measured.baseline_2d) * IPD_MM
        verdict.append(f"恒等変形の対照: d_2d が {shift:.3f}mm 動く（変形と再標本化だけの分）")

    # 折り返しが起きる δ の行は「δ ずれた顔」ではないので、判定から外す。
    usable = [
        item
        for item in measured.outcomes
        if item.delta_mm > 0.0 and item.reason is None and item.delta_mm <= measured.fold_mm
    ]
    folded = [item for item in measured.outcomes if item.delta_mm > measured.fold_mm]
    if folded:
        verdict.append(
            "折り返しのため判定から除外した δ: "
            + ", ".join(f"{item.delta_mm:.0f}mm" for item in folded)
            + f"（閾値 {measured.fold_mm:.2f}mm）"
        )
    if not usable:
        verdict.append("折り返さずに注入できた δ が 1 つも無い。この変位場ではゲインを測れない。")
        return verdict

    realized = [item.realized_mm / item.delta_mm for item in usable if isfinite(item.realized_mm)]
    if realized:
        verdict.append(
            f"landmark 水準の応答（実変位 ÷ δ）中央値 {float(np.median(realized)):.3f}"
            f"（{min(realized):.3f}〜{max(realized):.3f}）"
        )

    per_axis: tuple[tuple[str, list[float | None], float | None], ...] = (
        ("2D", [item.gain_2d for item in usable], _injection_slope(measured, axis="2d")),
        ("3D", [item.gain_3d for item in usable], _injection_slope(measured, axis="3d")),
    )
    for label, gains, slope in per_axis:
        values = [value for value in gains if value is not None]
        if not values:
            verdict.append(f"{label}: ゲインを求められた δ がない")
            continue
        median = float(np.median(values))
        verdict.append(
            f"{label} ゲイン: 回帰の傾き {'—' if slope is None else f'{slope:.3f}'}"
            f"（δ={', '.join(f'{item.delta_mm:.0f}mm' for item in usable)} ごとの値は "
            f"{min(values):.3f}〜{max(values):.3f}、中央値 {median:.3f}）"
        )
        headline = median if slope is None else slope
        if headline > GAIN_UPPER_ALARM:
            verdict.append(f"{label}: ゲインが {GAIN_UPPER_ALARM:.2f} を超えている。想定外。")

    rejected = [item for item in measured.outcomes if item.reason is not None]
    if rejected:
        verdict.append(
            "測れなかった δ: " + ", ".join(f"{item.delta_mm:.0f}mm" for item in rejected)
        )
    return verdict


def _mm_or_dash(value: float | None) -> str:
    return "—" if value is None else f"{value * IPD_MM:.3f}"


def _ratio_or_dash(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


# --------------------------------------------------------------------------
# テストF — 3D の鏡像テスト
# --------------------------------------------------------------------------
MIRROR_3D_NOTE = """
  テストA を d_3d に対して行う。d_3d でも、幾何学的に正しい検出器なら
  d_3d(I) と d_3d(mirror(I)) は厳密に一致する。

    鏡像を完全に検出できたなら X_mirror[i] = flip_x(X[sigma(i)]) である。
    object symmetry が内部で作る鏡映は M(Y)[i] = flip_x(Y[sigma(i)]) なので、
    M(X_mirror) = X が成り立つ。重ね合わせ後の残差 RMS は 2 つの配置を入れ替えて
    も同じなので、d_3d(mirror(I)) = d_3d(I) となる。差は検出器の左右バイアス
    だけである。

  2D と違うのは、Procrustes が剛体変換を吸収する点である。左右バイアスのうち
  「顔全体が一様にずれる・傾く」成分は 3D では消える。消えずに残るのは、剛体
  変換で説明できない形の左右非対称である。どれだけ消えるかがここで分かる。
"""


def section_f(capture: Capture, report: dict[str, Any]) -> None:
    heading("F", "3D 鏡像テスト — d_3d に残る検出器の左右バイアス")
    print(MIRROR_3D_NOTE)

    indices = list(SIGMA_INDICES)
    partners = [MIRROR_PERMUTATION[index] for index in indices]
    flip = np.array([-1.0, 1.0, 1.0], dtype=np.float64)

    solid_original: list[float] = []
    solid_mirror: list[float] = []
    flat_original: list[float] = []
    flat_mirror: list[float] = []
    bias_frames: list[NDArray[np.float64]] = []

    for item in capture.detections:
        if item.mirrored is None or item.points3d is None or item.mirrored3d is None:
            continue
        normal_original = normalized_3d_of(item.points3d)
        normal_mirror = normalized_3d_of(item.mirrored3d)
        if normal_original is None or normal_mirror is None:
            continue
        left = distance_3d_of(normal_original)
        right = distance_3d_of(normal_mirror)
        plane_left = distance_of(item.points)
        plane_right = distance_of(item.mirrored)
        if left is None or right is None or plane_left is None or plane_right is None:
            continue
        solid_original.append(left)
        solid_mirror.append(right)
        flat_original.append(plane_left)
        flat_mirror.append(plane_right)
        bias_frames.append(normal_mirror[partners] * flip - normal_original[indices])

    if not bias_frames:
        print("  鏡像側の z まで取れたフレームがありません（--section A も併せて実行すること）。")
        report["mirror_3d"] = {"frames": 0}
        return

    solid = np.asarray(solid_original, dtype=np.float64)
    solid_flipped = np.asarray(solid_mirror, dtype=np.float64)
    flat = np.asarray(flat_original, dtype=np.float64)
    flat_flipped = np.asarray(flat_mirror, dtype=np.float64)
    solid_gap = np.abs(solid_flipped - solid)
    flat_gap = np.abs(flat_flipped - flat)

    stacked = np.stack(bias_frames)
    bias_total = np.linalg.norm(stacked, axis=2)
    bias_plane = np.linalg.norm(stacked[:, :, :2], axis=2)
    bias_depth = np.abs(stacked[:, :, 2])
    systematic = np.linalg.norm(stacked.mean(axis=0), axis=1)

    print(f"  比較できたフレーム: {len(solid)} / {len(capture.detections)}")
    print()
    print("  1) d の水準")
    print(f"     {'':<14} {'d(I) 平均':>22} {'d(mirror(I)) 平均':>24}")
    print("     " + "-" * 62)
    for label, first, second in (
        ("現行 2D d", flat, flat_flipped),
        ("3D d_3d", solid, solid_flipped),
    ):
        print(
            f"     {label:<14} {both_units(float(first.mean())):>22} "
            f"{both_units(float(second.mean())):>24}"
        )
    print()

    print("  2) |d(I) - d(mirror(I))| — 幾何学的に正しい検出器なら 0 になる量")
    print_spread("現行 2D d の差", summarize(flat_gap))
    print_spread("3D d_3d の差", summarize(solid_gap))
    print()
    ratios: dict[str, float | None] = {}
    for label, gap, level in (("2D", flat_gap, flat), ("3D", solid_gap, solid)):
        mean_level = float(level.mean())
        share = float(np.median(gap)) / mean_level if mean_level > 0.0 else None
        ratios[label] = share
        if share is not None:
            print(
                f"     {label}: 差の中央値 ÷ d の平均 = {share:.2%}"
                "   ← 測定値のうち検出器の左右非対称で説明できる割合の目安"
            )
    flat_share, solid_share = ratios["2D"], ratios["3D"]
    # 2D の割合が測定誤差の水準しか無いときは、比を取っても意味が無い。
    if flat_share is not None and solid_share is not None and flat_share > 0.005:
        print(
            f"     → Procrustes が吸収した分: {flat_share:.2%} → {solid_share:.2%}"
            f"（{1.0 - solid_share / flat_share:.0%} 減）"
        )
    print()

    print("  3) landmark ごとの 3D 左右バイアス（sigma で入れ替え、x だけ反転して比較）")
    print_spread("フレームごとの差", summarize(bias_total))
    print_spread("うち xy 成分", summarize(bias_plane))
    print_spread("うち z 成分", summarize(bias_depth))
    print_spread("系統成分（3D）", summarize(systematic))
    print()
    print("  → 3) の z 成分が xy 成分より著しく小さければ、検出器の左右バイアスは")
    print("    面内で起きていて、z はそもそも左右差を持っていない（テストG と併せて読む）。")

    report["mirror_3d"] = {
        "frames": len(solid),
        "distance_3d_original": spread_json(summarize(solid)),
        "distance_3d_mirrored": spread_json(summarize(solid_flipped)),
        "distance_2d_original": spread_json(summarize(flat)),
        "distance_2d_mirrored": spread_json(summarize(flat_flipped)),
        "distance_3d_absolute_difference": spread_json(summarize(solid_gap)),
        "distance_2d_absolute_difference": spread_json(summarize(flat_gap)),
        "bias_share_of_level": ratios,
        "landmark_bias_3d": spread_json(summarize(bias_total)),
        "landmark_bias_3d_xy": spread_json(summarize(bias_plane)),
        "landmark_bias_3d_z": spread_json(summarize(bias_depth)),
        "landmark_bias_3d_systematic": spread_json(summarize(systematic)),
        "landmark_count": len(indices),
    }


# --------------------------------------------------------------------------
# テストG — Procrustes 残差の xy / z 分解
# --------------------------------------------------------------------------
RESIDUAL_NOTE = """
  d_3d は Procrustes 重ね合わせ後の残差 RMS である。残差はベクトルなので、
  面内（xy）成分と奥行き（z）成分へ分けられる。分解は厳密で、

      d_3d^2 = RMS_xy^2 + RMS_z^2

  が恒等的に成り立つ（この式が成り立つことを表の下で確認している）。

    z 成分が xy 成分と同程度      z は実際の顔の奥行き非対称を捉えている
    z 成分が xy より著しく小さい  z はテンプレート由来で、実質 2.5D

  ただし顔の奥行き方向の非対称が本質的に小さい可能性もあるので、これ単体では
  断定できない。テストE のゲインと併せて読むこと。
"""


def _residual_groups(available: int) -> tuple[tuple[str, list[int]], ...]:
    """残差行列の行 index を、部位ごとにまとめる。行の並びは MEASURED_INDICES。"""
    position = {landmark: order for order, landmark in enumerate(MEASURED_INDICES)}
    groups: list[tuple[str, list[int]]] = [("全体", list(range(available)))]
    for name in ("contour", "eyes", "nose", "mouth"):
        rows = [
            position[index]
            for pair in CONFIG.parts[name].pairs
            for index in pair
            if index in position and position[index] < available
        ]
        if rows:
            groups.append((name, rows))
    return tuple(groups)


def section_g(series: Series, report: dict[str, Any]) -> None:
    heading("G", "Procrustes 残差の xy / z 分解 — z は奥行きを見ているか")
    print(RESIDUAL_NOTE)

    residuals: list[NDArray[np.float64]] = []
    pair_gaps: list[NDArray[np.float64]] = []
    depth_spread: list[float] = []
    for frame3d, value in zip(series.normalized3d, series.distances3d, strict=True):
        if not isfinite(value):
            continue
        residual = procrustes_residual(frame3d)
        if residual is None:
            continue
        residuals.append(residual)
        lefts = [left for left, _ in CONFIG.all_pairs]
        rights = [right for _, right in CONFIG.all_pairs]
        # 鏡映対称なら x_l = -x_r、y_l = y_r、z_l = z_r。その破れを軸ごとに見る。
        gap = np.column_stack(
            [
                frame3d[lefts, 0] + frame3d[rights, 0],
                frame3d[lefts, 1] - frame3d[rights, 1],
                frame3d[lefts, 2] - frame3d[rights, 2],
            ]
        )
        pair_gaps.append(gap - gap.mean(axis=0))
        depth_spread.append(float(frame3d[list(MEASURED_INDICES), 2].std()))

    if not residuals:
        print("  z を取れたフレームがありません。")
        report["residual_axes"] = {"frames": 0}
        return

    stacked = np.stack(residuals)
    available = stacked.shape[1]
    print(f"  対象フレーム: {len(residuals)}")
    print()
    print("  1) 残差の分解（フレームと点をまとめた RMS）")
    print(f"     {'対象':<10} {'xy 成分':>22} {'z 成分':>22} {'z / xy':>10}")
    print("     " + "-" * 68)
    axes_report: dict[str, Any] = {}
    for label, rows in _residual_groups(available):
        block = stacked[:, rows, :]
        plane = float(np.sqrt((block[:, :, :2] ** 2).sum(axis=2).mean()))
        depth = float(np.sqrt((block[:, :, 2] ** 2).mean()))
        ratio = depth / plane if plane > 0.0 else float("nan")
        print(f"     {label:<10} {both_units(plane):>22} {both_units(depth):>22} {ratio:10.3f}")
        axes_report[label] = {
            "xy_rms": {"iod": plane, "mm": plane * IPD_MM},
            "z_rms": {"iod": depth, "mm": depth * IPD_MM},
            "z_over_xy": ratio,
        }
    total_plane = float(np.sqrt((stacked[:, :, :2] ** 2).sum(axis=2).mean()))
    total_depth = float(np.sqrt((stacked[:, :, 2] ** 2).mean()))
    combined = sqrt(total_plane**2 + total_depth**2)
    finite = series.distances3d[np.isfinite(series.distances3d)]
    quadratic_mean = float(np.sqrt((finite**2).mean()))
    print()
    print(
        f"     恒等式の確認: sqrt(xy^2 + z^2) = {combined:.6f} / "
        f"d_3d の二乗平均平方根 = {quadratic_mean:.6f}"
        f"（差 {abs(combined - quadratic_mean):.2e}）"
    )
    print()

    gaps = np.stack(pair_gaps)
    axis_rms = [float(np.sqrt((gaps[:, :, axis] ** 2).mean())) for axis in range(3)]
    print("  2) 重ね合わせ前の、ペアごとの生の左右差（軸別・フレームごとの平均を除いた後）")
    print(f"     {'軸':<28} {'RMS':>22}")
    print("     " + "-" * 52)
    for label, value in zip(
        ("x: x_left + x_right", "y: y_left - y_right", "z: z_left - z_right"),
        axis_rms,
        strict=True,
    ):
        print(f"     {label:<28} {both_units(value):>22}")
    reference = float(np.mean(depth_spread))
    print()
    print(f"     参考: 顔全体での z の広がり（計測点の標準偏差）{both_units(reference)}")
    print("     → z の広がりは大きいのに z の左右差だけが小さいなら、z は「奥行きの形は")
    print("       持つが左右で同一」——つまり左右対称なテンプレートの挙動である。")

    report["residual_axes"] = {
        "frames": len(residuals),
        "by_group": axes_report,
        "identity_check": {
            "sqrt_xy2_plus_z2": combined,
            "rms_of_distance_3d": quadratic_mean,
            "difference": abs(combined - quadratic_mean),
        },
        "pair_gap_rms": {
            "x": {"iod": axis_rms[0], "mm": axis_rms[0] * IPD_MM},
            "y": {"iod": axis_rms[1], "mm": axis_rms[1] * IPD_MM},
            "z": {"iod": axis_rms[2], "mm": axis_rms[2] * IPD_MM},
        },
        "depth_spread": {"iod": reference, "mm": reference * IPD_MM},
    }


# --------------------------------------------------------------------------
# テストH — 3D の姿勢層別
# --------------------------------------------------------------------------
def _depth_reproducibility(normalized3d: NDArray[np.float64]) -> tuple[float, float]:
    """計測点における z と xy の再現性（平均位置まわりの RMS の中央値）。"""
    available = normalized3d.shape[1]
    group = [index for index in MEASURED_INDICES if index < available]
    depths = normalized3d[:, :, 2]
    depth_rms = np.sqrt(((depths - depths.mean(axis=0)) ** 2).mean(axis=0))
    plane_rms, _ = landmark_spread(normalized3d[:, :, :2])
    return float(np.median(depth_rms[group])), float(np.median(plane_rms[group]))


def section_h(series: Series, report: dict[str, Any]) -> None:
    heading("H", "3D の姿勢層別 — 残存相関は z の誤差か、z の素のばらつきか")
    print("""
  テストD で d_3d と |yaw| の相関が残っていた。それが「z の誤差 × 姿勢の交互作用」
  なのか「z の素のばらつき」なのかを、姿勢を固定した部分集合で切り分ける。

    部分集合で z の再現性が大きく改善する  残存相関は z 誤差 × 姿勢の交互作用
    ほとんど変わらない                    z の素のばらつき
""")
    finite = np.isfinite(series.distances3d)
    if int(finite.sum()) < 2:
        print(f"  z を取れたフレームが足りません（{int(finite.sum())} / {series.count}）。")
        report["stable_yaw_3d"] = {"frames": 0}
        return

    selection = stable_yaw_mask(series)
    if selection is None:
        print("  姿勢を取得できたフレームがないので、層別できません。")
        report["stable_yaw_3d"] = {"frames": 0}
        return

    mask, band, median = selection
    subset = mask & finite
    count = int(subset.sum())
    print(
        f"  yaw 中央値 {median:+.2f}° / 帯域 ±{band:.1f}° / "
        f"該当 {count} 枚（z を取れた {int(finite.sum())} 枚のうち）"
    )
    if count < 2:
        print("  ※ 2 枚未満なので、ばらつきを計算できません。")
        report["stable_yaw_3d"] = {"frames": count, "band_deg": band, "median_yaw_deg": median}
        return
    if count < MIN_USEFUL_FRAMES:
        print(f"  ※ {MIN_USEFUL_FRAMES} 枚未満なので、以下の統計は目安にとどまる。")
    print()

    full_values = series.distances3d[finite]
    subset_values = series.distances3d[subset]
    full_floor = float(full_values.std(ddof=1))
    subset_floor = float(subset_values.std(ddof=1))

    print("  1) d_3d の分布とノイズ床")
    print(f"     {'':<12} {'枚数':>6} {'平均':>22} {'ノイズ床':>22} {'信号÷床':>10}")
    print("     " + "-" * 76)
    for label, values, floor in (
        ("全体", full_values, full_floor),
        (f"yaw ±{band:.1f}°", subset_values, subset_floor),
    ):
        ratio = SIGNAL_SPAN / floor if floor > 0.0 else float("nan")
        print(
            f"     {label:<12} {len(values):6d} {both_units(float(values.mean())):>22} "
            f"{both_units(floor):>22} {ratio:9.2f}x"
        )
    if full_floor > 0.0:
        print(f"     姿勢を除いた床は全体の {subset_floor / full_floor:.0%}")
    print()

    full_depth, full_plane = _depth_reproducibility(series.normalized3d[finite])
    subset_depth, subset_plane = _depth_reproducibility(series.normalized3d[subset])
    print("  2) 正規化空間での landmark ごとの再現性（計測点の中央値）")
    print(f"     {'':<12} {'z RMS':>22} {'xy RMS':>22} {'z / xy':>10}")
    print("     " + "-" * 68)
    for label, depth, plane in (
        ("全体", full_depth, full_plane),
        (f"yaw ±{band:.1f}°", subset_depth, subset_plane),
    ):
        ratio = depth / plane if plane > 0.0 else float("nan")
        print(f"     {label:<12} {both_units(depth):>22} {both_units(plane):>22} {ratio:10.3f}")
    improvement = 1.0 - subset_depth / full_depth if full_depth > 0.0 else float("nan")
    print()
    print(f"     z の再現性の改善: {improvement:.0%}（大きいほど、姿勢との交互作用が主因）")
    print(
        f"     判定閾値（docs level3 §4）: z の誤差 {Z_ERROR_BUDGET:.0%} IOD "
        f"= {Z_ERROR_BUDGET * IPD_MM:.2f}mm"
    )
    print()

    yaws = np.abs(yaw_values(series))
    pose_ok = subset & np.isfinite(yaws)
    correlation = (
        _correlation(yaws[pose_ok], series.distances3d[pose_ok])
        if int(pose_ok.sum()) >= 3
        else None
    )
    print("  3) 部分集合の中での |yaw| と d_3d の相関")
    print(f"     {'—' if correlation is None else f'{correlation:+.3f}'}")
    print("     ※ 帯域を絞ると yaw の分散がほぼ無くなるので、この相関は不安定である。")
    print("       判断の主材料は 2) の z 再現性であって、この係数ではない。")

    verdict: list[str] = []
    if isfinite(improvement):
        if improvement >= 0.5:
            verdict.append(
                f"部分集合で z の再現性が {improvement:.0%} 改善した"
                "。残存相関は z 誤差 × 姿勢の交互作用が主因。"
            )
        else:
            verdict.append(
                f"部分集合でも z の再現性は {improvement:.0%} しか改善しない"
                "。z の素のばらつきが主因。"
            )
    if subset_depth > Z_ERROR_BUDGET:
        verdict.append(f"姿勢を固定しても z の再現性は閾値超え（{subset_depth:.4f} IOD）")
    else:
        verdict.append(f"姿勢を固定すれば z の再現性は閾値内（{subset_depth:.4f} IOD）")
    print()
    print("  判定")
    for line in verdict:
        print(f"     - {line}")

    report["stable_yaw_3d"] = {
        "frames": count,
        "frames_with_depth": int(finite.sum()),
        "band_deg": band,
        "median_yaw_deg": median,
        "distance_3d_full": spread_json(summarize(full_values)),
        "distance_3d_subset": spread_json(summarize(subset_values)),
        "noise_floor_full": {"iod": full_floor, "mm": full_floor * IPD_MM},
        "noise_floor_subset": {"iod": subset_floor, "mm": subset_floor * IPD_MM},
        "z_reproducibility_full": {"iod": full_depth, "mm": full_depth * IPD_MM},
        "z_reproducibility_subset": {"iod": subset_depth, "mm": subset_depth * IPD_MM},
        "xy_reproducibility_full": {"iod": full_plane, "mm": full_plane * IPD_MM},
        "xy_reproducibility_subset": {"iod": subset_plane, "mm": subset_plane * IPD_MM},
        "z_improvement": improvement,
        "abs_yaw_correlation_3d_subset": correlation,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# セクション I / J 共通 — 部位ごとの注入測定
# --------------------------------------------------------------------------
def collect_part_injections(
    arguments: argparse.Namespace,
    capture: Capture,
) -> dict[str, InjectionReport] | None:
    """部位ごとの注入測定を 1 度だけ行い、セクション I と J で共有する。

    I が要るのは口 1 部位だけだが、J が 4 部位を要る。変形と再検出は測定の中で
    いちばん重いので、両方を選んだときに 2 度走らせない。フレームの選び方も
    変位場もテストE と同じで、変えているのは注入する点の集合だけである。
    """
    if arguments.self_test:
        print("  合成 capture には元画像が無いので、注入測定は行わない。")
        return None

    chosen = _median_yaw_detections(capture, arguments.gain_frames)
    if not chosen:
        print("  姿勢を取得できたフレームがないので、注入測定を行えません。")
        return None

    model_path, model_version = resolve_model(arguments.model)
    collected: dict[str, list[InjectionReport]] = {request.name: [] for request in PART_REQUESTS}
    used = 0
    for detection in chosen:
        image = frame_at(arguments, detection.index)
        if image is None:
            print(f"  フレーム {detection.index} を読み直せませんでした。飛ばします。")
            continue
        measured = _inject_one_frame(
            image,
            model_path=model_path,
            model_version=model_version,
            mode=arguments.mode,
            requests=PART_REQUESTS,
        )
        if measured is None:
            print(f"  フレーム {detection.index} は基準を作れませんでした。飛ばします。")
            continue
        used += 1
        for name, item in measured.items():
            collected[name].append(item)

    if not used:
        print("  注入測定できたフレームがありません。")
        return None
    print(f"  注入測定に使ったフレーム: {used} 枚 / 部位 {len(PART_REQUESTS)} 通り")
    return {name: aggregate_injection(reports) for name, reports in collected.items() if reports}


# --------------------------------------------------------------------------
# セクション I — 3 方式の直接比較
# --------------------------------------------------------------------------
THREE_WAY_NOTE = """
  比べる 3 方式

    ① 現行 2D        §8.3 正規化 → 正中線あてはめ → 鏡映 → ペア距離の RMS
    ② 2D Procrustes  §8.3 正規化 → 鏡映 → sigma でペア名入替 → 2D の並進と回転で
                     Procrustes 重ね合わせ → 残差 RMS。z を一切使わない
    ③ 3D Procrustes  ② と同じ手順を xyz で行う（テストD の d_3d）

  ① と ② の差は「折る平面を先に決めるかどうか」だけである。① は正中点から
  あてはめた 1 本の直線で折る。② はどの平面で折るかを先に決めず、いちばん合う
  重ね合わせを後から選ぶ。② と ③ の差は z を使うかどうかだけである。

  したがって 3 つを並べれば、テストD で見えた「① より ③ が良い」の理由を
  Procrustes と z へ分けられる。

    ② が ③ に近い  効いているのは Procrustes であって z ではない
    ② が ① 寄り    効いているのは z である

  ② に残る原理的な限界: 単位が 2D 投影の両眼間距離なので、yaw があると
  1/cos(yaw) 倍に膨らむ。奥行き方向の非対称も像には現れない。どちらも z 無しでは
  消せない。それでも正中線あてはめより自由度が大きいので、実測でどこまで届くかは
  測るしかない。

  ノイズ床・信号÷床・|yaw| 相関はテストB とテストD、鏡像差はテストA、ゲインは
  テストE と同じ定義である。ここで新しい量は導入していない。
"""

#: 方式 key → `_injection_slope` の軸名。
_SLOPE_AXIS: dict[str, str] = {
    "current_2d": "2d",
    "procrustes_2d": "2dp",
    "procrustes_3d": "3d",
}

#: ① と ③ の差がこの割合より小さい量では、② の位置を読まない（分母が小さすぎて
#: 位置が発散する）。
POSITION_MIN_SPAN_RATIO: float = 0.05


def _mirror_gap_by_method(capture: Capture) -> tuple[dict[str, NDArray[np.float64]], int]:
    """|d(I) - d(mirror(I))| を 3 方式それぞれで測る。定義はテストA の 2) と同じ。

    使えるフレーム数は方式で違う（③ は鏡像側の z も要る）ので、方式ごとに集める。
    返り値の第 2 要素は、鏡像側を 2D で比較できたフレーム数である。
    """
    gaps: dict[str, list[float]] = {key: [] for key in METHOD_KEYS}
    frames = 0
    for item in capture.detections:
        if item.mirrored is None:
            continue
        normal = normalized_of(item.points)
        normal_mirror = normalized_of(item.mirrored)
        if normal is None or normal_mirror is None:
            continue
        frames += 1

        solid = solid_mirror = None
        if item.points3d is not None and item.mirrored3d is not None:
            straight = normalized_3d_of(item.points3d)
            flipped = normalized_3d_of(item.mirrored3d)
            solid = None if straight is None else distance_3d_of(straight)
            solid_mirror = None if flipped is None else distance_3d_of(flipped)

        measured: tuple[tuple[str, float | None, float | None], ...] = (
            ("current_2d", distance_of(item.points), distance_of(item.mirrored)),
            (
                "procrustes_2d",
                distance_2d_procrustes_of(normal),
                distance_2d_procrustes_of(normal_mirror),
            ),
            ("procrustes_3d", solid, solid_mirror),
        )
        for key, left, right in measured:
            if left is not None and right is not None:
                gaps[key].append(abs(right - left))
    return {key: np.asarray(values, dtype=np.float64) for key, values in gaps.items()}, frames


def _response_median(measured: InjectionReport) -> float | None:
    """landmark 水準の応答（実変位 ÷ δ）の中央値。折り返した δ は除く。

    基準フレームを固定して測った `realized_fixed_mm` を使う。眼アンカー自身が
    動く部位では、正規化をやり直す `realized_mm` は応答ではなくフレームの移動を
    含んでしまう。
    """
    values = [
        item.realized_fixed_mm / item.delta_mm
        for item in measured.outcomes
        if item.delta_mm > 0.0
        and item.reason is None
        and item.delta_mm <= measured.fold_mm
        and isfinite(item.realized_fixed_mm)
    ]
    return float(np.median(values)) if values else None


def _between(first: float | None, middle: float | None, last: float | None) -> float | None:
    """① を 0、③ を 1 としたときの ② の位置。① と ③ に差が無い量では読まない。"""
    if first is None or middle is None or last is None:
        return None
    span = last - first
    reference = max(abs(first), abs(last))
    if reference <= 0.0 or abs(span) < POSITION_MIN_SPAN_RATIO * reference:
        return None
    return (middle - first) / span


def section_i(
    capture: Capture,
    series: Series,
    injections: dict[str, InjectionReport] | None,
    report: dict[str, Any],
) -> None:
    heading("I", "3 方式の直接比較 — 効いているのは Procrustes か z か")
    print(THREE_WAY_NOTE)

    mask = np.isfinite(series.distances3d) & np.isfinite(series.distances2dp)
    usable = int(mask.sum())
    if usable < 2:
        print(f"  3 方式をそろえられたフレームが足りません（{usable} / {series.count}）。")
        report["three_way"] = {"frames": usable}
        return

    values: dict[str, NDArray[np.float64]] = {
        "current_2d": series.distances[mask],
        "procrustes_2d": series.distances2dp[mask],
        "procrustes_3d": series.distances3d[mask],
    }
    yaws = np.abs(yaw_values(series))[mask]
    pose_ok = np.isfinite(yaws)
    gaps, mirror_frames = _mirror_gap_by_method(capture)
    field = None if injections is None else injections.get(PART_LABELS["mouth"])
    response = None if field is None else _response_median(field)

    print(f"  対象フレーム: {usable} / {series.count}（3 方式とも計算できたフレーム）")
    if usable < MIN_USEFUL_FRAMES:
        print(f"  ※ {MIN_USEFUL_FRAMES} 枚未満なので、以下の統計は目安にとどまる。")
    print(f"  鏡像差に使えたフレーム: {mirror_frames}（0 なら --section A も併せて実行すること）")
    if field is None:
        print("  ゲイン列は、注入測定を実行できたときだけ入る。")
    else:
        print(
            f"  ゲインは部位「{PART_LABELS['mouth']}」の領域変位場"
            f"（sigma = {field.sigma_iod:.2f} IOD、{field.frames} 枚の中央値）による。"
        )
    print()

    print("  1) 3 方式を同じ材料で並べる")
    print(
        f"     {'方式':<16} {'平均':>22} {'ノイズ床':>22} {'信号÷床':>8}"
        f" {'|yaw|相関':>10} {'鏡像差中央値':>12} {'landmark応答':>12} {'d ゲイン':>9}"
    )
    print("     " + "-" * 118)
    rows: dict[str, Any] = {}
    for key in METHOD_KEYS:
        sample = values[key]
        floor = float(sample.std(ddof=1))
        correlation = _correlation(yaws[pose_ok], sample[pose_ok])
        gap = gaps[key]
        gap_median = float(np.median(gap)) if gap.size else None
        gain = None if field is None else _injection_slope(field, axis=_SLOPE_AXIS[key])
        print(
            f"     {METHOD_LABELS[key]:<16} {both_units(float(sample.mean())):>22}"
            f" {both_units(floor):>22}"
            f" {SIGNAL_SPAN / floor if floor > 0.0 else float('nan'):7.2f}x"
            f" {'—' if correlation is None else f'{correlation:+.3f}':>10}"
            f" {'—' if gap_median is None else f'{gap_median * IPD_MM:.4f}mm':>12}"
            f" {'—' if response is None else f'{response:.3f}':>12}"
            f" {_ratio_or_dash(gain):>9}"
        )
        rows[key] = {
            "label": METHOD_LABELS[key],
            "frames": int(sample.size),
            "distance": spread_json(summarize(sample)),
            "noise_floor": {"iod": floor, "mm": floor * IPD_MM},
            "signal_to_noise": SIGNAL_SPAN / floor if floor > 0.0 else None,
            "abs_yaw_correlation": correlation,
            "mirror_gap_median": (
                None
                if gap_median is None
                else {"iod": gap_median, "mm": gap_median * IPD_MM, "frames": int(gap.size)}
            ),
            "landmark_response": response,
            "gain": gain,
        }
    print()
    print("     landmark 応答は方式に依らない量（変形画像から再検出した注入点の移動量 ÷ δ）")
    print("     なので、3 行に同じ値が入る。方式で変わるのは、その応答を d が")
    print("     どれだけ拾うかを表す「d ゲイン」のほうである。")
    print()

    print("  2) ② は ① と ③ のどちらに近いか（① を 0、③ を 1 とした位置）")
    print(f"     {'量':<20} {'①':>14} {'②':>14} {'③':>14} {'②の位置':>10}")
    print("     " + "-" * 78)
    quantities: tuple[tuple[str, str, dict[str, float | None]], ...] = (
        (
            "noise_floor",
            "ノイズ床(mm)",
            {key: rows[key]["noise_floor"]["mm"] for key in METHOD_KEYS},
        ),
        (
            "abs_yaw_correlation",
            "|yaw| 相関",
            {
                key: (
                    None
                    if rows[key]["abs_yaw_correlation"] is None
                    else abs(float(rows[key]["abs_yaw_correlation"]))
                )
                for key in METHOD_KEYS
            },
        ),
        (
            "mirror_gap",
            "鏡像差中央値(mm)",
            {
                key: (
                    None
                    if rows[key]["mirror_gap_median"] is None
                    else float(rows[key]["mirror_gap_median"]["mm"])
                )
                for key in METHOD_KEYS
            },
        ),
        ("gain", "d ゲイン", {key: rows[key]["gain"] for key in METHOD_KEYS}),
    )
    positions: dict[str, float | None] = {}
    for name, label, triple in quantities:
        place = _between(triple["current_2d"], triple["procrustes_2d"], triple["procrustes_3d"])
        positions[name] = place
        print(
            f"     {label:<20}"
            + "".join(
                f" {'—' if triple[key] is None else f'{triple[key]:.4f}':>14}"
                for key in METHOD_KEYS
            )
            + f" {'—' if place is None else f'{place:+.2f}':>10}"
        )
    print()
    print("     位置が 1.0 なら ② は ③ と同じ、0.0 なら ① と同じである。① と ③ の差が")
    print(
        f"     {POSITION_MIN_SPAN_RATIO:.0%} 未満の量は、位置の分母が小さすぎるので読まない（—）。"
    )
    print()

    verdict = _three_way_verdict(positions)
    print("  判定")
    for line in verdict:
        print(f"     - {line}")

    report["three_way"] = {
        "frames": usable,
        "mirror_frames": mirror_frames,
        "gain_field": None if field is None else field.field_name,
        "gain_frames": None if field is None else field.frames,
        "landmark_response": response,
        "methods": rows,
        "position_of_2d_procrustes": positions,
        "verdict": verdict,
    }


def _three_way_verdict(positions: dict[str, float | None]) -> list[str]:
    """② の位置から、実装方式の含意まで書き下す。"""
    verdict: list[str] = []
    place = [value for value in positions.values() if value is not None]
    if not place:
        verdict.append("① と ③ に差のある量が 1 つも無い。この材料では 3 方式を分離できない。")
        return verdict

    centre = float(np.median(place))
    verdict.append(
        f"② の位置は {len(place)} 量の中央値で {centre:+.2f}"
        f"（{min(place):+.2f}〜{max(place):+.2f}）"
    )
    if centre >= NEAR_3D_POSITION:
        verdict.append(
            f"② は ③ に近い（位置 ≧ {NEAR_3D_POSITION:.1f}）。効いているのは Procrustes であり、"
            "z ではない。"
        )
        verdict.append(
            "含意: 2D のまま実装できる。adapter の DTO へ z を通す変更も、z の精度という"
            "未解決要因も要らない。"
        )
    elif centre <= NEAR_2D_POSITION:
        verdict.append(
            f"② は ① 寄りである（位置 ≦ {NEAR_2D_POSITION:.1f}）。効いているのは z である。"
        )
        verdict.append(
            "含意: 3D 化が要る。adapter・DTO・domain へ z を通す変更と、"
            "テストH で見た z の再現性の扱いが前提になる。"
        )
    else:
        verdict.append(
            "② は ① と ③ の中間である。Procrustes と z の両方が効いており、"
            "どちらか一方では説明できない。"
        )
    verdict.append(
        "この位置は量ごとに散る。ノイズ床と |yaw| 相関が同じ向きを指しているかを、"
        "1) の表で確かめること。"
    )
    return verdict


# --------------------------------------------------------------------------
# セクション J — 注入部位を変えたゲイン
# --------------------------------------------------------------------------
PART_GAIN_NOTE = """
  テストE と同じ領域変位場（Gauss、sigma = 0.12 IOD）を、口以外の部位へも当てる。
  変えるのは注入する点の集合だけで、変位場の作り方も δ の刻みも判定の式も
  テストE と同一である。

  知りたいのは「landmark 応答（実変位 ÷ δ）が検出器の安定した性質かどうか」で
  ある。部位によらず同じ水準なら、それは MediaPipe が注入された変位をどれだけ
  追随するかという性質である。部位で大きく振れるなら、口の周辺に固有の何か
  （テクスチャ、テンプレートの拘束、三角形の密度）を見ていたことになる。

  複数の被写体で測り直せば同じ問いに直接答えられるが、他人の顔の撮影・保管は
  不変条件 1 に触れる。部位を変えることで代替している。

  読むときの注意

    部位「目」は §8.3 の眼アンカー（左右各 2 点）そのものを動かす。正規化フレーム
    が注入で動くため、d の真値・検出値の両方がその影響を受ける。ゲインは真値と
    検出値を同じ規約で測った比なので依然として読めるが、応答は基準フレームを
    固定して測った値（実変位 ÷ δ）を使う必要がある。表の「眼アンカー重み」が
    0 より大きい行がこれに当たる。
"""

#: 部位ごとの landmark 応答が、最大 ÷ 最小でこの倍率を超えたら部位依存とみなす。
#: 較正値ではなく、表を読むための目盛りである。
RESPONSE_SPREAD_ALARM: float = 1.5


def section_j(injections: dict[str, InjectionReport] | None, report: dict[str, Any]) -> None:
    heading("J", "注入部位を変えたゲイン — landmark 応答は検出器の性質か")
    print(PART_GAIN_NOTE)

    if not injections:
        print("  注入測定を実行できていません。")
        report["injection_by_part"] = {"parts": 0}
        return

    measured = [injections[label] for label in PART_LABELS.values() if label in injections]
    if not measured:
        print("  測定できた部位がありません。")
        report["injection_by_part"] = {"parts": 0}
        return

    print(f"  対象フレーム: {measured[0].frames} 枚（yaw 中央値の近傍、テストE と同じ選び方）")
    print(f"  変位: δ = {', '.join(f'{value:.0f}' for value in INJECTION_DELTAS_MM)} mm")
    print()
    print("  1) 部位ごとの応答とゲイン")
    print(
        f"     {'部位':<8} {'注入点':>6} {'影響点':>6} {'裏返りδ':>9} {'眼アンカー重み':>14}"
        f" {'landmark応答':>12} {'①ゲイン':>9} {'②ゲイン':>9} {'③ゲイン':>9}"
    )
    print("     " + "-" * 96)

    rows: list[dict[str, Any]] = []
    responses: list[float] = []
    for item in measured:
        response = _response_median(item)
        gains = {key: _injection_slope(item, axis=_SLOPE_AXIS[key]) for key in METHOD_KEYS}
        if response is not None:
            responses.append(response)
        print(
            f"     {item.field_name:<8} {item.injected_count:6d} {item.affected_count:6d}"
            f" {item.fold_mm:8.2f}mm {item.anchor_weight:14.3f}"
            f" {'—' if response is None else f'{response:.3f}':>12}"
            + "".join(f" {_ratio_or_dash(gains[key]):>9}" for key in METHOD_KEYS)
        )
        rows.append(
            {
                "part": item.part,
                "label": item.field_name,
                "sigma_iod": item.sigma_iod,
                "injected_landmarks": item.injected_count,
                "affected_landmarks": item.affected_count,
                "fold_threshold_mm": item.fold_mm,
                "eye_anchor_weight": item.anchor_weight,
                "landmark_response": response,
                "baseline_2d_mm": item.baseline_2d * IPD_MM,
                "baseline_2dp_mm": _scaled(item.baseline_2dp),
                "baseline_3d_mm": _scaled(item.baseline_3d),
                "gain": {key: gains[key] for key in METHOD_KEYS},
                "folded_deltas_mm": [
                    row.delta_mm for row in item.outcomes if row.delta_mm > item.fold_mm
                ],
                "rejected_deltas_mm": [
                    row.delta_mm for row in item.outcomes if row.reason is not None
                ],
            }
        )
    print()
    print("  裏返りδ = その変位場が三角形を裏返し始める δ。これを超えた δ は判定から除く。")
    print("  眼アンカー重み = 変位場が §8.3 の眼アンカー 4 点へ与える重みの最大値。")
    print("  landmark 応答 = 基準の正規化フレームを固定して測った、注入点の横移動 ÷ δ。")
    print()

    verdict = _part_gain_verdict(rows, responses)
    print("  判定")
    for line in verdict:
        print(f"     - {line}")

    report["injection_by_part"] = {
        "parts": len(rows),
        "frames": measured[0].frames,
        "sigma_iod": REGION_SIGMA_IOD,
        "deltas_mm": list(INJECTION_DELTAS_MM),
        "rows": rows,
        "verdict": verdict,
    }


def _part_gain_verdict(rows: list[dict[str, Any]], responses: list[float]) -> list[str]:
    """部位ごとの応答が、部位に依らない性質と読めるかどうか。"""
    verdict: list[str] = []
    if not responses:
        verdict.append("応答を測れた部位がない。折り返しか検出失敗で、全部位が判定から外れている。")
        return verdict

    lowest, highest = min(responses), max(responses)
    verdict.append(
        f"landmark 応答: 中央値 {float(np.median(responses)):.3f}"
        f"（{lowest:.3f}〜{highest:.3f}、{len(responses)} 部位）"
    )
    spread = highest / lowest if lowest > 0.0 else float("inf")
    if spread <= RESPONSE_SPREAD_ALARM:
        verdict.append(
            f"部位間の振れは {spread:.2f} 倍（目盛り {RESPONSE_SPREAD_ALARM:.1f} 倍以内）。"
            "応答は部位に依らない検出器の性質と読める。"
        )
    else:
        verdict.append(
            f"部位間の振れが {spread:.2f} 倍（目盛り {RESPONSE_SPREAD_ALARM:.1f} 倍超）。"
            "応答は部位に依存しており、口で測った値を一般化できない。"
        )

    folded = [row["label"] for row in rows if row["folded_deltas_mm"]]
    if folded:
        verdict.append(
            "折り返しのため一部の δ を除いた部位: "
            + ", ".join(folded)
            + "（部位ごとに landmark の間隔が違うので、裏返り閾値も変わる）"
        )
    rejected = [row["label"] for row in rows if row["rejected_deltas_mm"]]
    if rejected:
        verdict.append("測れなかった δ がある部位: " + ", ".join(rejected))
    moved_anchor = [row["label"] for row in rows if row["eye_anchor_weight"] > 0.05]
    if moved_anchor:
        verdict.append(
            "正規化フレームごと動いた部位: "
            + ", ".join(moved_anchor)
            + "（眼アンカーに重みが乗っている。ゲインは読めるが、d の水準は他部位と直接比べない）"
        )
    return verdict


# --------------------------------------------------------------------------
# セクション K — 3D 正規化座標の平均
# --------------------------------------------------------------------------
SMOOTHING_3D_NOTE = """
  テストC の座標 EMA は 2D 座標にしか当てていない。同じ式を 3D 正規化座標へ当て、
  d_3d のノイズ床がどこまで下がるかを測る。比べる 2 つはテストC と同じである。

    座標を EMA  … §8.3 を 3D へ広げた正規化座標へ EMA。誤差は符号を持つので
                  打ち消し合う
    d_3d を EMA … 現行の domain/smoothing.py と同じ式。d_3d は残差 RMS なので
                  必ず 0 以上で、検出誤差は d_3d を押し上げる方向にしか働かない

  姿勢はすでに Procrustes が消しているので、ここに残っているのは i.i.d. に近い
  誤差のはずである。実測の標準偏差比が理論値 sqrt(alpha/(2-alpha)) に近ければ
  その仮説が裏づけられ、理論より大きければ時間的に相関した成分（被写体の微動、
  z 推定のドリフト）が残っていることになる。

  z を取れなかったフレームは落として測る。フレーム間隔はその分だけ不揃いになる
  ので、alpha は実測の間隔から求める。
"""

#: 実測 SD 比 ÷ 理論 SD 比がこれを超えたら、i.i.d. では説明できないとみなす。
#: 較正値ではなく、表を読むための目盛りである。
IID_RATIO_ALARM: float = 1.2


def section_k(capture: Capture, series: Series, report: dict[str, Any]) -> None:
    heading("K", "3D 座標の平均 — d_3d のノイズ床は座標 EMA でどこまで下がるか")
    print(SMOOTHING_3D_NOTE)

    mask = np.isfinite(series.distances3d)
    usable = int(mask.sum())
    if usable < 2:
        print(f"  z を取れたフレームが足りません（{usable} / {series.count}）。")
        report["smoothing_3d"] = {"frames": usable}
        return

    times = series.times[mask]
    coordinates = series.normalized3d[mask]
    distances = series.distances3d[mask]
    steps = np.diff(times)
    delta = float(np.median(steps)) if steps.size else capture.delta_seconds
    if delta <= 0.0:
        print("  フレーム間隔を求められません。")
        report["smoothing_3d"] = {"frames": usable}
        return
    max_gap = max(2.0, 5.0 * delta)

    print(f"  対象フレーム: {usable} / {series.count}（z を取れたフレーム）")
    if usable < MIN_USEFUL_FRAMES:
        print(f"  ※ {MIN_USEFUL_FRAMES} 枚未満なので、以下の統計は目安にとどまる。")
    print(
        f"  フレーム間隔 delta_t = {delta:.4f} 秒（中央値、{1.0 / delta:.2f} fps 相当）"
        f" / 撮影間隔 {capture.delta_seconds:.4f} 秒"
    )
    print(f"  EMA の reset 閾値 {max_gap:.2f} 秒")
    print()

    raw_rms, raw_depth = landmark_spread_3d(coordinates)
    available = coordinates.shape[1]
    group = [index for index in MEASURED_INDICES if index < available]
    raw_mean = float(distances.mean())
    raw_floor = float(distances.std(ddof=1))

    print(
        f"  {'tau':>6} {'tau_app':>9} {'alpha':>7} {'a/(2-a)':>9} {'理論SD比':>9}"
        f" {'実測3D SD比':>12} {'実測z SD比':>11}"
        f" | {'座標EMA d_3d平均':>17} {'ノイズ床':>11}"
        f" | {'d_3d EMA 平均':>14} {'ノイズ床':>11}"
    )
    print("  " + "-" * 136)
    print(
        f"  {'生値':>6} {'—':>9} {'—':>7} {'—':>9} {'—':>9} {1.0:12.3f} {1.0:11.3f}"
        f" | {raw_mean:17.5f} {raw_floor:11.5f}"
        f" | {raw_mean:14.5f} {raw_floor:11.5f}"
    )

    rows: list[dict[str, Any]] = []
    for tau in DEFAULT_TAUS:
        alpha = 1.0 - exp(-delta / tau)
        variance_ratio = alpha / (2.0 - alpha)
        theoretical = sqrt(variance_ratio)
        tau_app = app_tau_for(alpha)

        smoothed = ema_array(coordinates, times, tau_seconds=tau, max_gap_seconds=max_gap)
        scalar = ema_scalar(distances, times, tau_seconds=tau, max_gap_seconds=max_gap)
        burn, warmed = _burn_in(usable, tau, delta)
        window = slice(burn, None)
        smoothed_rms, smoothed_depth = landmark_spread_3d(smoothed[window])
        solid_ratio = float(np.median(smoothed_rms[group] / np.maximum(raw_rms[group], 1e-12)))
        depth_ratio = float(np.median(smoothed_depth[group] / np.maximum(raw_depth[group], 1e-12)))

        coordinate_d = np.asarray(
            [
                value
                for value in (distance_3d_of(frame) for frame in smoothed[window])
                if value is not None
            ],
            dtype=np.float64,
        )
        scalar_d = scalar[window]
        if coordinate_d.size < 2:
            continue

        marker = " " if warmed else "*"
        print(
            f" {marker}{tau:6.1f} {tau_app:8.1f}s {alpha:7.3f} {variance_ratio:9.3f}"
            f" {theoretical:9.3f} {solid_ratio:12.3f} {depth_ratio:11.3f}"
            f" | {coordinate_d.mean():17.5f} {coordinate_d.std(ddof=1):11.5f}"
            f" | {scalar_d.mean():14.5f} {scalar_d.std(ddof=1):11.5f}"
        )
        rows.append(
            {
                "tau_seconds": tau,
                "tau_app_seconds": tau_app,
                "app_delta_seconds": APP_DELTA_SECONDS,
                "alpha": alpha,
                "variance_ratio_theory": variance_ratio,
                "std_ratio_theory": theoretical,
                "coordinate_std_ratio_measured_3d": solid_ratio,
                "coordinate_std_ratio_measured_z": depth_ratio,
                "frames_used": int(coordinate_d.size),
                "warmed_up": warmed,
                "coordinate_ema_distance_3d": spread_json(summarize(coordinate_d)),
                "scalar_ema_distance_3d": spread_json(summarize(scalar_d)),
                "noise_floor_coordinate_ema": {
                    "iod": float(coordinate_d.std(ddof=1)),
                    "mm": float(coordinate_d.std(ddof=1)) * IPD_MM,
                },
            }
        )

    print()
    print(f"  （tau ごとに立ち上がり分を除外。除外は最大 {usable // 3} 枚）")
    if any(not row["warmed_up"] for row in rows):
        print("  * その tau に対してフレームが足りず、EMA が定常に達していない行。")
    print("  読み方: 実測 SD 比が理論 SD 比に近ければ、3D 正規化座標の誤差は i.i.d. に")
    print("          近い。理論より大きければ、姿勢を Procrustes で消した後も時間的に")
    print("          相関した成分（被写体の微動、z のドリフト）が残っている。")
    print(
        f"          tau_app は、解析 {APP_DELTA_SECONDS} 秒間隔のアプリで同じ alpha を得る時定数。"
    )
    print()

    verdict = _smoothing_3d_verdict(rows, raw_floor)
    print("  判定")
    for line in verdict:
        print(f"     - {line}")

    report["smoothing_3d"] = {
        "frames": usable,
        "delta_seconds": delta,
        "capture_delta_seconds": capture.delta_seconds,
        "app_delta_seconds": APP_DELTA_SECONDS,
        "max_gap_seconds": max_gap,
        "raw_distance_3d": spread_json(summarize(distances)),
        "raw_noise_floor": {"iod": raw_floor, "mm": raw_floor * IPD_MM},
        "rows": rows,
        "verdict": verdict,
    }


def _smoothing_3d_verdict(rows: list[dict[str, Any]], raw_floor: float) -> list[str]:
    """座標 EMA が d_3d の床をどこまで下げたか、理論とどれだけ合うか。"""
    verdict: list[str] = []
    warmed = [row for row in rows if row["warmed_up"]]
    if not warmed:
        verdict.append("EMA が定常に達した tau が 1 つも無い。フレーム数を増やすこと。")
        return verdict

    best = min(warmed, key=lambda row: float(row["noise_floor_coordinate_ema"]["iod"]))
    floor = float(best["noise_floor_coordinate_ema"]["iod"])
    verdict.append(
        f"定常に達した tau のうち、tau = {best['tau_seconds']:.1f}s"
        f"（tau_app {best['tau_app_seconds']:.1f}s）でノイズ床が最小: "
        f"{both_units(raw_floor)} → {both_units(floor)}"
        + (f"（{floor / raw_floor:.0%}）" if raw_floor > 0.0 else "")
    )
    if floor > 0.0:
        line = f"そのときの 信号 ÷ ノイズ床 = {SIGNAL_SPAN / floor:.2f}x"
        if raw_floor > 0.0:
            line += f"（生値では {SIGNAL_SPAN / raw_floor:.2f}x）"
        verdict.append(line)

    gaps = [
        float(row["coordinate_std_ratio_measured_3d"]) / float(row["std_ratio_theory"])
        for row in warmed
        if float(row["std_ratio_theory"]) > 0.0
    ]
    if gaps:
        centre = float(np.median(gaps))
        verdict.append(f"実測 SD 比 ÷ 理論 SD 比の中央値 {centre:.2f}（1.0 なら i.i.d.）")
        if centre <= IID_RATIO_ALARM:
            verdict.append(
                "理論に近い。姿勢を Procrustes が消した後の残差は i.i.d. に近い誤差であり、"
                "座標平均がそのまま効く。"
            )
        else:
            verdict.append(
                "理論より大きい。時間的に相関した成分が残っており、平均だけでは"
                "床がここまでしか下がらない。"
            )
    return verdict


# --------------------------------------------------------------------------
# 合成フレーム（--self-test）
# --------------------------------------------------------------------------
def synthetic_capture(
    *,
    frames: int,
    sigma: float,
    yaw_deg: float,
    delta_seconds: float,
    mirror_offset: float = 0.0,
    seed: int = 0,
) -> Capture:
    """landmark を直接注入した合成 capture。既知の誤差だけが入っている。

    3D 顔モデルは `symmetry_level3_probe.py` のものを流用する。鏡像側は
    「幾何学的に正しい検出器」の応答、すなわち sigma で index を入れ替えて
    x を反転したものを作る。`mirror_offset` を与えると、輪郭の左側だけを
    ずらした既知の左右バイアスを鏡像側へ注入する（統計量の感度確認用であり、
    実在の検出器の挙動モデルではない）。
    """
    rng = np.random.default_rng(seed)
    rotated = np.asarray(FACE3D @ yaw_matrix(yaw_deg).T, dtype=np.float64)
    truth = rotated[:, :2]
    flip = np.array([-1.0, 1.0], dtype=np.float64)

    injected: list[int] = []
    if mirror_offset:
        for left, right in CONFIG.parts["contour"].pairs:
            injected.append(left if truth[left][0] < truth[right][0] else right)

    detections: list[Detection] = []
    for index in range(frames):
        # 2D と 3D で同じ誤差を見せるため、xyz を一度に引いて xy を切り出す。
        noise = rng.normal(0.0, sigma, size=rotated.shape) if sigma > 0 else np.zeros_like(rotated)
        points3d = rotated + noise
        points = points3d[:, :2].copy()

        mirrored = np.empty_like(points)
        flip3d = np.array([-1.0, 1.0, 1.0], dtype=np.float64)
        mirrored3d = np.empty_like(points3d)
        for source, target in MIRROR_PERMUTATION.items():
            mirrored[target] = points[source] * flip
            mirrored3d[target] = points3d[source] * flip3d
        # sigma の定義が無い点は測定に使われないが、配列の形を保つために埋める。
        undefined = [i for i in range(len(points)) if i not in MIRROR_PERMUTATION]
        mirrored[undefined] = points[undefined] * flip
        mirrored3d[undefined] = points3d[undefined] * flip3d
        if injected:
            partners = [MIRROR_PERMUTATION[i] for i in injected]
            mirrored[partners] += (mirror_offset, 0.0)
            mirrored3d[partners] += (mirror_offset, 0.0, 0.0)

        pose = FacePose(
            yaw_deg=yaw_deg + float(rng.normal(0.0, 0.2)),
            pitch_deg=float(rng.normal(0.0, 0.2)),
            roll_deg=float(rng.normal(0.0, 0.2)),
        )
        detections.append(
            Detection(
                index=index,
                points=points,
                mirrored=mirrored,
                pose=pose,
                points3d=points3d,
                mirrored3d=mirrored3d,
            )
        )

    return Capture(
        detections=tuple(detections),
        frames_seen=frames,
        delta_seconds=delta_seconds,
        source_kind="synthetic",
        mode="synthetic",
        model_version="synthetic",
        model_sha256="",
        running_mode="—",
        detector_class="synthetic",
    )


@dataclass(frozen=True, slots=True)
class Check:
    """健全性確認 1 件。"""

    name: str
    passed: bool
    detail: str


def _mirror_measurements(capture: Capture) -> tuple[float, float]:
    """合成 capture から |d 差| の最大と、landmark バイアスの最大を取り出す。"""
    indices = list(SIGMA_INDICES)
    partners = [MIRROR_PERMUTATION[index] for index in indices]
    flip = np.array([-1.0, 1.0], dtype=np.float64)
    distance_gap = 0.0
    bias = 0.0
    for item in capture.detections:
        assert item.mirrored is not None
        left = distance_of(item.points)
        right = distance_of(item.mirrored)
        normal_original = normalized_of(item.points)
        normal_mirror = normalized_of(item.mirrored)
        if left is None or right is None or normal_original is None or normal_mirror is None:
            continue
        distance_gap = max(distance_gap, abs(left - right))
        remapped = normal_mirror[partners] * flip
        difference = np.linalg.norm(remapped - normal_original[indices], axis=1)
        bias = max(bias, float(difference.max()))
    return distance_gap, bias


def run_self_test() -> list[Check]:
    """実写が無くても、統計量の計算が正しいことを確かめる。

    合成の landmark を直接注入するので、答えが分かっている。閾値は「浮動小数の
    丸め」と「有限フレーム数の統計誤差」を分ける水準に置いてある。
    """
    checks: list[Check] = []

    # 1. 幾何学的に正しい検出器なら、鏡像テストの両方の統計量が 0 になる。
    exact = synthetic_capture(frames=8, sigma=0.0, yaw_deg=0.0, delta_seconds=0.2)
    gap, bias = _mirror_measurements(exact)
    checks.append(
        Check(
            name="鏡像・等価な検出器では差が出ない",
            passed=gap < 1e-9 and bias < 1e-9,
            detail=f"|d 差| 最大 {gap:.3e} / landmark バイアス最大 {bias:.3e}（許容 1e-9）",
        )
    )

    # 2. 既知の左右バイアスを注入すると、両方の統計量が期待どおり動く。
    offset = 0.01
    contour_pairs = len(CONFIG.parts["contour"].pairs)
    expected_gap = offset * sqrt(contour_pairs / len(CONFIG.all_pairs))
    injected = synthetic_capture(
        frames=4, sigma=0.0, yaw_deg=0.0, delta_seconds=0.2, mirror_offset=offset
    )
    gap, bias = _mirror_measurements(injected)
    checks.append(
        Check(
            name="鏡像・既知バイアスを検出できる",
            passed=abs(bias - offset) < 1e-9 and abs(gap - expected_gap) < 1e-9,
            detail=(
                f"landmark バイアス {bias:.6f}（期待 {offset:.6f}） / "
                f"|d 差| {gap:.6f}（期待 {expected_gap:.6f}）"
            ),
        )
    )

    # 3〜5. 既知のノイズを載せた系列で、ばらつきと平滑化の統計を確かめる。
    sigma = 0.02
    noisy = synthetic_capture(frames=240, sigma=sigma, yaw_deg=0.0, delta_seconds=0.2, seed=7)
    series = build_series(noisy)
    rms, _ = landmark_spread(series.normalized)
    measured = float(np.median(rms[list(MEASURED_INDICES)]))
    expected_rms = sigma * sqrt(2.0)
    checks.append(
        Check(
            name="注入したノイズを回復できる",
            passed=0.6 * expected_rms < measured < 1.6 * expected_rms,
            detail=(
                f"per-landmark RMS 中央値 {measured:.5f}（軸ごと sigma={sigma} の "
                f"2D RMS 期待 {expected_rms:.5f}、正規化で 4 自由度を失う分だけずれる）"
            ),
        )
    )

    tau = 1.0
    delta = noisy.delta_seconds
    alpha = 1.0 - exp(-delta / tau)
    theoretical = sqrt(alpha / (2.0 - alpha))
    smoothed = ema_array(series.normalized, series.times, tau_seconds=tau, max_gap_seconds=2.0)
    burn, _ = _burn_in(series.count, tau, delta)
    smoothed_rms, _ = landmark_spread(smoothed[burn:])
    ratio = float(np.median(smoothed_rms[list(MEASURED_INDICES)] / rms[list(MEASURED_INDICES)]))
    checks.append(
        Check(
            name="EMA の分散低減が理論どおり",
            passed=abs(ratio - theoretical) < 0.2 * theoretical,
            detail=(
                f"実測の標準偏差比 {ratio:.4f} / "
                f"理論 sqrt(alpha/(2-alpha)) {theoretical:.4f}（許容 20%）"
            ),
        )
    )

    raw_mean = float(series.distances.mean())
    coordinate_d = np.asarray(
        [value for value in (distance_of(frame) for frame in smoothed[burn:]) if value],
        dtype=np.float64,
    )
    scalar_d = ema_scalar(series.distances, series.times, tau_seconds=tau, max_gap_seconds=2.0)[
        burn:
    ]
    coordinate_mean = float(coordinate_d.mean())
    scalar_mean = float(scalar_d.mean())
    checks.append(
        Check(
            name="座標 EMA だけが d の平均を下げる",
            passed=coordinate_mean < 0.8 * raw_mean
            and abs(scalar_mean - raw_mean) < 0.1 * raw_mean,
            detail=(
                f"生値 {raw_mean:.5f} / 座標 EMA {coordinate_mean:.5f} / d を EMA {scalar_mean:.5f}"
            ),
        )
    )

    checks.append(_check_app_tau())
    checks.append(_check_mode_wiring())
    checks.extend(_check_object_symmetry_3d())
    checks.extend(_check_object_symmetry_2d())
    checks.append(_check_residual_matches_distance())
    checks.extend(_check_injection_gain())
    return checks


def _check_app_tau() -> Check:
    """tau_app の換算が、alpha を保つ換算になっていること。"""
    delta = 1.0 / 29.97
    alpha = 1.0 - exp(-delta / 2.0)
    converted = app_tau_for(alpha)
    round_trip = 1.0 - exp(-APP_DELTA_SECONDS / converted)
    return Check(
        name="tau_app が alpha を保つ",
        passed=abs(round_trip - alpha) < 1e-12 and abs(converted - 11.99) < 0.1,
        detail=(
            f"tau=2.0s @29.97fps → alpha={alpha:.5f} → tau_app={converted:.3f}s "
            f"（逆算 alpha={round_trip:.5f}）"
        ),
    )


def _check_mode_wiring() -> Check:
    """--mode が、実際に構築される MediaPipe の RunningMode まで届いていること。

    偽の detector factory を差し込むので、モデルも MediaPipe も要らない。ここが
    通っていれば、両モードの数値が一致したとしても原因は配線ではない。
    """

    class _Fake:
        def detect_for_video(self, image: Any, timestamp_ms: int) -> Any:
            raise AssertionError("self-test で検出は行わない")

        def detect(self, image: Any) -> Any:
            raise AssertionError("self-test で検出は行わない")

        def close(self) -> None:
            return None

    stand_in = Path(__file__).resolve()
    observed: dict[str, str] = {}
    for mode in sorted(RUNNING_MODE_BY_MODE):
        stream = LandmarkStream(
            model_path=stand_in,
            model_version="self-test",
            mode=mode,
            detector_factory=lambda _path: _Fake(),
        )
        observed[mode] = f"{stream.detector_class}/{stream.running_mode}"
        stream.close()
    expected = {
        "image": "MediaPipeImageLandmarker/IMAGE",
        "video": "MediaPipeFaceLandmarker/VIDEO",
    }
    return Check(
        name="--mode が RunningMode まで届く",
        passed=observed == expected,
        detail=" / ".join(f"{mode} → {value}" for mode, value in observed.items()),
    )


def _check_object_symmetry_3d() -> list[Check]:
    """テストD の検証。既知の yaw で d_3d が動かず、既知の z 誤差では動くこと。

    不変性だけを見ると「常に 0 を返す実装」でも通ってしまうので、感度の確認を
    対にしてある。
    """
    degrees = (0.0, 3.0, 5.0, 8.0, 15.0)
    solid: list[float] = []
    flat: list[float] = []
    for angle in degrees:
        rotated = np.asarray(FACE3D @ yaw_matrix(angle).T, dtype=np.float64)
        normalized = normalized_3d_of(rotated)
        value = None if normalized is None else distance_3d_of(normalized)
        solid.append(float("nan") if value is None else value)
        plane = distance_of(rotated[:, :2])
        flat.append(float("nan") if plane is None else plane)

    solid_array = np.asarray(solid, dtype=np.float64)
    flat_array = np.asarray(flat, dtype=np.float64)
    span = float(solid_array.max() - solid_array.min()) if np.isfinite(solid_array).all() else 1.0
    invariance = Check(
        name="3D: 既知の yaw を与えても d_3d が変わらない",
        passed=bool(np.isfinite(solid_array).all())
        and span < 1e-9
        and float(solid_array.max()) < 1e-9
        and float(flat_array[-1]) > float(flat_array[0]) + 0.01,
        detail=(
            f"yaw {degrees[0]:.0f}〜{degrees[-1]:.0f}°: d_3d の幅 {span:.3e}（許容 1e-9） / "
            f"同じ入力の 2D d は {flat_array[0]:.4f} → {flat_array[-1]:.4f} と増える"
        ),
    )

    # z にだけ既知の誤差を載せる。docs level3 §4 は 5% IOD で d≈0.068 と予測する。
    error = 0.05
    values: list[float] = []
    for trial in range(20):
        rng = np.random.default_rng(trial)
        noisy = FACE3D.copy()
        noisy[:, 2] += rng.normal(0.0, error, size=len(noisy))
        normalized = normalized_3d_of(noisy)
        value = None if normalized is None else distance_3d_of(normalized)
        if value is not None:
            values.append(value)
    mean = float(np.mean(values)) if values else float("nan")
    sensitivity = Check(
        name="3D: z の誤差には反応する",
        passed=bool(values) and 0.04 < mean < 0.10,
        detail=f"z へ {error:.0%} IOD の誤差 → d_3d 平均 {mean:.4f}（docs level3 §4 の予測 0.068）",
    )
    return [invariance, sensitivity]


def _check_object_symmetry_2d() -> list[Check]:
    """セクション I の方式② の検証。

    1. 左右非対称な合成 2D 顔に剛体変換（回転・平行移動）と一様スケールを
       与えても、②（§8.3 正規化 → 2D object symmetry）の値が変わらないこと。
       左右対称な顔では ② が 0 になり、0 の不変性は「常に 0 を返す実装」でも
       成り立ってしまうので、値が 0 でない顔で確かめる。
    2. 既知の左右差を注入すると、その大きさに比例して ② が増えること。

    不変性だけでは感度を保証できないので、2 つを対にしてある。テストD の 3D 側
    と同じ組み立てである。
    """
    placements: tuple[tuple[float, float, tuple[float, float]], ...] = (
        (0.0, 1.0, (0.0, 0.0)),
        (7.0, 320.0, (640.0, 360.0)),
        (-23.5, 95.0, (-40.0, 210.0)),
        (180.0, 640.0, (1920.0, 1080.0)),
    )
    tilted = _asymmetric_face_3d(0.02)
    values: list[float] = []
    for roll, scale, offset in placements:
        placed = _synthetic_pixel_face(roll_deg=roll, scale=scale, offset=offset, points=tilted)
        normalized = normalized_of(placed[:, :2])
        value = None if normalized is None else distance_2d_procrustes_of(normalized)
        if value is None:
            return [
                Check(
                    name="I: 方式② が剛体変換に対して不変",
                    passed=False,
                    detail=f"roll {roll}°, scale {scale} の配置で ② を計算できない",
                )
            ]
        values.append(value)
    span = max(values) - min(values)
    invariance = Check(
        name="I: 方式② が剛体変換に対して不変",
        passed=span < 1e-9 and values[0] > 1e-3,
        detail=(
            f"輪郭を 0.02 IOD ずらした非対称顔を roll {placements[0][0]:.0f}〜"
            f"{placements[-1][0]:.0f}°・scale {min(item[1] for item in placements):.0f}〜"
            f"{max(item[1] for item in placements):.0f}・平行移動ありの "
            f"{len(values)} 配置へ置いて ② の幅 {span:.3e}"
            f"（許容 1e-9、値 {values[0]:.6f} は 0 でないこと）"
        ),
    )

    # 輪郭ペアの片側だけを既知量ずらす。② は距離なので、小さい δ では被写体自身の
    # 非対称と打ち消し合いうる。合成顔は厳密に対称なので打ち消しは起きず、応答は
    # δ の 1 次で読める。
    reference = _synthetic_pixel_face()
    baseline = normalized_of(reference[:, :2])
    assert baseline is not None
    injected = part_injection_indices(baseline, "contour")
    weights = injection_weights(baseline, injected, sigma_iod=0.0)
    responses: list[float] = []
    for delta in (0.005, 0.010, 0.020):
        moved = distance_2d_procrustes_of(inject_field(baseline, weights, delta))
        responses.append(float("nan") if moved is None else moved / delta)
    finite = [value for value in responses if isfinite(value) and value > 0.0]
    linearity = (
        max(finite) / min(finite) if len(finite) == len(responses) and min(finite) > 0.0 else None
    )
    sensitivity = Check(
        name="I: 方式② が既知の左右差に比例して反応する",
        passed=linearity is not None and linearity < 1.05,
        detail=(
            f"δ = 0.005/0.010/0.020 IOD に対する ②÷δ は "
            f"{'/'.join(f'{value:.4f}' for value in responses)}"
            f"（最大÷最小 {'—' if linearity is None else f'{linearity:.4f}'}、許容 1.05）"
        ),
    )
    return [invariance, sensitivity]


def _check_residual_matches_distance() -> Check:
    """テストG の残差が、テストD の d_3d と同じ重ね合わせから出ていること。

    `procrustes_residual` は `symmetry_level3_probe.distance_3d` の手順を写した
    ものである。写し間違いがあると G だけが静かに別の量を報告するので、恒等式
    `d_3d = RMS(残差)` を毎回確かめる。
    """
    worst = 0.0
    for trial in range(8):
        rng = np.random.default_rng(trial)
        sample = FACE3D + rng.normal(0.0, 0.03, size=FACE3D.shape)
        residual = procrustes_residual(sample)
        reference = distance_3d(sample)
        if residual is None:
            return Check(
                name="G: 残差が d_3d と一致する", passed=False, detail="残差を計算できない"
            )
        rms = float(np.sqrt((residual**2).sum(axis=1).mean()))
        worst = max(worst, abs(rms - reference))
    return Check(
        name="G: 残差の RMS が d_3d と一致する",
        passed=worst < 1e-12,
        detail=f"8 通りの配置で最大差 {worst:.3e}（許容 1e-12）",
    )


def _synthetic_pixel_face(
    *,
    roll_deg: float = 7.0,
    scale: float = 320.0,
    offset: tuple[float, float] = (640.0, 360.0),
    points: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """3D 顔を画素相当の座標へ置いたもの。roll と平行移動を入れてある。

    §8.3 の正規化はこれらを取り除く。注入を正規化フレームで行い、画素へ戻して
    から測り直す経路が本当に往復できているかを、self-test で試すために要る。
    `points` を与えると FACE3D の代わりにその配置を置く。方式② の不変性を、
    値が 0 でない顔で確かめるために使う。
    """
    source = FACE3D if points is None else points
    theta = np.deg2rad(roll_deg)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    placed = (source * scale) @ rotation.T
    placed[:, 0] += offset[0]
    placed[:, 1] += offset[1]
    return np.asarray(placed, dtype=np.float64)


def _asymmetric_face_3d(offset_iod: float) -> NDArray[np.float64]:
    """FACE3D の輪郭ペアの片側だけを既知量ずらした、左右非対称な合成顔。

    FACE3D は厳密に左右対称なので、そのままでは ② も ③ も 0 になる。0 の不変性
    は「常に 0 を返す実装」でも成り立ってしまうため、不変性の確認には値が 0 で
    ない顔が要る。
    """
    face = FACE3D.copy()
    for left, right in CONFIG.parts["contour"].pairs:
        chosen = left if face[left][0] < face[right][0] else right
        face[chosen, 0] += offset_iod
    return face


def _check_injection_gain() -> list[Check]:
    """テストE のゲイン測定を、検出を通さずに検証する。

    完全なパイプライン（注入した変位をそのまま返す）でゲインが 1.0 になること
    と、変位を半分しか返さないパイプラインで 0.5 になることを対にしてある。
    前者だけでは「常に 1.0 を返す実装」を見逃す。
    """
    pixel_face = _synthetic_pixel_face()
    baseline_points = pixel_face[:, :2].copy()

    def perfect(
        _source: NDArray[np.float64],
        target: NDArray[np.float64],
        target3d: NDArray[np.float64] | None,
    ) -> ProbeResult:
        return ProbeResult(target, target3d, None)

    def halved(
        source: NDArray[np.float64],
        target: NDArray[np.float64],
        target3d: NDArray[np.float64] | None,
    ) -> ProbeResult:
        blended3d = None if target3d is None else pixel_face + 0.5 * (target3d - pixel_face)
        return ProbeResult(source + 0.5 * (target - source), blended3d, None)

    checks: list[Check] = []
    for label, probe, expected, tolerance, sigma in (
        ("1.0", perfect, 1.0, 1e-6, 0.0),
        ("0.5", halved, 0.5, 0.02, 0.0),
        ("1.0", perfect, 1.0, 1e-6, REGION_SIGMA_IOD),
    ):
        field = "点のみ" if sigma <= 0.0 else "領域"
        measured = measure_injection(
            baseline_points=baseline_points,
            baseline_points3d=pixel_face,
            deltas_mm=INJECTION_DELTAS_MM,
            probe=probe,
            field_name=field,
            sigma_iod=sigma,
        )
        if measured is None:
            checks.append(
                Check(name=f"E: ゲインが {label} になる", passed=False, detail="測定できない")
            )
            continue
        usable = [item for item in measured.outcomes if item.delta_mm > 0.0]
        gains_2d: list[float | None] = [item.gain_2d for item in usable]
        gains_3d: list[float | None] = [item.gain_3d for item in usable]
        realized: list[float | None] = [item.realized_mm / item.delta_mm for item in usable]
        values = [value for value in gains_2d + gains_3d + realized if value is not None]
        worst = max(abs(value - expected) for value in values) if values else float("inf")
        deltas = "/".join(f"{item.delta_mm:.0f}" for item in usable)
        checks.append(
            Check(
                name=(
                    f"E: 変位場「{field}」で、変位を {expected:.1f} 倍返す"
                    f"パイプラインのゲインが {label} になる"
                ),
                passed=len(values) == 3 * len(usable) and worst < tolerance,
                detail=(
                    f"δ={deltas}mm で 2D {_format_gains(gains_2d)} / "
                    f"3D {_format_gains(gains_3d)} / 実変位比 {_format_gains(realized)}"
                    f"（期待 {expected:.1f}、最大差 {worst:.2e}、許容 {tolerance:g}）"
                ),
            )
        )
    return checks


def _format_gains(values: Sequence[float | None]) -> str:
    return "[" + ", ".join("—" if value is None else f"{value:.4f}" for value in values) + "]"


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
SECTIONS: dict[str, str] = {
    "A": "鏡像テスト",
    "B": "再現性テスト",
    "C": "平滑化の予測",
    "D": "3D object symmetry",
    "E": "既知量注入（ゲイン）",
    "F": "3D 鏡像テスト",
    "G": "残差の xy / z 分解",
    "H": "3D の姿勢層別",
    "I": "3 方式の直接比較",
    "J": "部位別の注入ゲイン",
    "K": "3D 座標の平均",
}

#: 鏡像側の検出を必要とするセクション。要らないときは検出を半分に減らせる。
MIRROR_SECTIONS: frozenset[str] = frozenset({"A", "F", "I"})

#: 変形画像の再検出（注入測定）を必要とするセクション。
INJECTION_SECTIONS: frozenset[str] = frozenset({"I", "J"})


def build_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", type=Path, help="画像が入ったディレクトリ")
    source.add_argument("--video", type=Path, help="動画ファイル")
    source.add_argument(
        "--self-test",
        action="store_true",
        help="実写なしで、統計量の計算が正しいことを確認する",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="処理枚数の上限")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="フレームレート（--frames のとき必須の仮定。動画では取得失敗時のみ使う）",
    )
    parser.add_argument("--model", type=Path, default=None, help="landmark モデルの明示指定")
    parser.add_argument(
        "--mode",
        choices=("image", "video"),
        default="image",
        help="検出モード。image はフレーム間で独立、video は追跡を引き継ぐ",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=sorted(SECTIONS),
        help="実行するセクション（省略時は全部）",
    )
    parser.add_argument(
        "--gain-frames",
        type=int,
        default=1,
        help="テストE で注入するフレーム数（yaw 中央値に近い順）。既定は 1",
    )
    parser.add_argument("--json", type=Path, default=None, help="統計量の保存先")
    return parser.parse_args(argv)


def validate_arguments(arguments: argparse.Namespace) -> None:
    """入力の検証を、モデルを触る前にすべて済ませる。

    フレームの読み出しは generator なので、存在しないパスの診断は最初の 1 枚を
    要求するまで出ない。それでは MediaPipe の初期化ログを延々と流したあとで
    「見つかりません」が出ることになる。argparse で拾えない条件をここで先に
    落として、失敗を最初の 1 行にする。
    """
    if arguments.max_frames < 1:
        raise SystemExit("--max-frames は 1 以上である必要があります。")
    if arguments.gain_frames < 1:
        raise SystemExit("--gain-frames は 1 以上である必要があります。")
    if not isfinite(arguments.fps) or arguments.fps <= 0:
        raise SystemExit("--fps は正の有限値である必要があります。")
    if arguments.self_test:
        return
    if arguments.model is not None and not arguments.model.is_file():
        raise SystemExit(f"モデルが見つかりません: {arguments.model}")
    if arguments.video is not None:
        if not arguments.video.is_file():
            raise SystemExit(f"動画が見つかりません: {arguments.video}")
        return
    directory: Path = arguments.frames
    if not directory.is_dir():
        raise SystemExit(f"ディレクトリが見つかりません: {directory}")
    if not any(path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir()):
        raise SystemExit(f"画像が見つかりません: {directory}")


def _build_capture(arguments: argparse.Namespace, with_mirror: bool) -> Capture:
    if arguments.self_test:
        return synthetic_capture(
            frames=min(arguments.max_frames, 240),
            sigma=0.02,
            yaw_deg=1.0,
            delta_seconds=1.0 / arguments.fps,
            seed=11,
        )

    model_path, model_version = resolve_model(arguments.model)
    if arguments.video is not None:
        fps = video_fps(arguments.video, arguments.fps)
        frames = iter_video_frames(arguments.video, arguments.max_frames)
        source_kind = "video"
    else:
        fps = arguments.fps
        frames = iter_directory_frames(arguments.frames, arguments.max_frames)
        source_kind = "frames"

    print(f"検出中（モード {arguments.mode}、鏡像 {'あり' if with_mirror else 'なし'}）…")
    return capture_detections(
        frames,
        model_path=model_path,
        model_version=model_version,
        mode=arguments.mode,
        with_mirror=with_mirror,
        delta_seconds=1.0 / fps,
        source_kind=source_kind,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_arguments(argv)
    selected = arguments.section or sorted(SECTIONS)
    validate_arguments(arguments)
    validate_permutation(CONFIG, MIRROR_PERMUTATION)

    capture = _build_capture(arguments, with_mirror=bool(MIRROR_SECTIONS & set(selected)))
    series = build_series(capture)

    print(f"metric  : {CONFIG.metric_version}")
    print(f"landmark: {CONFIG.landmark_map_version}")
    print(f"model   : {capture.model_version}")
    print(f"入力    : {capture.source_kind} / {capture.frames_seen} 枚 / {capture.fps:.2f} fps")
    print(
        f"検出器  : --mode {capture.mode} → {capture.detector_class} "
        f"/ RunningMode.{capture.running_mode}"
    )
    print(f"IPD     : {IPD_MM:.0f}mm （両眼間距離比 × {IPD_MM:.0f} で mm 換算）")

    report: dict[str, Any] = {
        "schema": JSON_SCHEMA,
        "metric_version": CONFIG.metric_version,
        "landmark_map_version": CONFIG.landmark_map_version,
        "model_version": capture.model_version,
        "model_sha256": capture.model_sha256,
        "source_kind": capture.source_kind,
        "detection_mode": capture.mode,
        "detector_class": capture.detector_class,
        "running_mode": capture.running_mode,
        "ipd_mm": IPD_MM,
        "fps": capture.fps,
        "delta_seconds": capture.delta_seconds,
        "frames_seen": capture.frames_seen,
        "frames_detected": series.count,
    }

    if "A" in selected:
        section_a(capture, report)
    if "B" in selected:
        section_b(capture, series, report)
    if "C" in selected:
        section_c(capture, series, report)
    if "D" in selected:
        section_d(capture, series, report)
    if "E" in selected:
        section_e(arguments, capture, report)
    if "F" in selected:
        section_f(capture, report)
    if "G" in selected:
        section_g(series, report)
    if "H" in selected:
        section_h(series, report)

    injections: dict[str, InjectionReport] | None = None
    if INJECTION_SECTIONS & set(selected):
        print()
        print("[I/J] 部位ごとの注入測定（両セクションで共有するので 1 度だけ実行する）…")
        injections = collect_part_injections(arguments, capture)
    if "I" in selected:
        section_i(capture, series, injections, report)
    if "J" in selected:
        section_j(injections, report)
    if "K" in selected:
        section_k(capture, series, report)

    failures = 0
    if arguments.self_test:
        heading("S", "健全性確認 — 既知の合成データで統計量を検証する")
        for check in run_self_test():
            mark = "PASS" if check.passed else "FAIL"
            failures += 0 if check.passed else 1
            print(f"  [{mark}] {check.name}")
            print(f"         {check.detail}")
        print()
        print("  上のセクションは、この合成データ（既知のノイズ）を流した結果である。")

    if arguments.json is not None:
        # プライバシー: 画像も landmark 座標も書かない。統計量だけ（不変条件 1）。
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n統計量を書き出しました: {arguments.json}")

    print()
    print("=" * 100)
    print("解説: docs/symmetry-metric-analysis.md §7 / docs/symmetry-metric-level3.md §3")
    print("=" * 100)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
