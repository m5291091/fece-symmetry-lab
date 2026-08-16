#!/usr/bin/env python3
"""追跡ズーム経路の `z` スケール写像を実測する（docs §13）。

`docs/symmetry-metric-measurements.md` §1〜§12 の take01 / take02 は landmarker を
直接叩いて取得したもので、`TrackingZoomLandmarker` を通していない。ところが
`ui/camera.py` はリアルタイム経路の landmarker を常にこの wrapper で包むので、
利用者が実際に見る `z` はすべて `domain/tracking_zoom.remap_depths` を経由する。

    return tuple(depth * window.width for depth in depths)

この写像は `requirement.md` §8.1.1 の「`z` は画像の幅で正規化される」という
スケール規約から導いたものであり、実測で確かめられていない。本スクリプトはその
1 点だけを測る。

    W: 統制実験  1 枚のフレームを固定し、切り出し窓の大きさだけを変えて
                 `d_3d` / `z` / 対照の `d_2d` がどう動くかを見る
    R: 実写追認  take01 / take02 / take03 を zoom 経路と直叩きの両方へ通し、
                 窓幅の実変動・相関・表示値中央値のテイク間差を見る

いずれのセクションも `z_zoom ≈ alpha · z_direct + beta` を全 478 点まとめて最小二乗で
解き、**alpha が窓幅と相関するか**を主結論の材料として出す。生の `z` は頭部中心
付近を原点とする相対量なので、1 点ずつ比べても意味がない。

    uv run python scripts/measure_zoom_depth_scale.py --video data/takes/take01.mov
    uv run python scripts/measure_zoom_depth_scale.py --video data/takes/take01.mov \
        --video data/takes/take02.mov --video data/takes/take03.mov --max-frames 600
    uv run python scripts/measure_zoom_depth_scale.py --self-test

`measure_metric_noise.py`（A〜L）へは一切触れない。指標・表示式・正規化・3D 距離は
すべて同スクリプトと `domain` から import しており、この診断だけが持つ写しは
`WindowTracker` / `crop_box_of`（adapter が窓を公開しないための追跡）だけである。
その 2 つが adapter と一致することは、セクション W が実写で、self-test が stub で
毎回検査する。

`src/` は読むだけで一切変更しない。画像と landmark 座標は JSON へ書き出さず、
統計量だけを残す（CLAUDE.md §5 のプライバシー不変条件）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import MediaPipeFaceLandmarker
from face_symmetry_lab.adapters.inference.tracking_zoom_landmarker import TrackingZoomLandmarker
from face_symmetry_lab.application.ports import LandmarkProvider
from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    build_landmark_frame,
    to_pixel_array,
    to_pixel_array_3d,
)
from face_symmetry_lab.domain.tracking_zoom import (
    FULL_FRAME,
    BoundingBox,
    TrackingZoomPolicy,
    bounding_box_of,
    next_zoom_window,
    remap_depths,
    remap_points,
    resize_scale_for,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.measure_metric_noise import (  # noqa: E402
    CONFIG,
    CONFIG_3D,
    IPD_MM,
    distance_2d_procrustes_of,
    distance_of,
    heading,
    iter_video_frames,
    normalized_of,
    percent_value,
    resolve_model,
    symmetry_3d_of,
    video_fps,
)

#: セクション W が掃引する窓幅（フレーム比）。下 3 つは自動調整
#: （`domain/auto_tuning.recommend_zoom_policy` の margin_ratio 0.3〜1.2）が
#: 実際に作る帯で、1.00 は `recheck_interval_ms` ごとの全フレーム再確認そのもの
#: である。間は補間のために置いた。
WINDOW_TARGETS: tuple[float, ...] = (0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)

#: 統制実験を何枚のフレームで繰り返すか。1 枚では窓幅 8 点しか取れず、
#: 「alpha と窓幅の相関」を読む材料が薄い。各フレームは独立した反復であって、
#: フレームをまたいで平均するのは窓ごとの alpha だけである。
DEFAULT_SWEEP_FRAMES: int = 5

DEFAULT_MAX_FRAMES: int = 600
DEFAULT_FPS: float = 30.0

#: 実写追認で、窓幅を「全フレーム」とみなす下限。`_crop` の画素丸めで 1.0 から
#: わずかに外れることがあるため、厳密一致では数えられない。
FULL_FRAME_EPSILON: float = 1e-9

#: 統計の見積りが不安定になる下限。下回ったら数値を出さずに注意を出す。
MIN_USEFUL_FRAMES: int = 30

JSON_SCHEMA: str = "measure-zoom-depth-scale/1.0.0"

SECTIONS: dict[str, str] = {
    "W": "窓幅の統制実験（1 フレーム固定）",
    "R": "実写での追認（zoom 経路 vs 直叩き）",
}

#: 実写追認で並べる表示値の系列。テイク間の差もこの順で出す。
DISPLAY_KEYS: tuple[str, ...] = ("zoom_3d", "direct_3d", "zoom_2d", "direct_2d")


# --------------------------------------------------------------------------
# 窓の追跡 —— adapter が選んだ窓を、同じ public 関数で言い当てる
# --------------------------------------------------------------------------
def crop_box_of(window: BoundingBox, *, width_px: int, height_px: int) -> BoundingBox:
    """`TrackingZoomLandmarker` が実際に使う切り出し矩形（画素丸め後）。

    adapter は要求した窓をそのまま使わず、画素境界へ丸めた矩形を `remap_points`
    と `remap_depths` へ渡す。`remap_depths` が掛ける「窓の幅」はこの丸めた後の
    幅なので、測定側も同じ丸めを通さないと窓幅が 1 画素ぶんずれる。

    ここは adapter の `_crop` の座標計算だけを写したものである。写しが合っている
    ことは、セクション W が実写で（`replica_detect` と adapter の出力の厳密一致）、
    self-test が stub で毎回検査する。
    """
    x0 = round(window.x_min * width_px)
    y0 = round(window.y_min * height_px)
    x1 = min(width_px, max(x0 + 1, round(window.x_max * width_px)))
    y1 = min(height_px, max(y0 + 1, round(window.y_max * height_px)))
    return BoundingBox(x0 / width_px, y0 / height_px, x1 / width_px, y1 / height_px)


class WindowTracker:
    """`TrackingZoomLandmarker._select_window` と同じ規則で、使われた窓を追う。

    adapter は選んだ窓を公開しない。ここは同じ public 関数（`FULL_FRAME`、
    `next_zoom_window`、`bounding_box_of`）を adapter と同じ順序で呼ぶだけの写し
    である。`select` を呼んでから、その検出結果で `observe` する。

    写しの正しさは 2 通りに検証できる。1) 全フレーム窓と判定したフレームでは、
    zoom 経路の点が直叩きと厳密に一致するはずである（セクション R が数える）。
    2) 予測した窓で `replica_detect` を回すと adapter の出力と厳密に一致するはず
    である（セクション W が数える）。
    """

    def __init__(self, policy: TrackingZoomPolicy) -> None:
        self._policy = policy
        self._last_bbox: BoundingBox | None = None
        self._last_full_frame_check_ms: int | None = None

    def select(self, timestamp_ms: int) -> BoundingBox:
        if self._last_bbox is None or self._due_for_full_frame_recheck(timestamp_ms):
            self._last_full_frame_check_ms = timestamp_ms
            return FULL_FRAME
        return next_zoom_window(self._last_bbox, policy=self._policy)

    def observe(self, frame: LandmarkFrame) -> None:
        if not frame.has_single_face:
            self._last_bbox = None
            return
        self._last_bbox = bounding_box_of(frame.points)

    def _due_for_full_frame_recheck(self, timestamp_ms: int) -> bool:
        if self._last_full_frame_check_ms is None:
            return True
        return timestamp_ms - self._last_full_frame_check_ms >= self._policy.recheck_interval_ms


def replica_detect(
    inner: LandmarkProvider,
    rgb_image: NDArray[np.uint8],
    window: BoundingBox,
    *,
    frame_id: int,
    timestamp_ms: int,
    policy: TrackingZoomPolicy,
) -> LandmarkFrame:
    """指定した窓で、zoom 経路と同じ処理を通す（切り出し → 拡大 → 検出 → 逆写像）。

    adapter は窓を外から受け取れないので、統制実験のためにここで同じ手順を組む。
    切り出し矩形・拡大率・逆写像はすべて adapter が呼ぶのと同じ public 関数
    （`crop_box_of` は `_crop` の座標計算の写し、`resize_scale_for` /
    `remap_points` / `remap_depths` は domain そのもの）である。

    このスクリプトが adapter とずれていないことは、セクション W が同じ窓で
    adapter を走らせて出力を突き合わせることで確かめる。
    """
    height, width = rgb_image.shape[:2]
    box = crop_box_of(window, width_px=width, height_px=height)
    x0 = round(box.x_min * width)
    y0 = round(box.y_min * height)
    x1 = round(box.x_max * width)
    y1 = round(box.y_max * height)
    cropped: NDArray[np.uint8] = np.ascontiguousarray(rgb_image[y0:y1, x0:x1])

    scale = resize_scale_for(cropped.shape[1], cropped.shape[0], policy=policy)
    if scale > 1.0:
        new_size = (
            max(1, round(cropped.shape[1] * scale)),
            max(1, round(cropped.shape[0] * scale)),
        )
        cropped = np.asarray(
            Image.fromarray(cropped).resize(new_size, resample=Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    detected = inner.detect_video(cropped, frame_id=frame_id, timestamp_ms=timestamp_ms)
    if not detected.has_single_face:
        return build_landmark_frame(
            frame_id=detected.frame_id,
            face_count=detected.face_count,
            width=width,
            height=height,
        )
    return build_landmark_frame(
        frame_id=detected.frame_id,
        face_count=1,
        points=remap_points(detected.points, box),
        pose=detected.pose,
        width=width,
        height=height,
        depths=remap_depths(detected.depths, box),
    )


def sweep_policy(window_width: float) -> TrackingZoomPolicy:
    """窓幅だけを `window_width` へ固定する方針。

    `min_window_size` は窓の下限を決める policy 項目で、`margin_ratio` を 0 に
    すると窓は `max(bbox の幅, min_window_size)` になる。顔 bbox より広い値を
    与えれば窓幅はその値そのものになるので、被写体を動かさずに窓だけを動かせる。
    高さも同じ下限で押さえられるため、`min_window_size` が bbox の高さを下回る
    間は高さが動かない。窓の幅と高さの実測値は表へ併記する。

    再確認の間隔を実質無限にしてあるのは、2 枚目以降で全フレーム窓へ戻らない
    ようにするためである。窓の大きさ *だけ* を変えるのが本実験の条件である。
    """
    return TrackingZoomPolicy(
        margin_ratio=0.0,
        min_window_size=window_width,
        recheck_interval_ms=10**9,
        target_px=TrackingZoomPolicy().target_px,
        max_scale=TrackingZoomPolicy().max_scale,
    )


# --------------------------------------------------------------------------
# `z_zoom ≈ alpha · z_direct + beta` の最小二乗
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ScaleFit:
    """全 478 点をまとめて 1 つの (alpha, beta) へ当てた結果。単位はフレーム比。"""

    count: int
    alpha: float
    beta: float
    #: beta を 0 に固定したときの傾き。alpha と離れていれば beta が効いている。
    alpha_no_offset: float
    #: 残差の RMS。alpha と beta では説明できない部分。
    residual_rms: float
    residual_max: float
    #: 残差 RMS を `z_direct` の標準偏差で割った値。「説明できない割合」。
    residual_ratio: float


def fit_scale(direct: NDArray[np.float64], zoom: NDArray[np.float64]) -> ScaleFit | None:
    """`zoom ≈ alpha * direct + beta` を最小二乗で解く。

    `z` の原点は頭部中心付近の相対量なので、点ごとの比 `z_zoom / z_direct` には
    意味がない（分母が 0 を跨ぐ）。全点をまとめて 1 本の直線へ当てる。
    """
    x = np.asarray(direct, dtype=np.float64).ravel()
    y = np.asarray(zoom, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return None
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    spread = float(x.std())
    if spread <= 0.0:
        return None

    design = np.column_stack([x, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha = float(solution[0])
    beta = float(solution[1])
    residual = y - (alpha * x + beta)
    denominator = float(x @ x)
    return ScaleFit(
        count=int(x.size),
        alpha=alpha,
        beta=beta,
        alpha_no_offset=float(x @ y) / denominator if denominator > 0.0 else float("nan"),
        residual_rms=float(np.sqrt((residual**2).mean())),
        residual_max=float(np.abs(residual).max()),
        residual_ratio=float(np.sqrt((residual**2).mean()) / spread),
    )


def correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float | None:
    """Pearson の相関係数。定数列や 2 点未満では None。"""
    first = np.asarray(left, dtype=np.float64).ravel()
    second = np.asarray(right, dtype=np.float64).ravel()
    keep = np.isfinite(first) & np.isfinite(second)
    first, second = first[keep], second[keep]
    if first.size < 3 or first.std() <= 0.0 or second.std() <= 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


# --------------------------------------------------------------------------
# 1 フレーム分の測定値
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Measured:
    """1 つの検出結果から取れる、この診断が読む量すべて。"""

    #: 全フレーム比の 2D 画素座標 (N, 2)。
    pixels: NDArray[np.float64]
    #: 全フレーム比の 3D 画素座標 (N, 3)。z が欠測なら None。
    pixels3d: NDArray[np.float64] | None
    #: 逆写像後の `z`（フレーム比）。z が欠測なら None。
    depths: NDArray[np.float64] | None
    #: 現行 2D の総合距離 d。
    d_2d: float | None
    #: xy だけで解いた object symmetry（対照）。
    d_2dp: float | None
    #: §8.5.1 の 3D object symmetry。
    d_3d: float | None
    #: 3D の両眼 anchor 間距離（画素）。beta を IOD 比へ直すのに使う。
    iod_px: float | None


def _anchor_of(points3d: NDArray[np.float64], corners: tuple[int, int]) -> NDArray[np.float64]:
    return np.asarray(points3d[list(corners)].mean(axis=0), dtype=np.float64)


def measure_frame(frame: LandmarkFrame) -> Measured | None:
    """`LandmarkFrame` から、この診断が読む量をすべて取り出す。

    指標は `measure_metric_noise` の関数（=`domain` の実装）をそのまま呼ぶ。
    セクション L までと同じ量であることが、比較の前提だからである。
    """
    if not frame.has_single_face:
        return None
    try:
        pixels = to_pixel_array(frame)
    except LandmarkError:
        return None

    pixels3d: NDArray[np.float64] | None = None
    depths: NDArray[np.float64] | None = None
    iod_px: float | None = None
    if frame.has_depth:
        try:
            pixels3d = to_pixel_array_3d(frame)
        except LandmarkError:
            pixels3d = None
    if pixels3d is not None:
        depths = np.asarray(frame.depths, dtype=np.float64)
        left = _anchor_of(pixels3d, CONFIG_3D.left_eye_corners)
        right = _anchor_of(pixels3d, CONFIG_3D.right_eye_corners)
        distance = float(np.linalg.norm(right - left))
        iod_px = distance if distance > 0.0 else None

    normalized = normalized_of(pixels)
    return Measured(
        pixels=pixels,
        pixels3d=pixels3d,
        depths=depths,
        d_2d=distance_of(pixels),
        d_2dp=None if normalized is None else distance_2d_procrustes_of(normalized),
        d_3d=None if pixels3d is None else symmetry_3d_of(pixels3d),
        iod_px=iod_px,
    )


def _pt(distance: float | None) -> float | None:
    """距離を §8.6.3 の表示値（pt）へ。欠測は欠測のまま。"""
    return None if distance is None else percent_value(distance)


def _shift_pt(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None:
        return None
    return value - reference


def _dash(value: float | None, digits: int = 4) -> str:
    return "—" if value is None or not isfinite(value) else f"{value:.{digits}f}"


# --------------------------------------------------------------------------
# 素材の読み出し
# --------------------------------------------------------------------------
def sweep_frame_indices(count: int, limit: int) -> tuple[int, ...]:
    """統制実験に使うフレームの添字。先頭 `limit` 枚へ等間隔に散らす。"""
    if count <= 1:
        return (0,)
    return tuple(int(value) for value in np.unique(np.linspace(0, limit - 1, count).round()))


def read_frames(path: Path, indices: tuple[int, ...]) -> dict[int, NDArray[np.uint8]]:
    """指定した添字のフレームだけをメモリへ取る。"""
    wanted = set(indices)
    picked: dict[int, NDArray[np.uint8]] = {}
    for index, image in enumerate(iter_video_frames(path, max(indices) + 1)):
        if index in wanted:
            picked[index] = image
    return picked


def frames_of(path: Path, limit: int) -> Iterator[NDArray[np.uint8]]:
    return iter_video_frames(path, limit)


# --------------------------------------------------------------------------
# セクション W —— 窓の大きさだけを変える統制実験
# --------------------------------------------------------------------------
CONTROLLED_NOTE = """
  実写では「被写体が動いた」と「窓が変わった」が同時に起きるので分離できない。
  ここでは 1 枚のフレームを固定し、切り出し窓の大きさだけを人工的に変える。

  対照について

    切り出しは検出器への入力画像そのものを変えるので、`z` の写像が正しくても
    検出結果は動く。そこで同じ窓で `d_2d`（現行 2D）と `d_2dp`（xy だけで解いた
    object symmetry）も測る。この 2 つは `z` を一切参照しないので、その窓依存は
    まるごと「検出が変わったこと」による。`d_3d` の窓依存からこれを引いた残りが
    `z` 由来の候補である。

  窓幅 1.00 の行は全フレーム検出そのもの（`recheck_interval_ms` ごとにアプリが
  必ず通る経路）であり、直叩きと同一の入力になる。したがって alpha = 1、beta = 0、
  残差 0 になるはずで、この行は測定系の恒等性の確認を兼ねる。

  検出器の状態について（この設計にした理由）

    MediaPipe の VIDEO モードは、同じ大きさのフレームを流し続ける限り状態を
    持たない（§1 の測定で確認済み）。しかし 1 枚目に全フレーム、2 枚目に切り出し
    を与えると、直前の顔から作った ROI が残り、同じ切り出しでも新品の検出器と
    違う結果（顔 2 個を含む）を返すことがある。統制実験でこれが混ざると、窓幅の
    効果と検出器の状態の効果を分けられない。

    そこで本セクションは **窓ごとに新品の検出器へ切り出しだけを与える**。手順・
    切り出し・逆写像は adapter とまったく同じ public 関数を使う（`replica_detect`）。
    実際の adapter（1 枚目に全フレームを通した後の状態）でも同じ窓を測り、
    両者の食い違いを 5) で数える。それが実経路で起きる状態依存の大きさである。
"""


@dataclass(frozen=True, slots=True)
class SweepRow:
    """統制実験の 1 行（1 フレーム × 1 窓幅）。"""

    frame_index: int
    target_width: float
    window_width: float
    window_height: float
    crop_width_px: int
    crop_height_px: int
    resize_scale: float
    detected: bool
    #: adapter（全フレーム検出で状態を持った後）の出力が、新品検出器の出力と
    #: 厳密に一致したか。窓の写しが合っていても、検出器の状態が違えばずれる。
    adapter_matches: bool | None
    #: adapter 側が返した顔の数。1 以外なら、実経路ではそのフレームが落ちる。
    adapter_face_count: int | None
    xy_shift_median: float | None
    xy_shift_max: float | None
    fit: ScaleFit | None
    #: 同じ最小二乗を x / y へ当てたもの（対照）。xy の逆写像は平行移動つきの
    #: 相似変換で、正しいことが構成から分かっている。したがって 1 からのずれは
    #: まるごと「検出が変わったこと」で、alpha の estimator の雑音床になる。
    fit_x: ScaleFit | None
    fit_y: ScaleFit | None
    beta_iod: float | None
    measured: Measured | None


def _sweep_one_frame(
    image: NDArray[np.uint8],
    frame_index: int,
    *,
    model_path: Path,
    model_version: str,
) -> tuple[Measured | None, list[SweepRow]]:
    """1 枚のフレームについて、窓幅を掃引する。

    基準（`z_direct`）は全フレーム窓を新品の検出器へ与えたもので、これは窓幅
    1.00 の行そのものである。各窓の測定も新品の検出器を使うので、行の間で違うのは
    切り出し窓の大きさだけになる。

    同じ窓を実 adapter でも測る（全フレーム検出で状態を持たせた後の 2 枚目）。
    こちらは「窓の写しが合っているか」と「検出器の状態がどれだけ効くか」を同時に
    見るための対照であり、表の数値には使わない。
    """
    rows: list[SweepRow] = []
    height, width = image.shape[:2]
    default_policy = TrackingZoomPolicy()

    reference_detector = MediaPipeFaceLandmarker(model_path=model_path, model_version=model_version)
    try:
        reference_frame = replica_detect(
            reference_detector,
            image,
            FULL_FRAME,
            frame_id=0,
            timestamp_ms=0,
            policy=default_policy,
        )
    finally:
        reference_detector.close()
    if not reference_frame.has_single_face:
        return None, rows
    baseline = measure_frame(reference_frame)
    bbox = bounding_box_of(reference_frame.points)
    if baseline is None or bbox is None:
        return baseline, rows

    for target in WINDOW_TARGETS:
        policy = sweep_policy(target)
        window = next_zoom_window(bbox, policy=policy)
        box = crop_box_of(window, width_px=width, height_px=height)
        crop_w = round(box.x_max * width) - round(box.x_min * width)
        crop_h = round(box.y_max * height) - round(box.y_min * height)
        scale = resize_scale_for(crop_w, crop_h, policy=policy)

        fresh = MediaPipeFaceLandmarker(model_path=model_path, model_version=model_version)
        primed = MediaPipeFaceLandmarker(model_path=model_path, model_version=model_version)
        try:
            windowed = replica_detect(
                fresh, image, window, frame_id=0, timestamp_ms=0, policy=policy
            )
            zoom = TrackingZoomLandmarker(primed, policy=policy)
            # 1 枚目は必ず全フレーム窓。ここで adapter の `_last_bbox` が埋まる。
            zoom.detect_video(image, frame_id=0, timestamp_ms=0)
            # 2 枚目が目的の窓。再確認間隔を跨がないので全フレームへは戻らない。
            adapter_frame = zoom.detect_video(image, frame_id=1, timestamp_ms=1)
        finally:
            fresh.close()
            primed.close()

        measured = measure_frame(windowed)
        fit: ScaleFit | None = None
        fit_x: ScaleFit | None = None
        fit_y: ScaleFit | None = None
        beta_iod: float | None = None
        xy_median: float | None = None
        xy_max: float | None = None
        if measured is not None:
            if baseline.depths is not None and measured.depths is not None:
                fit = fit_scale(baseline.depths, measured.depths)
            if fit is not None and measured.iod_px is not None:
                beta_iod = fit.beta * width / measured.iod_px
            fit_x = fit_scale(baseline.pixels[:, 0], measured.pixels[:, 0])
            fit_y = fit_scale(baseline.pixels[:, 1], measured.pixels[:, 1])
            shift = np.linalg.norm(measured.pixels - baseline.pixels, axis=1)
            reference = baseline.iod_px if baseline.iod_px else 1.0
            xy_median = float(np.median(shift) / reference)
            xy_max = float(shift.max() / reference)

        rows.append(
            SweepRow(
                frame_index=frame_index,
                target_width=target,
                window_width=box.width,
                window_height=box.height,
                crop_width_px=crop_w,
                crop_height_px=crop_h,
                resize_scale=scale,
                detected=measured is not None,
                adapter_matches=_frames_identical(adapter_frame, windowed),
                adapter_face_count=adapter_frame.face_count,
                xy_shift_median=xy_median,
                xy_shift_max=xy_max,
                fit=fit,
                fit_x=fit_x,
                fit_y=fit_y,
                beta_iod=beta_iod,
                measured=measured,
            )
        )
    return baseline, rows


def _frames_identical(left: LandmarkFrame, right: LandmarkFrame) -> bool:
    """2 つの検出結果が厳密に一致するか（点と奥行きの両方）。"""
    if left.has_single_face != right.has_single_face:
        return False
    if not left.has_single_face:
        return True
    if left.points != right.points:
        return False
    return left.depths == right.depths


def section_w(
    arguments: argparse.Namespace,
    report: dict[str, Any],
) -> None:
    heading("W", "窓の大きさだけを変える（統制実験）")
    print(CONTROLLED_NOTE)

    path: Path = arguments.video[0]
    model_path, model_version = resolve_model(arguments.model)
    indices = sweep_frame_indices(arguments.sweep_frames, arguments.max_frames)
    print(f"  素材: {path.name} / フレーム {list(indices)} / 窓幅 {list(WINDOW_TARGETS)}")
    images = read_frames(path, indices)

    all_rows: list[SweepRow] = []
    baselines: dict[int, Measured] = {}
    for index in indices:
        image = images.get(index)
        if image is None:
            print(f"  フレーム {index} を読めませんでした。")
            continue
        baseline, rows = _sweep_one_frame(
            image, index, model_path=model_path, model_version=model_version
        )
        if baseline is not None:
            baselines[index] = baseline
        all_rows.extend(rows)

    if not all_rows:
        print("  顔を検出できたフレームがありません。")
        report["controlled_sweep"] = {"rows": 0}
        return

    primary = indices[0]
    print()
    print(f"  1) 先頭フレーム（index {primary}）の掃引")
    _print_sweep_table([row for row in all_rows if row.frame_index == primary])

    print()
    print("  2) 窓幅ごとの alpha（フレームをまたいだ反復）")
    per_window = _print_alpha_by_window(all_rows)

    print()
    print("  3) 相関 —— 窓幅に対して何が動くか")
    correlations = _print_sweep_correlations(all_rows, baselines)

    print()
    print("  4) 窓の写しの検査 —— adapter と同じ窓・同じ切り出しを使えているか")
    single = [row for row in all_rows if row.adapter_face_count == 1 and row.detected]
    agreed = [row for row in single if row.adapter_matches]
    print(f"     adapter が単一顔を返した {len(single)} 行のうち、{len(agreed)} 行で厳密一致")
    if len(agreed) != len(single):
        print("     ** 一致しない行がある。`crop_box_of` の写しを疑うこと。")

    print()
    print("  5) 検出器の状態の影響 —— adapter（全フレーム検出の後）と新品の食い違い")
    state = _print_state_effect(all_rows)

    report["controlled_sweep"] = {
        "source": path.name,
        "frame_indices": list(indices),
        "window_targets": list(WINDOW_TARGETS),
        "rows": len(all_rows),
        "per_window": per_window,
        "correlations": correlations,
        "window_replication": {"single_face_rows": len(single), "identical": len(agreed)},
        "detector_state": state,
        "primary_frame": [_sweep_row_json(row) for row in all_rows if row.frame_index == primary],
    }


def _print_state_effect(rows: list[SweepRow]) -> dict[str, Any]:
    """同じ窓・同じ画像で、adapter と新品の検出器がどれだけ違うか。

    adapter は 1 枚目に全フレーム検出を通しているので VIDEO の状態を持つ。
    新品はその窓の切り出ししか見ていない。差はまるごと検出器の状態から来る。
    """
    print(
        f"     {'窓幅':>6} {'行数':>4} {'新品が単一顔':>12} {'adapterが単一顔':>15} {'厳密一致':>8}"
    )
    print("     " + "-" * 54)
    summary: list[dict[str, Any]] = []
    for target in WINDOW_TARGETS:
        group = [row for row in rows if row.target_width == target]
        if not group:
            continue
        fresh_ok = sum(1 for row in group if row.detected)
        adapter_ok = sum(1 for row in group if row.adapter_face_count == 1)
        identical = sum(1 for row in group if row.adapter_matches)
        print(
            f"     {target:>6.2f} {len(group):>4d} {fresh_ok:>12d} {adapter_ok:>15d} "
            f"{identical:>8d}"
        )
        summary.append(
            {
                "window_width": target,
                "rows": len(group),
                "fresh_single_face": fresh_ok,
                "adapter_single_face": adapter_ok,
                "identical": identical,
            }
        )
    multiple = [
        row for row in rows if row.adapter_face_count is not None and row.adapter_face_count > 1
    ]
    print()
    print(
        f"     adapter が 2 顔以上を返した行: {len(multiple)} / {len(rows)}"
        "（実経路ではそのフレームの分析が落ちる）"
    )
    return {"per_window": summary, "adapter_multi_face_rows": len(multiple), "rows": len(rows)}


def _print_sweep_table(rows: list[SweepRow]) -> None:
    print(
        f"     {'窓幅':>6} {'窓高':>6} {'切出px':>10} {'拡大':>5} "
        f"{'alpha':>9} {'beta':>11} {'beta(IOD)':>9} {'残差RMS':>10} "
        f"{'d_3d':>8} {'d_2d':>8} {'d_2dp':>8} {'xy中央':>8}"
    )
    print("     " + "-" * 118)
    for row in rows:
        fit = row.fit
        measured = row.measured
        print(
            f"     {row.target_width:>6.2f} {row.window_height:>6.3f} "
            f"{row.crop_width_px:>4d}x{row.crop_height_px:<5d} {row.resize_scale:>5.2f} "
            f"{_dash(None if fit is None else fit.alpha, 5):>9} "
            f"{_dash(None if fit is None else fit.beta, 7):>11} "
            f"{_dash(row.beta_iod, 5):>9} "
            f"{_dash(None if fit is None else fit.residual_rms, 7):>10} "
            f"{_dash(None if measured is None else measured.d_3d, 5):>8} "
            f"{_dash(None if measured is None else measured.d_2d, 5):>8} "
            f"{_dash(None if measured is None else measured.d_2dp, 5):>8} "
            f"{_dash(row.xy_shift_median, 5):>8}"
        )
    print()
    print(
        "     alpha / beta は `z_zoom ~ alpha*z_direct + beta` の最小二乗"
        "（全 478 点、単位はフレーム比）。"
    )
    print("     beta(IOD) は beta を 3D 両眼間距離で割ったもの。d は両眼間距離比。")
    print("     xy中央 は直叩きとの xy のずれの中央値（IOD 比）。検出が動いた量の目安。")


def _print_alpha_by_window(rows: list[SweepRow]) -> list[dict[str, Any]]:
    print(
        f"     {'窓幅':>6} {'枚数':>4} {'alpha中央':>9} {'alpha最小':>9} {'alpha最大':>9} "
        f"{'beta中央':>11} {'残差RMS中央':>12} {'残差/SD':>8} "
        f"{'beta(IOD)中央':>13} {'対照 x':>9} {'対照 y':>9}"
    )
    print("     " + "-" * 116)
    summary: list[dict[str, Any]] = []
    for target in WINDOW_TARGETS:
        group = [row for row in rows if row.target_width == target and row.fit is not None]
        fits = [row.fit for row in group if row.fit is not None]
        if not fits:
            continue
        alphas = np.asarray([fit.alpha for fit in fits], dtype=np.float64)
        betas = np.asarray([fit.beta for fit in fits], dtype=np.float64)
        residuals = np.asarray([fit.residual_rms for fit in fits], dtype=np.float64)
        ratios = np.asarray([fit.residual_ratio for fit in fits], dtype=np.float64)
        alpha_x = _median_alpha([row.fit_x for row in group])
        alpha_y = _median_alpha([row.fit_y for row in group])
        beta_iods = [row.beta_iod for row in group if row.beta_iod is not None]
        beta_iod = float(np.median(beta_iods)) if beta_iods else None
        beta_iod_worst = float(max(abs(value) for value in beta_iods)) if beta_iods else None
        print(
            f"     {target:>6.2f} {len(fits):>4d} {float(np.median(alphas)):>9.5f} "
            f"{float(alphas.min()):>9.5f} {float(alphas.max()):>9.5f} "
            f"{float(np.median(betas)):>11.7f} {float(np.median(residuals)):>12.7f} "
            f"{float(np.median(ratios)):>8.4f} {_dash(beta_iod, 6):>13} "
            f"{_dash(alpha_x, 5):>9} {_dash(alpha_y, 5):>9}"
        )
        summary.append(
            {
                "window_width": target,
                "frames": len(fits),
                "alpha_median": float(np.median(alphas)),
                "alpha_min": float(alphas.min()),
                "alpha_max": float(alphas.max()),
                "beta_median": float(np.median(betas)),
                "beta_iod_median": beta_iod,
                "beta_iod_abs_max": beta_iod_worst,
                "residual_rms_median": float(np.median(residuals)),
                "residual_ratio_median": float(np.median(ratios)),
                "alpha_x_median": alpha_x,
                "alpha_y_median": alpha_y,
            }
        )
    print()
    print("     対照 x / y は、同じ最小二乗を xy の逆写像へ当てたときの傾きの中央値。")
    print("     xy の写像は構成から正しいので、1 からのずれは検出が動いたぶんだけである。")
    print("     alpha の 1 からのずれをこれと比べれば、z 固有のずれがあるかを読める。")
    return summary


def _median_alpha(fits: list[ScaleFit | None]) -> float | None:
    values = [fit.alpha for fit in fits if fit is not None]
    return float(np.median(values)) if values else None


def _print_sweep_correlations(
    rows: list[SweepRow],
    baselines: dict[int, Measured],
) -> dict[str, Any]:
    """窓幅に対する各量の相関。`d_2d` / `d_2dp` が対照である。"""
    usable = [row for row in rows if row.measured is not None]
    widths = np.asarray([row.window_width for row in usable], dtype=np.float64)

    def column(getter: Any) -> NDArray[np.float64]:
        return np.asarray(
            [float("nan") if getter(row) is None else float(getter(row)) for row in usable],
            dtype=np.float64,
        )

    columns: dict[str, NDArray[np.float64]] = {
        "alpha": column(lambda row: None if row.fit is None else row.fit.alpha),
        "beta": column(lambda row: None if row.fit is None else row.fit.beta),
        "alpha_x": column(lambda row: None if row.fit_x is None else row.fit_x.alpha),
        "alpha_y": column(lambda row: None if row.fit_y is None else row.fit_y.alpha),
        "d_3d": column(lambda row: row.measured.d_3d if row.measured else None),
        "d_2d": column(lambda row: row.measured.d_2d if row.measured else None),
        "d_2dp": column(lambda row: row.measured.d_2dp if row.measured else None),
    }
    labels = {
        "alpha": "alpha（z の倍率）",
        "beta": "beta（z の平行移動）",
        "alpha_x": "対照 x の傾き",
        "alpha_y": "対照 y の傾き",
        "d_3d": "d_3d（z を使う）",
        "d_2d": "d_2d（対照・z 不使用）",
        "d_2dp": "d_2dp（対照・z 不使用）",
    }
    # 窓幅 1.00 は直叩きそのもの（恒等）なので、必ず alpha = 1 / 差 0 になる。
    # その行を含めた相関と、外した相関の両方を出す。含めると恒等点が相関を
    # 引き下げ、外すと標本が窓幅の狭い範囲に偏る。片方だけでは読めない。
    varying = widths < 1.0 - FULL_FRAME_EPSILON
    print(
        f"     {'量':<24} {'窓幅との相関':>12} {'恒等を除く':>10} "
        f"{'最小':>10} {'最大':>10} {'全幅':>10}"
    )
    print("     " + "-" * 82)
    result: dict[str, Any] = {}
    for key, values in columns.items():
        keep = np.isfinite(values)
        if not keep.any():
            continue
        span = float(values[keep].max() - values[keep].min())
        coefficient = correlation(widths[keep], values[keep])
        inner_keep = keep & varying
        inner = correlation(widths[inner_keep], values[inner_keep]) if inner_keep.any() else None
        print(
            f"     {labels[key]:<24} {_dash(coefficient, 4):>12} {_dash(inner, 4):>10} "
            f"{float(values[keep].min()):>10.5f} {float(values[keep].max()):>10.5f} "
            f"{span:>10.5f}"
        )
        result[key] = {
            "correlation_with_window_width": coefficient,
            "correlation_excluding_identity": inner,
            "min": float(values[keep].min()),
            "max": float(values[keep].max()),
            "span": span,
        }
    print()
    print("     基準（窓幅 1.00 = 直叩き）との差を pt で置くと:")
    for key in ("d_3d", "d_2d"):
        shifts: list[float] = []
        for row in usable:
            reference = baselines.get(row.frame_index)
            if reference is None or row.measured is None:
                continue
            value = _shift_pt(
                _pt(getattr(reference, key)),
                _pt(getattr(row.measured, key)),
            )
            if value is not None:
                shifts.append(value)
        if not shifts:
            continue
        array = np.asarray(shifts, dtype=np.float64)
        print(
            f"       {labels[key]:<24} 中央 {float(np.median(array)):+7.4f}pt   "
            f"最小 {float(array.min()):+7.4f}pt   最大 {float(array.max()):+7.4f}pt"
        )
        result.setdefault(key, {})["display_shift_pt"] = {
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return result


def _sweep_row_json(row: SweepRow) -> dict[str, Any]:
    fit = row.fit
    measured = row.measured
    return {
        "frame_index": row.frame_index,
        "target_width": row.target_width,
        "window_width": row.window_width,
        "window_height": row.window_height,
        "crop_px": [row.crop_width_px, row.crop_height_px],
        "resize_scale": row.resize_scale,
        "adapter_matches": row.adapter_matches,
        "adapter_face_count": row.adapter_face_count,
        "alpha": None if fit is None else fit.alpha,
        "alpha_x": None if row.fit_x is None else row.fit_x.alpha,
        "alpha_y": None if row.fit_y is None else row.fit_y.alpha,
        "alpha_no_offset": None if fit is None else fit.alpha_no_offset,
        "beta": None if fit is None else fit.beta,
        "beta_iod": row.beta_iod,
        "residual_rms": None if fit is None else fit.residual_rms,
        "residual_max": None if fit is None else fit.residual_max,
        "residual_ratio": None if fit is None else fit.residual_ratio,
        "d_3d": None if measured is None else measured.d_3d,
        "d_2d": None if measured is None else measured.d_2d,
        "d_2dp": None if measured is None else measured.d_2dp,
        "xy_shift_median_iod": row.xy_shift_median,
        "xy_shift_max_iod": row.xy_shift_max,
    }


# --------------------------------------------------------------------------
# セクション R —— 実写での追認
# --------------------------------------------------------------------------
REAL_NOTE = """
  take01 / take02 / take03 を 2 経路へ同時に通す。

    直叩き    `MediaPipeFaceLandmarker` へ生フレームをそのまま渡す（§1〜§12 と同じ）
    zoom 経路 `TrackingZoomLandmarker` で包む（`ui/camera.py` と同じ既定 policy）

  同じフレーム・同じ時刻・同じモデルなので、差は追跡ズームだけから来る。

  窓の追跡の検証について

    adapter は選んだ窓を公開しないので、`WindowTracker` が言い当てる。その言い当て
    が合っていることは、**包んだ検出器が実際に受け取った画像の大きさ**と突き合わせて
    確かめる（`_ShapeRecorder`）。検出器の決定性に依存しないので、セクション W が
    見つけた VIDEO 状態の影響を受けない。
"""


class _ShapeRecorder:
    """包んだ `LandmarkProvider` へ渡る画像の大きさを控えるだけの通し役。

    `TrackingZoomLandmarker` は選んだ窓を公開しない。窓の言い当てが合っている
    ことを検出器の決定性に頼らず確かめるため、切り出し画像の shape をここで
    記録する。検出結果には一切触れない。
    """

    def __init__(self, inner: LandmarkProvider) -> None:
        self._inner = inner
        self.last_shape: tuple[int, int] | None = None

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.last_shape = (int(rgb_image.shape[0]), int(rgb_image.shape[1]))
        return self._inner.detect_video(rgb_image, frame_id=frame_id, timestamp_ms=timestamp_ms)

    def close(self) -> None:
        self._inner.close()


def expected_crop_shape(
    window: BoundingBox,
    *,
    width_px: int,
    height_px: int,
    policy: TrackingZoomPolicy,
) -> tuple[int, int]:
    """予測した窓で adapter が作るはずの切り出し画像の大きさ `(高さ, 幅)`。"""
    box = crop_box_of(window, width_px=width_px, height_px=height_px)
    crop_w = round(box.x_max * width_px) - round(box.x_min * width_px)
    crop_h = round(box.y_max * height_px) - round(box.y_min * height_px)
    scale = resize_scale_for(crop_w, crop_h, policy=policy)
    if scale <= 1.0:
        return (crop_h, crop_w)
    return (max(1, round(crop_h * scale)), max(1, round(crop_w * scale)))


@dataclass(frozen=True, slots=True)
class RealFrame:
    """実写 1 フレームの、2 経路ぶんの測定値。"""

    index: int
    window_width: float
    window_height: float
    full_frame: bool
    identical: bool
    shape_matches: bool
    direct_faces: int
    zoom_faces: int
    direct: Measured | None
    zoomed: Measured | None
    fit: ScaleFit | None
    #: セクション W と同じ xy の対照。1 からのずれが検出の動いたぶんである。
    fit_x: ScaleFit | None
    fit_y: ScaleFit | None


def _capture_real(
    path: Path,
    *,
    model_path: Path,
    model_version: str,
    limit: int,
    delta_seconds: float,
    stride: int,
) -> tuple[list[RealFrame], int]:
    """2 経路で 1 本のテイクを通す。

    `stride` は解析へ渡すフレームの間引き幅である。アプリは撮影 fps ではなく
    §14.3d の解析周期（5Hz）で検出器を呼ぶので、包んだ検出器が「前回」として
    覚えている顔は 1/30 秒前ではなく 0.2 秒前のものになる。時刻は間引きの前後で
    変えない（元の撮影時刻を渡す）ので、`recheck_interval_ms` の効き方も実際の
    アプリと同じになる。
    """
    direct_provider = MediaPipeFaceLandmarker(model_path=model_path, model_version=model_version)
    inner = MediaPipeFaceLandmarker(model_path=model_path, model_version=model_version)
    recorder = _ShapeRecorder(inner)
    policy = TrackingZoomPolicy()
    zoom = TrackingZoomLandmarker(recorder, policy=policy)
    tracker = WindowTracker(policy)

    collected: list[RealFrame] = []
    seen = 0
    try:
        for index, image in enumerate(frames_of(path, limit)):
            if index % stride:
                continue
            seen += 1
            height, width = image.shape[:2]
            timestamp_ms = max(index, round(index * delta_seconds * 1000.0))

            window = tracker.select(timestamp_ms)
            box = crop_box_of(window, width_px=width, height_px=height)
            expected_shape = expected_crop_shape(
                window, width_px=width, height_px=height, policy=policy
            )
            direct_frame = direct_provider.detect_video(
                image, frame_id=index, timestamp_ms=timestamp_ms
            )
            zoom_frame = zoom.detect_video(image, frame_id=index, timestamp_ms=timestamp_ms)
            tracker.observe(zoom_frame)

            direct = measure_frame(direct_frame)
            zoomed = measure_frame(zoom_frame)
            fit = None
            fit_x = None
            fit_y = None
            if direct is not None and zoomed is not None:
                if direct.depths is not None and zoomed.depths is not None:
                    fit = fit_scale(direct.depths, zoomed.depths)
                fit_x = fit_scale(direct.pixels[:, 0], zoomed.pixels[:, 0])
                fit_y = fit_scale(direct.pixels[:, 1], zoomed.pixels[:, 1])
            collected.append(
                RealFrame(
                    index=index,
                    window_width=box.width,
                    window_height=box.height,
                    full_frame=box.width >= 1.0 - FULL_FRAME_EPSILON,
                    identical=_frames_identical(direct_frame, zoom_frame),
                    shape_matches=recorder.last_shape == expected_shape,
                    direct_faces=direct_frame.face_count,
                    zoom_faces=zoom_frame.face_count,
                    direct=direct,
                    zoomed=zoomed,
                    fit=fit,
                    fit_x=fit_x,
                    fit_y=fit_y,
                )
            )
            if seen % 100 == 0:
                print(f"    {seen} 枚処理", flush=True)
    finally:
        direct_provider.close()
        zoom.close()
    return collected, seen


def section_r(arguments: argparse.Namespace, report: dict[str, Any]) -> None:
    heading("R", "実写での追認（zoom 経路 vs 直叩き）")
    print(REAL_NOTE)

    model_path, model_version = resolve_model(arguments.model)
    stride: int = arguments.stride
    takes: dict[str, dict[str, Any]] = {}
    medians: dict[str, dict[str, float | None]] = {}

    for path in arguments.video:
        fps = video_fps(path, arguments.fps)
        print(
            f"  {path.name} を 2 経路で検出中（{arguments.max_frames} 枚上限、"
            f"{fps:.2f}fps、間引き 1/{stride} = 実効 {fps / stride:.2f}Hz）…"
        )
        frames, seen = _capture_real(
            path,
            model_path=model_path,
            model_version=model_version,
            limit=arguments.max_frames,
            delta_seconds=1.0 / fps,
            stride=stride,
        )
        summary = _print_take(path.name, frames, seen, fps)
        summary["stride"] = stride
        takes[path.name] = summary
        display: dict[str, float | None] = summary.get("display_pt", {})
        medians[path.name] = {key: display.get(key) for key in DISPLAY_KEYS}

    print()
    print("  テイク間の表示値中央値の差（§12.2 の 0.031pt が zoom 経路でも保たれるか）")
    pairs = _print_take_gaps(medians)

    report["real_footage"] = {"stride": stride, "takes": takes, "take_gaps": pairs}


def _print_take(name: str, frames: list[RealFrame], seen: int, fps: float) -> dict[str, Any]:
    detected = [item for item in frames if item.direct is not None and item.zoomed is not None]
    shape_ok = sum(1 for item in frames if item.shape_matches)
    direct_only = sum(1 for item in frames if item.direct_faces == 1 and item.zoom_faces != 1)
    zoom_multi = sum(1 for item in frames if item.zoom_faces > 1)
    print()
    print(f"  ▼ {name}  読み込み {seen} 枚 / 2 経路とも検出 {len(detected)} 枚")
    print(f"     窓の言い当て  切り出し画像の大きさが予測と一致 {shape_ok} / {len(frames)} 枚")
    if shape_ok != len(frames):
        print("     ** 一致しないフレームがある。`WindowTracker` の写しを疑うこと。")
    print(f"     顔数  直叩きだけが単一顔 {direct_only} 枚 / zoom 経路が 2 顔以上 {zoom_multi} 枚")
    if len(detected) < MIN_USEFUL_FRAMES:
        print(f"     {MIN_USEFUL_FRAMES} 枚に満たないので数値を出しません。")
        return {
            "frames_seen": seen,
            "frames_detected": len(detected),
            "usable": False,
            "shape_agreement": {"checked": len(frames), "matched": shape_ok},
            "face_count": {"direct_only_single": direct_only, "zoom_multi_face": zoom_multi},
        }

    widths = np.asarray([item.window_width for item in detected], dtype=np.float64)
    full = [item for item in detected if item.full_frame]
    identical = [item for item in full if item.identical]
    partial = widths[widths < 1.0 - FULL_FRAME_EPSILON]

    print(
        f"     窓幅  最小 {widths.min():.4f} / 中央 {np.median(widths):.4f} / "
        f"最大 {widths.max():.4f} / 全幅 {widths.max() - widths.min():.4f}"
    )
    if partial.size:
        print(
            f"           うち全フレーム窓を除く {partial.size} 枚: "
            f"最小 {partial.min():.4f} / 中央 {np.median(partial):.4f} / "
            f"最大 {partial.max():.4f} / 全幅 {partial.max() - partial.min():.4f}"
        )
    print(
        f"     全フレーム窓 {len(full)} 枚（{len(full) / len(detected) * 100:.1f}%）。"
        f"うち直叩きと厳密一致 {len(identical)} 枚"
        "（一致しない分は VIDEO 状態の差。セクション W の 5) を参照）"
    )

    fits = [item for item in detected if item.fit is not None]
    alphas = np.asarray([item.fit.alpha for item in fits if item.fit], dtype=np.float64)
    betas = np.asarray([item.fit.beta for item in fits if item.fit], dtype=np.float64)
    residuals = np.asarray([item.fit.residual_rms for item in fits if item.fit], dtype=np.float64)
    ratios = np.asarray([item.fit.residual_ratio for item in fits if item.fit], dtype=np.float64)
    fit_widths = np.asarray([item.window_width for item in fits], dtype=np.float64)

    print(
        f"     alpha     中央 {np.median(alphas):.5f} / 最小 {alphas.min():.5f} / "
        f"最大 {alphas.max():.5f}   （{alphas.size} 枚）"
    )
    print(
        f"     beta  中央 {np.median(betas):.7f} / 最小 {betas.min():.7f} / 最大 {betas.max():.7f}"
    )
    print(f"     残差  RMS 中央 {np.median(residuals):.7f} / 残差/SD 中央 {np.median(ratios):.4f}")
    control_x = _median_alpha([item.fit_x for item in detected])
    control_y = _median_alpha([item.fit_y for item in detected])
    print(
        f"     対照  xy 逆写像へ同じ最小二乗を当てた傾きの中央値  "
        f"x {_dash(control_x, 5)} / y {_dash(control_y, 5)}"
    )

    partial_mask = fit_widths < 1.0 - FULL_FRAME_EPSILON
    correlations = {
        "width_alpha": correlation(fit_widths, alphas),
        "width_alpha_partial": (
            correlation(fit_widths[partial_mask], alphas[partial_mask])
            if partial_mask.sum() >= 3
            else None
        ),
        "width_beta": correlation(fit_widths, betas),
    }
    zoom_3d = np.asarray(
        [
            item.zoomed.d_3d if item.zoomed and item.zoomed.d_3d is not None else np.nan
            for item in detected
        ],
        dtype=np.float64,
    )
    direct_3d = np.asarray(
        [
            item.direct.d_3d if item.direct and item.direct.d_3d is not None else np.nan
            for item in detected
        ],
        dtype=np.float64,
    )
    zoom_2d = np.asarray(
        [
            item.zoomed.d_2d if item.zoomed and item.zoomed.d_2d is not None else np.nan
            for item in detected
        ],
        dtype=np.float64,
    )
    direct_2d = np.asarray(
        [
            item.direct.d_2d if item.direct and item.direct.d_2d is not None else np.nan
            for item in detected
        ],
        dtype=np.float64,
    )
    correlations["width_d3d_zoom"] = correlation(widths, zoom_3d)
    correlations["width_d3d_direct"] = correlation(widths, direct_3d)
    correlations["width_d2d_zoom"] = correlation(widths, zoom_2d)

    print(
        f"     相関  窓幅×alpha {_dash(correlations['width_alpha'], 4)}"
        f"（全フレーム窓を除くと {_dash(correlations['width_alpha_partial'], 4)}）"
    )
    print(f"           窓幅×beta {_dash(correlations['width_beta'], 4)}")
    print(
        f"           窓幅×d_3d(zoom) {_dash(correlations['width_d3d_zoom'], 4)}   "
        f"窓幅×d_3d(直叩き) {_dash(correlations['width_d3d_direct'], 4)}   "
        f"窓幅×d_2d(zoom) {_dash(correlations['width_d2d_zoom'], 4)}"
    )

    display = {
        "zoom_3d": _median_pt(zoom_3d),
        "direct_3d": _median_pt(direct_3d),
        "zoom_2d": _median_pt(zoom_2d),
        "direct_2d": _median_pt(direct_2d),
    }
    print(
        f"     表示値中央値(pt)  3D zoom {_dash(display['zoom_3d'], 3)} / "
        f"3D 直叩き {_dash(display['direct_3d'], 3)}   "
        f"2D zoom {_dash(display['zoom_2d'], 3)} / 2D 直叩き {_dash(display['direct_2d'], 3)}"
    )
    return {
        "frames_seen": seen,
        "frames_detected": len(detected),
        "usable": True,
        "fps": fps,
        "shape_agreement": {"checked": len(frames), "matched": shape_ok},
        "face_count": {"direct_only_single": direct_only, "zoom_multi_face": zoom_multi},
        "window_width": {
            "min": float(widths.min()),
            "median": float(np.median(widths)),
            "max": float(widths.max()),
            "full_frame_count": len(full),
            "full_frame_identical": len(identical),
            "partial_min": float(partial.min()) if partial.size else None,
            "partial_median": float(np.median(partial)) if partial.size else None,
            "partial_max": float(partial.max()) if partial.size else None,
        },
        "alpha": {
            "count": int(alphas.size),
            "median": float(np.median(alphas)),
            "min": float(alphas.min()),
            "max": float(alphas.max()),
        },
        "beta": {
            "median": float(np.median(betas)),
            "min": float(betas.min()),
            "max": float(betas.max()),
        },
        "control_xy": {"alpha_x_median": control_x, "alpha_y_median": control_y},
        "residual": {
            "rms_median": float(np.median(residuals)),
            "ratio_median": float(np.median(ratios)),
        },
        "correlations": correlations,
        "display_pt": display,
    }


def _median_pt(distances: NDArray[np.float64]) -> float | None:
    values = distances[np.isfinite(distances)]
    if values.size == 0:
        return None
    return float(np.median(np.clip((1.0 - values) * 100.0, 0.0, 100.0)))


def _print_take_gaps(medians: dict[str, dict[str, float | None]]) -> list[dict[str, Any]]:
    names = list(medians)
    if len(names) < 2:
        print("     テイクが 1 つなので差は出せません。")
        return []
    print(f"     {'比較':<28} {'3D zoom':>10} {'3D 直叩き':>10} {'2D zoom':>10} {'2D 直叩き':>10}")
    print("     " + "-" * 74)
    pairs: list[dict[str, Any]] = []
    for first, second in (
        (names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
    ):
        row = {
            key: _difference(medians[first].get(key), medians[second].get(key))
            for key in DISPLAY_KEYS
        }
        print(
            f"     {first} - {second:<12} {_dash(row['zoom_3d'], 3):>10} "
            f"{_dash(row['direct_3d'], 3):>10} {_dash(row['zoom_2d'], 3):>10} "
            f"{_dash(row['direct_2d'], 3):>10}"
        )
        pairs.append({"left": first, "right": second, "difference_pt": row})
    print()
    print("     単位は pt。§12.2 の記録は 3D -0.031pt / 現行 2D -4.080pt（直叩き・IMAGE モード）。")
    return pairs


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


# --------------------------------------------------------------------------
# 実写なしの健全性確認
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str


class _StubLandmarker:
    """切り出し画像の形と内容を控えるだけの `LandmarkProvider`。

    返す正規化座標は固定なので、adapter と `replica_detect` の出力が違えば、
    それは逆写像へ渡した窓が違うことを意味する。切り出し画素そのものの一致は
    記録した `shape` と画素和で確かめる。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.calls.append((tuple(rgb_image.shape), int(rgb_image.sum())))
        height, width = rgb_image.shape[:2]
        points = ((0.30, 0.40), (0.55, 0.42), (0.45, 0.70), (0.62, 0.66))
        depths = (-0.020, 0.015, 0.004, -0.011)
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=1,
            points=points,
            width=width,
            height=height,
            depths=depths,
        )

    def close(self) -> None:
        return None


def _synthetic_frame(width: int = 1280, height: int = 720) -> NDArray[np.uint8]:
    """画素位置が値へ現れる勾配画像。切り出し位置がずれれば内容も変わる。"""
    xs = np.linspace(0, 255, width, dtype=np.float64)
    ys = np.linspace(0, 255, height, dtype=np.float64)
    field = (xs[None, :] * 0.5 + ys[:, None] * 0.5).astype(np.uint8)
    return np.ascontiguousarray(np.dstack([field, field[:, ::-1], field[::-1, :]]))


def _check_fit_scale() -> list[Check]:
    rng = np.random.default_rng(7)
    direct = rng.normal(0.0, 0.03, size=478)
    exact = fit_scale(direct, 0.37 * direct + 0.004)
    checks = [
        Check(
            "fit_scale が既知の (alpha, beta) を厳密に取り戻す",
            exact is not None
            and abs(exact.alpha - 0.37) < 1e-12
            and abs(exact.beta - 0.004) < 1e-12
            and exact.residual_rms < 1e-15,
            "—"
            if exact is None
            else f"alpha={exact.alpha:.12f} beta={exact.beta:.12f} 残差={exact.residual_rms:.3e}",
        )
    ]
    noisy = fit_scale(direct, 1.0 * direct + rng.normal(0.0, 0.002, size=478))
    checks.append(
        Check(
            "雑音を足すと残差が立ち、alpha は 1 の近くに残る",
            noisy is not None and abs(noisy.alpha - 1.0) < 0.05 and noisy.residual_rms > 1e-4,
            "—" if noisy is None else f"alpha={noisy.alpha:.5f} 残差RMS={noisy.residual_rms:.6f}",
        )
    )
    identical = fit_scale(direct, direct)
    checks.append(
        Check(
            "同一入力では alpha=1 / beta=0 / 残差 0",
            identical is not None
            and abs(identical.alpha - 1.0) < 1e-12
            and abs(identical.beta) < 1e-12
            and identical.residual_rms < 1e-15,
            "—" if identical is None else f"alpha={identical.alpha:.12f} beta={identical.beta:.3e}",
        )
    )
    return checks


def _check_sweep_policy() -> list[Check]:
    bbox = BoundingBox(0.40, 0.30, 0.57, 0.63)
    rows: list[str] = []
    ok = True
    for target in WINDOW_TARGETS:
        window = next_zoom_window(bbox, policy=sweep_policy(target))
        expected_width = min(max(bbox.width, target), 1.0)
        expected_height = min(max(bbox.height, target), 1.0)
        if abs(window.width - expected_width) > 1e-12:
            ok = False
        if abs(window.height - expected_height) > 1e-12:
            ok = False
        rows.append(f"{target:.2f}→{window.width:.3f}")
    return [
        Check(
            "sweep_policy が窓幅を狙いどおりに固定する",
            ok,
            " ".join(rows),
        )
    ]


def _check_replica_matches_adapter() -> list[Check]:
    """stub 検出器で、adapter と `replica_detect` が同じ窓・同じ切り出しを使うか。"""
    image = _synthetic_frame()
    mismatched: list[str] = []
    crop_mismatched: list[str] = []
    for target in WINDOW_TARGETS:
        policy = sweep_policy(target)
        adapter_stub = _StubLandmarker()
        replica_stub = _StubLandmarker()
        zoom = TrackingZoomLandmarker(adapter_stub, policy=policy)
        first = zoom.detect_video(image, frame_id=0, timestamp_ms=0)
        bbox = bounding_box_of(first.points)
        assert bbox is not None
        window = next_zoom_window(bbox, policy=policy)
        actual = zoom.detect_video(image, frame_id=1, timestamp_ms=1)
        expected = replica_detect(
            replica_stub, image, window, frame_id=1, timestamp_ms=1, policy=policy
        )
        if not _frames_identical(actual, expected):
            mismatched.append(f"{target:.2f}")
        if adapter_stub.calls[-1] != replica_stub.calls[-1]:
            crop_mismatched.append(f"{target:.2f}")
    checks = [
        Check(
            "adapter と replica_detect の出力が厳密一致する",
            not mismatched,
            "全窓幅で一致" if not mismatched else f"不一致: {', '.join(mismatched)}",
        ),
        Check(
            "adapter と replica_detect が同一の切り出し画素を渡す",
            not crop_mismatched,
            "全窓幅で一致" if not crop_mismatched else f"不一致: {', '.join(crop_mismatched)}",
        ),
    ]
    return checks


def _check_window_tracker() -> list[Check]:
    """`WindowTracker` が adapter の窓選択と同じ順序で全フレームへ戻るか。"""
    image = _synthetic_frame()
    policy = TrackingZoomPolicy()
    stub = _StubLandmarker()
    zoom = TrackingZoomLandmarker(stub, policy=policy)
    tracker = WindowTracker(policy)

    predicted_full: list[int] = []
    observed_full: list[int] = []
    for index in range(90):
        timestamp_ms = index * 33
        window = tracker.select(timestamp_ms)
        before = len(stub.calls)
        frame = zoom.detect_video(image, frame_id=index, timestamp_ms=timestamp_ms)
        tracker.observe(frame)
        shape = stub.calls[before][0]
        if window.width >= 1.0 - FULL_FRAME_EPSILON:
            predicted_full.append(index)
        # 全フレーム窓なら stub が受け取る画像は元フレームそのもの。
        if shape[:2] == image.shape[:2]:
            observed_full.append(index)
    return [
        Check(
            "WindowTracker の全フレーム窓の予測が adapter と一致する",
            predicted_full == observed_full,
            f"予測 {predicted_full[:6]}… 実際 {observed_full[:6]}…",
        )
    ]


def _check_display_scale() -> list[Check]:
    """表示値の読み替えが `measure_metric_noise` と同一であること。"""
    samples = (0.0, 0.0172, 0.0741, 0.5, 1.2)
    ok = all(
        abs(percent_value(value) - min(100.0, max(0.0, (1.0 - value) * 100.0))) < 1e-12
        for value in samples
    )
    return [
        Check(
            "表示値は §8.6.3 の式そのもの（A〜L と同じ関数を呼ぶ）",
            ok,
            f"d={samples} → " + ", ".join(f"{percent_value(value):.3f}" for value in samples),
        )
    ]


def run_self_test() -> list[Check]:
    checks: list[Check] = []
    checks.extend(_check_fit_scale())
    checks.extend(_check_sweep_policy())
    checks.extend(_check_replica_matches_adapter())
    checks.extend(_check_window_tracker())
    checks.extend(_check_display_scale())
    return checks


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def build_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=None,
        help="動画ファイル（複数指定可。セクション W は最初の 1 本を使う）",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="実写なしで、窓の写しと最小二乗が正しいことを確認する",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="処理枚数の上限")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="fps 取得失敗時の仮定")
    parser.add_argument("--model", type=Path, default=None, help="landmark モデルの明示指定")
    parser.add_argument(
        "--sweep-frames",
        type=int,
        default=DEFAULT_SWEEP_FRAMES,
        help="セクション W が掃引を繰り返すフレーム数（等間隔に選ぶ）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="セクション R が解析へ渡すフレームの間引き幅。アプリの解析 5Hz は撮影 30fps で 6",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=sorted(SECTIONS),
        help="実行するセクション（省略時は全部）",
    )
    parser.add_argument("--json", type=Path, default=None, help="統計量の保存先")
    return parser.parse_args(argv)


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_frames < 1:
        raise SystemExit("--max-frames は 1 以上である必要があります。")
    if arguments.sweep_frames < 1:
        raise SystemExit("--sweep-frames は 1 以上である必要があります。")
    if arguments.stride < 1:
        raise SystemExit("--stride は 1 以上である必要があります。")
    if not isfinite(arguments.fps) or arguments.fps <= 0:
        raise SystemExit("--fps は正の有限値である必要があります。")
    if arguments.model is not None and not arguments.model.is_file():
        raise SystemExit(f"モデルが見つかりません: {arguments.model}")
    if arguments.self_test:
        return
    if not arguments.video:
        raise SystemExit("--video か --self-test のどちらかが必要です。")
    for path in arguments.video:
        if not path.is_file():
            raise SystemExit(f"動画が見つかりません: {path}")


def main(argv: list[str] | None = None) -> int:
    arguments = build_arguments(argv)
    validate_arguments(arguments)
    selected = arguments.section or sorted(SECTIONS)

    print(f"metric  : {CONFIG.metric_version} / {CONFIG_3D.metric_version}")
    print(f"landmark: {CONFIG.landmark_map_version}")
    print(f"IPD     : {IPD_MM:.0f}mm （両眼間距離比 × {IPD_MM:.0f} で mm 換算）")
    report: dict[str, Any] = {
        "schema": JSON_SCHEMA,
        "metric_version": CONFIG.metric_version,
        "metric_version_3d": CONFIG_3D.metric_version,
        "landmark_map_version": CONFIG.landmark_map_version,
        "ipd_mm": IPD_MM,
        "max_frames": arguments.max_frames,
        "zoom_policy": {
            "margin_ratio": TrackingZoomPolicy().margin_ratio,
            "min_window_size": TrackingZoomPolicy().min_window_size,
            "recheck_interval_ms": TrackingZoomPolicy().recheck_interval_ms,
            "target_px": TrackingZoomPolicy().target_px,
            "max_scale": TrackingZoomPolicy().max_scale,
        },
    }

    if not arguments.self_test:
        _, model_version = resolve_model(arguments.model)
        report["model_version"] = model_version
        print(f"model   : {model_version}")
        print(f"入力    : {', '.join(path.name for path in arguments.video)}")
        if "W" in selected:
            section_w(arguments, report)
        if "R" in selected:
            section_r(arguments, report)

    failures = 0
    if arguments.self_test:
        heading("S", "健全性確認 — 実写なしで、窓の写しと最小二乗を検証する")
        for check in run_self_test():
            mark = "PASS" if check.passed else "FAIL"
            failures += 0 if check.passed else 1
            print(f"  [{mark}] {check.name}")
            print(f"         {check.detail}")

    if arguments.json is not None:
        # プライバシー: 画像も landmark 座標も書かない。統計量だけ（不変条件 1）。
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n統計量を書き出しました: {arguments.json}")

    print()
    print("=" * 100)
    print("解説: docs/symmetry-metric-measurements.md §13 / requirement.md §8.1.1")
    print("=" * 100)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
