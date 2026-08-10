#!/usr/bin/env python3
"""セッションを跨いだ表示値のばらつきと、その共変量を実測する（docs §14）。

`docs/symmetry-metric-measurements.md` §12.2 は take01 と take02 だけで
「3D の表示値中央値はテイクを変えても 0.031pt しか動かない」と記録した。§13.5 で
take03 を足すと、3 本の中央値は 98.252 / 98.233 / 98.530 に散り、テイク間の幅は
0.28pt になる。**2 本と 3 本で 10 倍違う。**

本スクリプトは N 本のテイクを一度に読み、次の 3 つを同じ実行の中で出す。

    P: 撮影条件のプロファイル  顔の大きさ・姿勢・明るさ・検出・尺を、テイクごとに並べる
    V: セッション間ばらつき    表示値の分布と、テイク間の中央値の散らばり（3D / 2D）
    G: 信号 ÷ ばらつき         §12.5 と同じ注入を各テイクで再実行し、V の散らばりと比べる

P が要るのは、V で見えた差が「その日の顔の違い」なのか「撮り方の違い」なのかを
分けるためである。撮り方なら揃えれば消せる。**ただし本スクリプトは相関係数を
出さない。** テイクは 3 本しかなく、n=3 の相関には意味がないためである。出すのは
数字を並べた表と、テイク間の全幅だけである。

    uv run python scripts/measure_session_variation.py --video take01.mov \
        --video take02.mov --video take03.mov --max-frames 600 --gain-frames 5
    uv run python scripts/measure_session_variation.py --self-test

`measure_metric_noise.py`（A〜L）と `measure_zoom_depth_scale.py`（W / R）へは一切
触れない。指標・正規化・表示式・注入・表示統計はすべて前者から import しており、
本スクリプトが持つ独自の計算は「撮影条件の共変量」と「テイクを跨いだ散らばり」
だけである。したがって take01 / take02 の表示値は §12.1 の数値とそのまま一致する
はずで、セクション V がその一致を毎回確認できる形で出す。

`src/` は読むだけで一切変更しない。画像と landmark 座標は JSON へ書き出さず、
統計量だけを残す（CLAUDE.md §5 のプライバシー不変条件）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.frame_adjustments import mean_luminance
from face_symmetry_lab.domain.landmarks import LandmarkError, LandmarkFrame, to_pixel_array
from face_symmetry_lab.domain.tracking_zoom import (
    TrackingZoomPolicy,
    bounding_box_of,
    next_zoom_window,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.measure_metric_noise import (  # noqa: E402
    _AXIS_READERS,
    CONFIG,
    CONFIG_3D,
    INJECTION_DELTAS_MM,
    IPD_MM,
    MIN_USEFUL_FRAMES,
    PART_LABELS,
    REGION_SIGMA_IOD,
    SIGNAL_SPAN,
    Capture,
    Detection,
    InjectionReport,
    LandmarkStream,
    Series,
    build_series,
    collect_part_injections,
    display_flicker,
    display_levels,
    display_shift,
    display_stats,
    display_variants,
    heading,
    iter_video_frames,
    percent_value,
    resolve_model,
    to_pixel_3d,
    usable_injection_outcomes,
    video_fps,
)

DEFAULT_MAX_FRAMES: int = 600
DEFAULT_GAIN_FRAMES: int = 5
DEFAULT_FPS: float = 30.0

#: 白飛びと黒つぶれの判定閾値。画素の **最大チャンネル** に対して当てる。
#: 片チャンネルだけが振り切れている画素も白飛びに数えたいので最大を見る。
#: 黒つぶれも同じ量で測ると、両者が同じ 1 本の尺度の両端になる。
CLIP_HIGH_LEVEL: int = 250
CLIP_LOW_LEVEL: int = 10

#: セッション間の散らばりを出す最小のテイク数。
MIN_TAKES: int = 2

JSON_SCHEMA: str = "measure-session-variation/1.0.0"

SECTIONS: dict[str, str] = {
    "P": "撮影条件のプロファイル（テイクごとの共変量）",
    "V": "セッション間の表示値ばらつき",
    "G": "注入応答 ÷ セッション間ばらつき",
}

#: セクション V と G が読む指標。key は JSON、値は表の見出し。
METRIC_LABELS: dict[str, str] = {"symmetry_3d": "3D", "current_2d": "現行 2D"}

N_NOTE: str = (
    "  ※ テイクはこの実行で読んだ {takes} 本しかない。以下の「標準偏差」は n = {takes} の"
    "標準偏差であって、母集団の推定量として読める量ではない。"
)


# --------------------------------------------------------------------------
# フレーム 1 枚ぶんの撮影条件
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FrameProfile:
    """1 枚のフレームから読める撮影条件。顔が 1 つに定まらない枚は顔側が None。

    座標も画素も残さない。残すのはスカラーだけである（不変条件 1）。
    """

    index: int
    time_seconds: float
    face_count: int
    #: フレーム全体の平均輝度（0〜255）。`domain.frame_adjustments.mean_luminance`
    #: そのもので、FR-028 の自動明るさが読む量と同じ定義である。
    frame_luminance: float
    #: 最大チャンネルが `CLIP_HIGH_LEVEL` 以上 / `CLIP_LOW_LEVEL` 以下の画素の割合。
    frame_clip_high: float
    frame_clip_low: float
    face_luminance: float | None = None
    face_clip_high: float | None = None
    face_clip_low: float | None = None
    #: 眼アンカー中点どうしの距離（画素）。注入測定が使う `interocular_px` と同じ。
    interocular_px: float | None = None
    #: landmark 外接矩形の幅・高さ（フレーム比）。
    face_width_ratio: float | None = None
    face_height_ratio: float | None = None
    #: この検出結果から既定 policy が要求する追跡窓の幅（フレーム比）。
    window_width: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None

    @property
    def detected(self) -> bool:
        return self.face_count >= 1

    @property
    def single_face(self) -> bool:
        return self.face_count == 1


@dataclass(frozen=True, slots=True)
class Take:
    """1 本のテイク。撮影条件と、A〜L と同じ経路で出した d の時系列を持つ。"""

    name: str
    path: Path
    fps: float
    frames_seen: int
    profiles: tuple[FrameProfile, ...]
    capture: Capture
    series: Series

    @property
    def delta_seconds(self) -> float:
        return self.capture.delta_seconds

    @property
    def duration_seconds(self) -> float:
        return self.frames_seen * self.delta_seconds


# --------------------------------------------------------------------------
# 撮影条件の読み出し
# --------------------------------------------------------------------------
def _clip_ratios(image: NDArray[np.uint8]) -> tuple[float, float]:
    """白飛び・黒つぶれの画素割合。判定はチャンネルの最大値で行う。"""
    peak = image.max(axis=2)
    return (
        float(np.count_nonzero(peak >= CLIP_HIGH_LEVEL) / peak.size),
        float(np.count_nonzero(peak <= CLIP_LOW_LEVEL) / peak.size),
    )


def _face_region(
    image: NDArray[np.uint8],
    points: NDArray[np.float64],
) -> NDArray[np.uint8] | None:
    """landmark の外接矩形で画像を切る。フレーム外へはみ出した分は詰める。"""
    height, width = image.shape[:2]
    x0 = int(max(0, np.floor(points[:, 0].min())))
    y0 = int(max(0, np.floor(points[:, 1].min())))
    x1 = int(min(width, np.ceil(points[:, 0].max()) + 1))
    y1 = int(min(height, np.ceil(points[:, 1].max()) + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def profile_frame(
    image: NDArray[np.uint8],
    frame: LandmarkFrame,
    *,
    index: int,
    time_seconds: float,
    policy: TrackingZoomPolicy,
) -> FrameProfile:
    """1 枚ぶんの撮影条件。顔が 1 つに定まらない枚では顔側の欄を欠測にする。

    欠測を 0 で埋めない（不変条件 11）。中央値は欠測を除いた枚数で取る。
    """
    frame_high, frame_low = _clip_ratios(image)
    base = {
        "index": index,
        "time_seconds": time_seconds,
        "face_count": frame.face_count,
        "frame_luminance": mean_luminance(image),
        "frame_clip_high": frame_high,
        "frame_clip_low": frame_low,
    }
    if not frame.has_single_face:
        return FrameProfile(**base)

    try:
        points = to_pixel_array(frame)
    except LandmarkError:
        return FrameProfile(**base)

    left = points[list(CONFIG.left_eye_corners)].mean(axis=0)
    right = points[list(CONFIG.right_eye_corners)].mean(axis=0)
    box = bounding_box_of(frame.points)
    window = None if box is None else next_zoom_window(box, policy=policy)
    region = _face_region(image, points)
    face_high = face_low = face_luminance = None
    if region is not None and region.size > 0:
        face_luminance = mean_luminance(region)
        face_high, face_low = _clip_ratios(region)
    pose = frame.pose
    return FrameProfile(
        **base,
        face_luminance=face_luminance,
        face_clip_high=face_high,
        face_clip_low=face_low,
        interocular_px=float(np.linalg.norm(right - left)),
        face_width_ratio=None if box is None else box.width,
        face_height_ratio=None if box is None else box.height,
        window_width=None if window is None else window.width,
        yaw_deg=None if pose is None else pose.yaw_deg,
        pitch_deg=None if pose is None else pose.pitch_deg,
        roll_deg=None if pose is None else pose.roll_deg,
    )


def capture_take(
    path: Path,
    *,
    model_path: Path,
    model_version: str,
    mode: str,
    max_frames: int,
    fallback_fps: float,
) -> Take:
    """1 本のテイクを読み、撮影条件と d の時系列を同時に作る。

    検出は A〜L とまったく同じ `LandmarkStream`（= adapter の直叩き）である。
    追跡ズーム経路は通していない —— そちらは §13 の測定であり、本節が測るのは
    「同じ人を別の日に撮ると表示値がどれだけ動くか」なので、経路の違いを混ぜない。
    """
    fps = video_fps(path, fallback_fps)
    delta_seconds = 1.0 / fps
    policy = TrackingZoomPolicy()
    stream = LandmarkStream(model_path=model_path, model_version=model_version, mode=mode)
    profiles: list[FrameProfile] = []
    detections: list[Detection] = []
    seen = 0
    try:
        for index, image in enumerate(iter_video_frames(path, max_frames)):
            seen += 1
            # VIDEO モードは狭義単調増加を要求するので、A〜L と同じ下限で押さえる。
            timestamp_ms = max(index, round(index * delta_seconds * 1000.0))
            frame = stream.detect(image, frame_id=index, timestamp_ms=timestamp_ms)
            profiles.append(
                profile_frame(
                    image,
                    frame,
                    index=index,
                    time_seconds=index * delta_seconds,
                    policy=policy,
                )
            )
            if not frame.has_single_face:
                continue
            try:
                points = to_pixel_array(frame)
            except LandmarkError:
                continue
            height, width = image.shape[:2]
            detections.append(
                Detection(
                    index=index,
                    points=points,
                    mirrored=None,
                    pose=frame.pose,
                    points3d=to_pixel_3d(stream.last_points_3d, width=width, height=height),
                )
            )
            if seen % 100 == 0:
                print(f"    {path.name}: {seen} 枚処理", flush=True)
        model_sha256 = stream.model_sha256
        running_mode = stream.running_mode
        detector_class = stream.detector_class
    finally:
        stream.close()

    capture = Capture(
        detections=tuple(detections),
        frames_seen=seen,
        delta_seconds=delta_seconds,
        source_kind="video",
        mode=mode,
        model_version=model_version,
        model_sha256=model_sha256,
        running_mode=running_mode,
        detector_class=detector_class,
    )
    return Take(
        name=path.stem,
        path=path,
        fps=fps,
        frames_seen=seen,
        profiles=tuple(profiles),
        capture=capture,
        series=build_series(capture),
    )


# --------------------------------------------------------------------------
# 統計の器
# --------------------------------------------------------------------------
def column_stats(values: list[float | None]) -> dict[str, float] | None:
    """欠測を除いた 1 列ぶんの統計。全部欠けていれば None。"""
    usable = [value for value in values if value is not None and isfinite(value)]
    if not usable:
        return None
    array = np.asarray(usable, dtype=np.float64)
    return {
        "count": float(array.size),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.percentile(array, 5.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def spread_across_takes(values: list[float | None]) -> dict[str, float] | None:
    """テイクを跨いだ散らばり。`std` は **n = テイク数** の標準偏差である。"""
    usable = [value for value in values if value is not None and isfinite(value)]
    if len(usable) < MIN_TAKES:
        return None
    array = np.asarray(usable, dtype=np.float64)
    return {
        "takes": float(array.size),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "span": float(array.max() - array.min()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def outlier_view(values: list[float | None]) -> dict[str, Any] | None:
    """「1 本だけ離れているか」を割り算 1 つで見る。統計的な検定ではない。

        離れ方 = |最も外れたテイク - 残りの中央値| ÷ |残りの全幅|

    n=3 なら分母は残り 2 本の差そのものである。分母が 0 に近い列では発散する
    ので、その場合は比を出さずに差だけを返す。**この量は目盛りであって、
    有意性の主張ではない。**
    """
    usable = [(order, value) for order, value in enumerate(values) if value is not None]
    if len(usable) < 3:
        return None
    array = np.asarray([value for _, value in usable], dtype=np.float64)
    middle = float(np.median(array))
    position = int(np.argmax(np.abs(array - middle)))
    rest = np.delete(array, position)
    gap = float(abs(array[position] - float(np.median(rest))))
    span = float(rest.max() - rest.min())
    return {
        "take_index": usable[position][0],
        "gap": gap,
        "rest_span": span,
        "ratio": gap / span if span > 1e-12 else None,
    }


def _cell(value: float | None, digits: int) -> str:
    return "—" if value is None or not isfinite(value) else f"{value:.{digits}f}"


def _take_column(takes: list[Take], reader: Callable[[Take], float | None]) -> list[float | None]:
    return [reader(take) for take in takes]


# --------------------------------------------------------------------------
# 共変量の定義
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Covariate:
    """テイクごとに 1 つの数へ潰せる撮影条件。"""

    key: str
    label: str
    unit: str
    digits: int
    reader: Callable[[Take], float | None]
    #: 表に「標準偏差」列を出すか（姿勢のように揺れ自体が条件になる量）。
    with_spread: bool = False


def _frame_column(
    take: Take,
    field: str,
    *,
    single_only: bool = True,
) -> list[float | None]:
    rows = [item for item in take.profiles if item.single_face] if single_only else take.profiles
    return [getattr(item, field) for item in rows]


def _median_of(field: str, *, single_only: bool = True) -> Callable[[Take], float | None]:
    def read(take: Take) -> float | None:
        stats = column_stats(_frame_column(take, field, single_only=single_only))
        return None if stats is None else stats["median"]

    return read


def _std_of(field: str, *, single_only: bool = True) -> Callable[[Take], float | None]:
    def read(take: Take) -> float | None:
        stats = column_stats(_frame_column(take, field, single_only=single_only))
        return None if stats is None else stats["std"]

    return read


def _detection_rate(take: Take) -> float | None:
    if take.frames_seen == 0:
        return None
    return 100.0 * sum(1 for item in take.profiles if item.detected) / take.frames_seen


def _single_face_rate(take: Take) -> float | None:
    if take.frames_seen == 0:
        return None
    return 100.0 * sum(1 for item in take.profiles if item.single_face) / take.frames_seen


def _multi_face_rate(take: Take) -> float | None:
    if take.frames_seen == 0:
        return None
    return 100.0 * sum(1 for item in take.profiles if item.face_count >= 2) / take.frames_seen


def _percent_of(field: str) -> Callable[[Take], float | None]:
    def read(take: Take) -> float | None:
        stats = column_stats(_frame_column(take, field, single_only=False))
        return None if stats is None else 100.0 * stats["median"]

    return read


def _face_percent_of(field: str) -> Callable[[Take], float | None]:
    def read(take: Take) -> float | None:
        stats = column_stats(_frame_column(take, field))
        return None if stats is None else 100.0 * stats["median"]

    return read


COVARIATES: tuple[Covariate, ...] = (
    Covariate("frames_seen", "読込フレーム", "枚", 0, lambda take: float(take.frames_seen)),
    Covariate("fps", "fps", "", 2, lambda take: take.fps),
    Covariate("duration_s", "実時間", "秒", 2, lambda take: take.duration_seconds),
    Covariate("detection_rate", "検出率（顔 1 つ以上）", "%", 2, _detection_rate),
    Covariate("single_face_rate", "単一顔率", "%", 2, _single_face_rate),
    Covariate("multi_face_rate", "複数顔率", "%", 2, _multi_face_rate),
    Covariate("interocular_px", "IOD", "px", 2, _median_of("interocular_px"), with_spread=True),
    Covariate("face_width_ratio", "顔 bbox 幅", "フレーム比", 4, _median_of("face_width_ratio")),
    Covariate("face_height_ratio", "顔 bbox 高", "フレーム比", 4, _median_of("face_height_ratio")),
    Covariate("window_width", "追跡窓の幅（要求値）", "フレーム比", 4, _median_of("window_width")),
    Covariate("yaw_deg", "yaw", "度", 3, _median_of("yaw_deg"), with_spread=True),
    Covariate("pitch_deg", "pitch", "度", 3, _median_of("pitch_deg"), with_spread=True),
    Covariate("roll_deg", "roll", "度", 3, _median_of("roll_deg"), with_spread=True),
    Covariate(
        "frame_luminance",
        "フレーム輝度",
        "0〜255",
        2,
        _median_of("frame_luminance", single_only=False),
        with_spread=True,
    ),
    Covariate("face_luminance", "顔領域の輝度", "0〜255", 2, _median_of("face_luminance")),
    Covariate("frame_clip_high", "白飛び率（フレーム）", "%", 4, _percent_of("frame_clip_high")),
    Covariate("frame_clip_low", "黒つぶれ率（フレーム）", "%", 4, _percent_of("frame_clip_low")),
    Covariate("face_clip_high", "白飛び率（顔領域）", "%", 4, _face_percent_of("face_clip_high")),
    Covariate("face_clip_low", "黒つぶれ率（顔領域）", "%", 4, _face_percent_of("face_clip_low")),
)

#: 取ろうとして取れなかった量。理由まで含めて出力へ残す（作業単位の切り直し用）。
UNAVAILABLE: tuple[tuple[str, str], ...] = (
    (
        "landmark の信頼度",
        "MediaPipe Tasks の FaceLandmarker が返す NormalizedLandmark は "
        "visibility / presence がいずれも None で、face_blendshapes も空である"
        "（本 manifest の option では無効）。顔単位のスコアも返らない。"
        "adapter を変えずに取れる信頼度は存在しない。",
    ),
    (
        "実際の追跡窓の幅",
        "本節は landmarker を直叩きするので、`TrackingZoomLandmarker` が実際に"
        "選んだ窓ではない。表の窓幅は「この検出結果から既定 policy が要求する幅」"
        "であり、全フレーム再確認は含まない。実経路の窓幅は §13.3 にある。",
    ),
)


# --------------------------------------------------------------------------
# セクション P — 撮影条件のプロファイル
# --------------------------------------------------------------------------
PROFILE_NOTE = """
  何を並べるか

  「同じ人の顔が変わっていないとき、セッションを跨いでも同じ値が出るか」を
  判定するには、テイク間で表示値が動いたときに、それが顔の違いなのか撮り方の
  違いなのかを分けられなければならない。ここは後者の候補を並べる表である。

    顔の大きさ  IOD の画素数、顔 bbox の幅・高さ、追跡窓の幅
    姿勢        yaw / pitch / roll の中央値と標準偏差
    明るさ      フレーム輝度、顔領域の輝度、白飛び / 黒つぶれの割合
    検出        検出率、単一顔率、複数顔率
    尺          フレーム数、fps、実時間

  輝度は `domain.frame_adjustments.mean_luminance`（FR-028 の自動明るさが読む量）
  そのもので、知覚的な重み付けを持たない単純な画素平均である。白飛び / 黒つぶれ
  は画素の **最大チャンネル** が {high} 以上 / {low} 以下である割合とした。

  顔まわりの欄は、顔がちょうど 1 つに定まったフレームだけを母数にしている。
  フレーム輝度と白飛び / 黒つぶれ（フレーム）は全読込フレームが母数である。
"""


def section_p(takes: list[Take], report: dict[str, Any]) -> None:
    heading("P", "撮影条件のプロファイル — テイクごとの共変量")
    print(PROFILE_NOTE.format(high=CLIP_HIGH_LEVEL, low=CLIP_LOW_LEVEL))

    names = [take.name for take in takes]
    width = max(24, max(len(name) for name in names) + 2)
    print(f"  {'共変量':<26}{'単位':<12}" + "".join(f"{name:>{width}}" for name in names))
    print("  " + "-" * (38 + width * len(names)))
    rows: list[dict[str, Any]] = []
    for covariate in COVARIATES:
        values = _take_column(takes, covariate.reader)
        print(
            f"  {covariate.label:<26}{covariate.unit:<12}"
            + "".join(f"{_cell(value, covariate.digits):>{width}}" for value in values)
        )
        entry: dict[str, Any] = {
            "key": covariate.key,
            "label": covariate.label,
            "unit": covariate.unit,
            "median_by_take": dict(zip(names, values, strict=True)),
            "across_takes": spread_across_takes(values),
            "outlier": outlier_view(values),
        }
        if covariate.with_spread:
            spreads = _take_column(takes, _std_of(covariate.key))
            if covariate.key == "frame_luminance":
                spreads = _take_column(takes, _std_of("frame_luminance", single_only=False))
            print(
                f"  {'  ↑ 標準偏差（テイク内）':<26}{covariate.unit:<12}"
                + "".join(f"{_cell(value, covariate.digits):>{width}}" for value in spreads)
            )
            entry["within_take_std"] = dict(zip(names, spreads, strict=True))
        rows.append(entry)
    print()

    print("  テイク間でどれだけ動いたか（中央値どうしの比較）")
    print(
        f"  {'共変量':<26}{'単位':<12}{'最小':>12}{'最大':>12}"
        f"{'全幅':>12}{'相対全幅':>12}{'外れたテイク':>16}{'離れ方':>10}"
    )
    print("  " + "-" * 112)
    ranked: list[tuple[float, str]] = []
    for covariate, entry in zip(COVARIATES, rows, strict=True):
        across = entry["across_takes"]
        if across is None:
            continue
        relative = (
            100.0 * across["span"] / abs(across["median"])
            if abs(across["median"]) > 1e-12
            else None
        )
        outlier = entry["outlier"]
        ratio = None if outlier is None else outlier["ratio"]
        # 残りの全幅が 0 の列では「どれが外れているか」を決められない。名前も出さない。
        who = "—" if outlier is None or ratio is None else names[int(outlier["take_index"])]
        print(
            f"  {covariate.label:<26}{covariate.unit:<12}"
            f"{_cell(across['min'], covariate.digits):>12}"
            f"{_cell(across['max'], covariate.digits):>12}"
            f"{_cell(across['span'], covariate.digits):>12}"
            f"{('—' if relative is None else f'{relative:.2f}%'):>12}"
            f"{who:>16}{('—' if ratio is None else f'{ratio:.2f}x'):>10}"
        )
        entry["relative_span_percent"] = relative
        if relative is not None:
            ranked.append((relative, covariate.label))
    print()
    print("  相対全幅 = 全幅 ÷ |テイク間中央値|。単位の違う列を同じ尺度で見るために置いた。")
    print("  外れたテイク / 離れ方 = 中央値からいちばん離れたテイクと、")
    print("      |その値 - 残りの中央値| ÷ |残りの全幅|。")
    print("      **n が小さいので、これは目盛りであって検定ではない。**")
    print()

    if ranked:
        ranked.sort(reverse=True)
        print("  相対全幅の大きい順（上ほど、テイク間で動いている列）")
        for order, (relative, label) in enumerate(ranked[:8], start=1):
            print(f"     {order}. {label}   {relative:.2f}%")
        print()

    print("  取れなかった量")
    for label, reason in UNAVAILABLE:
        print(f"     ● {label}")
        for line in _wrap(reason):
            print(f"       {line}")
    print()

    report["capture_profile"] = {
        "section": "P",
        "takes": names,
        "clip_high_level": CLIP_HIGH_LEVEL,
        "clip_low_level": CLIP_LOW_LEVEL,
        "covariates": rows,
        "unavailable": [{"item": label, "reason": reason} for label, reason in UNAVAILABLE],
    }


def _wrap(text: str) -> list[str]:
    """句点で切るだけの折り返し。語の途中で改行しないことだけを目的にしている。"""
    parts = [part for part in text.split("。") if part]
    return [part + "。" for part in parts]


# --------------------------------------------------------------------------
# セクション V — セッション間の表示値ばらつき
# --------------------------------------------------------------------------
VARIATION_NOTE = """
  何を測るか

  各テイクの d を `requirement.md` §8.6.3 の表示式へ通し、テイクごとの分布と、
  **テイク間で中央値がどれだけ動くか**を出す。

      symmetry_percent = clamp(100 × (1 - d), 0, 100)

  1pt（表示 1 パーセントポイント）= 0.01 IOD である。この対応は厳密で、IPD の
  代表値の当否と無関係である（§12.0）。

  分布はセクション L と同じ関数（`display_stats` / `display_flicker` /
  `display_levels`）で出しているので、take01 / take02 の行は §12.1 の記録と
  一致するはずである。一致しなければ、測定系のどこかが変わったということである。
"""


def _display_values(take: Take, metric: str) -> NDArray[np.float64] | None:
    """テイク 1 本ぶんの表示値（生値、フレームごと）。単位は pt。"""
    series = take.series
    if series.count == 0:
        return None
    if metric == "current_2d":
        distances, times = series.distances, series.times
    else:
        mask = np.isfinite(series.distances3d)
        if int(mask.sum()) < 2:
            return None
        distances, times = series.distances3d[mask], series.times[mask]
    return display_variants(distances, times, take.delta_seconds)[0].values


def section_v(takes: list[Take], report: dict[str, Any]) -> None:
    heading("V", "セッション間の表示値ばらつき")
    print(VARIATION_NOTE)
    print(N_NOTE.format(takes=len(takes)))
    print()

    names = [take.name for take in takes]
    values: dict[str, dict[str, NDArray[np.float64] | None]] = {
        key: {take.name: _display_values(take, key) for take in takes} for key in METRIC_LABELS
    }

    print("  1) 各テイクの表示値の分布（生値、フレームごと。単位はすべて pt）")
    print(
        f"     {'方式':<8}{'テイク':<12}{'枚数':>6}{'最小':>8}{'p05':>8}{'p25':>8}"
        f"{'中央値':>9}{'p75':>8}{'p95':>8}{'最大':>8}{'標準偏差':>10}{'p95-p05':>10}"
    )
    print("     " + "-" * 105)
    distributions: dict[str, dict[str, dict[str, float] | None]] = {}
    for key, label in METRIC_LABELS.items():
        distributions[key] = {}
        for take in takes:
            array = values[key][take.name]
            stats = None if array is None else display_stats(array)
            distributions[key][take.name] = stats
            if stats is None:
                print(f"     {label:<8}{take.name:<12}{'—':>6}")
                continue
            print(
                f"     {label:<8}{take.name:<12}{int(stats['count']):6d}"
                f"{stats['min']:8.2f}{stats['p05']:8.2f}{stats['p25']:8.2f}"
                f"{stats['median']:9.3f}{stats['p75']:8.2f}{stats['p95']:8.2f}"
                f"{stats['max']:8.2f}{stats['std']:10.3f}{stats['span_p05_p95']:10.3f}"
            )
        if len(takes) > 1:
            print("     " + "-" * 105)
    print()

    print("  2) チラつきと、実際に出る数字の種類（生値、撮影 fps の隣接フレーム）")
    print(
        f"     {'方式':<8}{'テイク':<12}{'|Δ|中央値':>11}{'|Δ|p95':>10}"
        f"{'|Δ|最大':>10}{'整数%の種類':>13}{'小数2桁の種類':>15}"
    )
    print("     " + "-" * 80)
    flickers: dict[str, dict[str, dict[str, Any] | None]] = {}
    for key, label in METRIC_LABELS.items():
        flickers[key] = {}
        for take in takes:
            array = values[key][take.name]
            flicker = None if array is None else display_flicker(array)
            levels = None if array is None else display_levels(array)
            flickers[key][take.name] = None if flicker is None else {**flicker, "levels": levels}
            if flicker is None or levels is None:
                print(f"     {label:<8}{take.name:<12}{'—':>11}")
                continue
            print(
                f"     {label:<8}{take.name:<12}{flicker['median']:11.3f}{flicker['p95']:10.3f}"
                f"{flicker['max']:10.3f}{levels['integer']:13d}{levels['two_decimals']:15d}"
            )
    print()

    print("  3) テイク間の中央値（これがセッション間ばらつきの本体）")
    print(
        f"     {'方式':<8}"
        + "".join(f"{name:>12}" for name in names)
        + f"{'標準偏差':>12}{'全幅':>10}{'テイク内の幅':>14}"
    )
    print("     " + "-" * (20 + 12 * len(names) + 36))
    session: dict[str, dict[str, Any]] = {}
    for key, label in METRIC_LABELS.items():
        medians = [
            None if distributions[key][name] is None else distributions[key][name]["median"]
            for name in names
        ]
        across = spread_across_takes(medians)
        inside = [
            None if distributions[key][name] is None else distributions[key][name]["span_p05_p95"]
            for name in names
        ]
        inside_stats = column_stats(inside)
        print(
            f"     {label:<8}"
            + "".join(f"{_cell(value, 3):>12}" for value in medians)
            + f"{_cell(None if across is None else across['std'], 3):>12}"
            + f"{_cell(None if across is None else across['span'], 3):>10}"
            + f"{_cell(None if inside_stats is None else inside_stats['median'], 3):>14}"
        )
        session[key] = {
            "median_by_take": dict(zip(names, medians, strict=True)),
            "across_takes": across,
            "within_take_span_p05_p95_median": (
                None if inside_stats is None else inside_stats["median"]
            ),
            "outlier": outlier_view(medians),
        }
    print()
    print(f"     標準偏差は n = {len(takes)} の標準偏差である（ddof=1）。全幅 = 最大 - 最小。")
    print("     テイク内の幅 = 各テイクの p95-p05 の中央値。テイク内とテイク間を混ぜないこと。")
    print(f"     参考: 文献の信号全幅（対称な成人 ↔ 非対称な成人）は {SIGNAL_SPAN * 100.0:.2f}pt。")
    print()

    if len(takes) >= 3:
        print("  4) 1 本だけ離れているか")
        for key, label in METRIC_LABELS.items():
            outlier = session[key]["outlier"]
            if outlier is None:
                continue
            who = names[int(outlier["take_index"])]
            ratio = outlier["ratio"]
            print(
                f"     {label:<8}最も離れたテイク: {who}"
                f"   離れ {outlier['gap']:.3f}pt"
                f"   残りの全幅 {outlier['rest_span']:.3f}pt"
                f"   比 {'—' if ratio is None else f'{ratio:.2f}x'}"
            )
        print()

    print("  5) 共変量との対応（テイク間で動いた量を、表示値の中央値と並べる）")
    header = f"     {'量':<26}{'単位':<12}" + "".join(f"{name:>12}" for name in names)
    print(header + f"{'全幅':>12}")
    print("     " + "-" * (38 + 12 * len(names) + 12))
    correspondence: list[dict[str, Any]] = []
    for key, label in METRIC_LABELS.items():
        medians = [
            None if distributions[key][name] is None else distributions[key][name]["median"]
            for name in names
        ]
        across = spread_across_takes(medians)
        print(
            f"     {f'表示値の中央値（{label}）':<26}{'pt':<12}"
            + "".join(f"{_cell(value, 3):>12}" for value in medians)
            + f"{_cell(None if across is None else across['span'], 3):>12}"
        )
    for covariate in COVARIATES:
        column = _take_column(takes, covariate.reader)
        across = spread_across_takes(column)
        print(
            f"     {covariate.label:<26}{covariate.unit:<12}"
            + "".join(f"{_cell(value, covariate.digits):>12}" for value in column)
            + f"{_cell(None if across is None else across['span'], covariate.digits):>12}"
        )
        correspondence.append(
            {
                "key": covariate.key,
                "label": covariate.label,
                "by_take": dict(zip(names, column, strict=True)),
                "span": None if across is None else across["span"],
            }
        )
    print()
    print(f"     n = {len(takes)} なので相関係数は出さない。並べた数字と全幅までにとどめる。")
    print()

    report["session_variation"] = {
        "section": "V",
        "takes": names,
        "distribution_pt": {
            key: {name: distributions[key][name] for name in names} for key in METRIC_LABELS
        },
        "flicker_pt": {key: {name: flickers[key][name] for name in names} for key in METRIC_LABELS},
        "session": session,
        "covariate_correspondence": correspondence,
        "literature_signal_span_pt": SIGNAL_SPAN * 100.0,
    }


# --------------------------------------------------------------------------
# セクション G — 注入応答 ÷ セッション間ばらつき
# --------------------------------------------------------------------------
GAIN_NOTE = """
  何を測るか

  §12.5 と同じ領域変位場（Gauss、sigma = {sigma:.2f} IOD）を各テイクで再実行し、
  「折り返さない最大の δ を入れたとき表示値が何 pt 動くか」を部位ごとに出す。
  そのうえで、セクション V が出したセッション間ばらつきで割る。

      信号 ÷ セッション間ばらつき = |注入の Δpt| ÷ テイク間中央値の散らばり

  この比が 1 を大きく超えていれば、「前回と比べてどう変わったか」の表示は、
  その部位の変化をセッション間の再現性の外側で見せられる。1 前後なら、
  同じ顔を撮り直しただけでも同じ大きさの変化が出る。

  注入・フレームの選び方（yaw 中央値の近傍 n 枚）・δ の刻みは、セクション E / J
  とまったく同じ関数（`collect_part_injections`）を呼んでいる。
"""


def section_g(
    arguments: argparse.Namespace,
    takes: list[Take],
    report: dict[str, Any],
) -> None:
    heading("G", "注入応答 ÷ セッション間ばらつき")
    print(GAIN_NOTE.format(sigma=REGION_SIGMA_IOD))

    variation = report.get("session_variation")
    if variation is None:
        print("  セクション V を実行していないので、比の分母がありません。V も選んでください。")
        report["signal_over_session"] = {"section": "G", "reason": "no-session-variation"}
        return

    names = [take.name for take in takes]
    collected: dict[str, dict[str, InjectionReport] | None] = {}
    for take in takes:
        print(f"  ● {take.name}")
        shim = argparse.Namespace(
            self_test=False,
            gain_frames=arguments.gain_frames,
            model=arguments.model,
            mode=arguments.mode,
            video=take.path,
            frames=None,
        )
        collected[take.name] = collect_part_injections(shim, take.capture)
    print()

    deltas = [value for value in INJECTION_DELTAS_MM if value > 0.0]
    print(f"  1) 各テイクの注入応答（Δpt、折り返さない最大の δ。刻みは {deltas} mm）")
    rows: list[dict[str, Any]] = []
    ratios: dict[str, dict[str, Any]] = {}
    for metric_key, label in METRIC_LABELS.items():
        axis = "3d" if metric_key == "symmetry_3d" else "2d"
        baseline_read, truth_read, detected_read = _AXIS_READERS[axis]
        across = variation["session"][metric_key]["across_takes"]
        print()
        print(f"     ● {label}")
        print(
            f"       {'部位':<6}"
            + "".join(f"{name:>12}" for name in names)
            + f"{'中央値':>10}{'真値δ最大':>12}"
        )
        print("       " + "-" * (18 + 12 * len(names) + 22))
        for part, part_label in PART_LABELS.items():
            shifts: list[float | None] = []
            truths: list[float | None] = []
            for take in takes:
                reports = collected[take.name]
                item = None if reports is None else reports.get(part_label)
                if item is None:
                    shifts.append(None)
                    truths.append(None)
                    continue
                baseline = baseline_read(item)
                usable = usable_injection_outcomes(item)
                widest = usable[-1] if usable else None
                shifts.append(
                    None if widest is None else display_shift(baseline, detected_read(widest))
                )
                truths.append(
                    None if widest is None else display_shift(baseline, truth_read(widest))
                )
            middle = column_stats(shifts)
            truth_middle = column_stats(truths)
            print(
                f"       {part_label:<6}"
                + "".join(f"{_cell(value, 3):>12}" for value in shifts)
                + f"{_cell(None if middle is None else middle['median'], 3):>10}"
                + f"{_cell(None if truth_middle is None else truth_middle['median'], 3):>12}"
            )
            entry = {
                "axis": metric_key,
                "part": part,
                "label": part_label,
                "detected_shift_pt_by_take": dict(zip(names, shifts, strict=True)),
                "truth_shift_pt_by_take": dict(zip(names, truths, strict=True)),
                "detected_shift_pt_median": None if middle is None else middle["median"],
                "detected_shift_pt_span": (
                    None if middle is None else middle["max"] - middle["min"]
                ),
            }
            rows.append(entry)
        ratios[metric_key] = {"across_takes": across}
    print()
    print("     Δpt は負が「より非対称に見える」向き。真値δ最大 = 同じ変位を landmark へ")
    print("     直接入れたときの Δpt（検出を通さない）。テイクごとの値の中央値を併記した。")
    print()

    print("  2) 信号 ÷ セッション間ばらつき")
    print(
        f"     {'方式':<8}{'部位':<6}{'|Δpt| 中央値':>14}{'÷ 標準偏差':>14}"
        f"{'÷ 全幅':>12}{'÷ テイク内の幅':>16}"
    )
    print("     " + "-" * 74)
    for metric_key, label in METRIC_LABELS.items():
        across = variation["session"][metric_key]["across_takes"]
        inside = variation["session"][metric_key]["within_take_span_p05_p95_median"]
        for entry in [item for item in rows if item["axis"] == metric_key]:
            signal = entry["detected_shift_pt_median"]
            magnitude = None if signal is None else abs(signal)
            by_std = _ratio(magnitude, None if across is None else across["std"])
            by_span = _ratio(magnitude, None if across is None else across["span"])
            by_inside = _ratio(magnitude, inside)
            print(
                f"     {label:<8}{entry['label']:<6}{_cell(magnitude, 3):>14}"
                f"{_ratio_cell(by_std):>14}{_ratio_cell(by_span):>12}{_ratio_cell(by_inside):>16}"
            )
            entry["signal_over_session_std"] = by_std
            entry["signal_over_session_span"] = by_span
            entry["signal_over_within_take_span"] = by_inside
    print()
    print(f"     分母（方式ごと）: 標準偏差 = n={len(takes)} のテイク間中央値の標準偏差、")
    print("     全幅 = テイク間中央値の 最大 - 最小、テイク内の幅 = 各テイクの p95-p05 の中央値。")
    print("     **どの分母も標本数がテイク数しかない。比の絶対値を有効数字どおりに読まないこと。**")
    print()

    report["signal_over_session"] = {
        "section": "G",
        "takes": names,
        "sigma_iod": REGION_SIGMA_IOD,
        "deltas_mm": deltas,
        "gain_frames": arguments.gain_frames,
        "rows": rows,
        "denominators": ratios,
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def _ratio_cell(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}x"


# --------------------------------------------------------------------------
# 実写なしの健全性確認
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str


def _synthetic_landmark_frame(
    *,
    face_count: int = 1,
    center: tuple[float, float] = (0.5, 0.5),
    half_width: float = 0.10,
    half_height: float = 0.14,
) -> LandmarkFrame:
    """既知の外接矩形と眼アンカーを持つ合成フレーム。画像も検出器も要らない。"""
    from face_symmetry_lab.domain.landmarks import build_landmark_frame

    count = max(CONFIG.landmark_count, 0)
    points = [(center[0], center[1]) for _ in range(count)]
    points[CONFIG.left_eye_corners[0]] = (center[0] - half_width, center[1])
    points[CONFIG.left_eye_corners[1]] = (center[0] - half_width, center[1])
    points[CONFIG.right_eye_corners[0]] = (center[0] + half_width, center[1])
    points[CONFIG.right_eye_corners[1]] = (center[0] + half_width, center[1])
    points[0] = (center[0] - half_width, center[1] - half_height)
    points[1] = (center[0] + half_width, center[1] + half_height)
    return build_landmark_frame(
        frame_id=0,
        face_count=face_count,
        points=tuple(points) if face_count == 1 else (),
        width=1280,
        height=720,
    )


def _check_display_scale() -> list[Check]:
    """表示値の読み替えが `measure_metric_noise`（A〜L）と同一であること。"""
    samples = (0.0, 0.0172, 0.0741, 0.5, 1.2)
    ok = all(
        abs(percent_value(value) - min(100.0, max(0.0, (1.0 - value) * 100.0))) < 1e-12
        for value in samples
    )
    shift = display_shift(0.02, 0.03)
    return [
        Check(
            "表示値は §8.6.3 の式そのもの（A〜L と同じ関数を呼ぶ）",
            ok,
            f"d={samples} → " + ", ".join(f"{percent_value(value):.3f}" for value in samples),
        ),
        Check(
            "Δpt は d の差の 100 倍（1pt = 0.01 IOD）",
            shift is not None and abs(shift - (-1.0)) < 1e-12,
            f"d 0.02 → 0.03 で {shift:+.6f}pt" if shift is not None else "—",
        ),
    ]


def _check_spread() -> list[Check]:
    """テイクを跨いだ散らばりが、閉じた式と一致すること。"""
    medians: list[float | None] = [98.252, 98.233, 98.530]
    across = spread_across_takes(medians)
    expected_std = float(np.std(np.asarray(medians, dtype=np.float64), ddof=1))
    expected_span = 98.530 - 98.233
    # 離れ方の分子は「外れた 1 本 - 残りの *中央値*」であって、残りの最大値ではない。
    expected_gap = 98.530 - (98.252 + 98.233) / 2.0
    outlier = outlier_view(medians)
    return [
        Check(
            "テイク間の標準偏差は ddof=1（n = テイク数）",
            across is not None and abs(across["std"] - expected_std) < 1e-12,
            "—" if across is None else f"std={across['std']:.6f} 期待={expected_std:.6f}",
        ),
        Check(
            "テイク間の全幅は 最大 - 最小",
            across is not None and abs(across["span"] - expected_span) < 1e-12,
            "—" if across is None else f"span={across['span']:.6f} 期待={expected_span:.6f}",
        ),
        Check(
            "外れたテイクは中央値からいちばん離れた 1 本、比は残りの全幅で割った値",
            outlier is not None
            and outlier["take_index"] == 2
            and abs(outlier["gap"] - expected_gap) < 1e-12
            and outlier["ratio"] is not None
            and abs(outlier["ratio"] - expected_gap / (98.252 - 98.233)) < 1e-9,
            "—"
            if outlier is None
            else f"index={outlier['take_index']} gap={outlier['gap']:.6f} "
            f"ratio={outlier['ratio']:.4f}",
        ),
        Check(
            "テイクが 1 本しかなければ散らばりを出さない",
            spread_across_takes([1.0]) is None,
            f"MIN_TAKES={MIN_TAKES}",
        ),
    ]


def _check_luminance() -> list[Check]:
    """輝度と白飛び・黒つぶれが、既知の合成画像で閉じた値になること。"""
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :10] = 255
    image[:, 10:] = 5
    expected_luminance = (255.0 * 10 + 5.0 * 10) / 20.0
    high, low = _clip_ratios(image)
    frame = _synthetic_landmark_frame()
    region = _face_region(np.zeros((720, 1280, 3), dtype=np.uint8), to_pixel_array(frame))
    return [
        Check(
            "フレーム輝度は domain の mean_luminance そのもの",
            abs(mean_luminance(image) - expected_luminance) < 1e-9,
            f"{mean_luminance(image):.3f} 期待={expected_luminance:.3f}",
        ),
        Check(
            f"白飛び = 最大チャンネル ≥ {CLIP_HIGH_LEVEL}、黒つぶれ = ≤ {CLIP_LOW_LEVEL}",
            abs(high - 0.5) < 1e-12 and abs(low - 0.5) < 1e-12,
            f"白飛び {high:.3f} / 黒つぶれ {low:.3f}（半分ずつの合成画像）",
        ),
        Check(
            "顔領域は landmark の外接矩形で切る",
            region is not None
            and region.shape[0] == int(np.ceil(0.64 * 720) + 1) - int(np.floor(0.36 * 720))
            and region.shape[1] == int(np.ceil(0.60 * 1280) + 1) - int(np.floor(0.40 * 1280)),
            "—" if region is None else f"shape={region.shape}",
        ),
    ]


def _check_profile() -> list[Check]:
    """共変量の抽出が、既知の合成フレームで閉じた値になること。"""
    image = np.full((720, 1280, 3), 128, dtype=np.uint8)
    policy = TrackingZoomPolicy()
    single = profile_frame(
        image,
        _synthetic_landmark_frame(),
        index=0,
        time_seconds=0.0,
        policy=policy,
    )
    multi = profile_frame(
        image,
        _synthetic_landmark_frame(face_count=2),
        index=1,
        time_seconds=0.2,
        policy=policy,
    )
    expected_iod = 0.20 * 1280.0
    expected_window = min(max(0.20 * (1.0 + policy.margin_ratio), policy.min_window_size), 1.0)
    return [
        Check(
            "IOD は眼アンカー中点どうしの画素距離",
            single.interocular_px is not None and abs(single.interocular_px - expected_iod) < 1e-9,
            f"{single.interocular_px:.3f}px 期待={expected_iod:.3f}px"
            if single.interocular_px is not None
            else "—",
        ),
        Check(
            "顔 bbox の幅・高さはフレーム比",
            single.face_width_ratio is not None
            and abs(single.face_width_ratio - 0.20) < 1e-9
            and single.face_height_ratio is not None
            and abs(single.face_height_ratio - 0.28) < 1e-9,
            f"幅 {single.face_width_ratio:.4f} / 高さ {single.face_height_ratio:.4f}"
            if single.face_width_ratio is not None and single.face_height_ratio is not None
            else "—",
        ),
        Check(
            "追跡窓の幅は domain の next_zoom_window（既定 policy）そのもの",
            single.window_width is not None and abs(single.window_width - expected_window) < 1e-12,
            f"{single.window_width:.4f} 期待={expected_window:.4f}"
            if single.window_width is not None
            else "—",
        ),
        Check(
            "顔が 1 つに定まらない枚では顔側を欠測にする（0 で埋めない）",
            multi.face_count == 2
            and multi.interocular_px is None
            and multi.face_luminance is None
            and multi.window_width is None
            and isfinite(multi.frame_luminance),
            f"face_count={multi.face_count} フレーム輝度={multi.frame_luminance:.1f} 顔側=None",
        ),
    ]


def _check_ratio() -> list[Check]:
    """信号 ÷ ばらつきの算術。分母 0 で発散させないこと。"""
    return [
        Check(
            "比は 素直な割り算",
            _ratio(1.75, 0.28) is not None and abs(_ratio(1.75, 0.28) - 6.25) < 1e-9,  # type: ignore[operator]
            f"1.750 ÷ 0.280 = {_ratio(1.75, 0.28):.4f}",  # type: ignore[str-format]
        ),
        Check(
            "分母が 0 なら比を出さない",
            _ratio(1.0, 0.0) is None and _ratio(1.0, None) is None,
            "0 と None のどちらも None を返す",
        ),
    ]


def run_self_test() -> list[Check]:
    checks: list[Check] = []
    checks.extend(_check_display_scale())
    checks.extend(_check_spread())
    checks.extend(_check_luminance())
    checks.extend(_check_profile())
    checks.extend(_check_ratio())
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
        help="動画ファイル（複数指定可。1 本 = 1 テイク）",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="実写なしで、共変量と散らばりの計算が正しいことを確認する",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="処理枚数の上限")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="fps 取得失敗時の仮定")
    parser.add_argument("--model", type=Path, default=None, help="landmark モデルの明示指定")
    parser.add_argument(
        "--mode",
        choices=("image", "video"),
        default="image",
        help="検出モード。§12 と揃えるなら image",
    )
    parser.add_argument(
        "--gain-frames",
        type=int,
        default=DEFAULT_GAIN_FRAMES,
        help="セクション G が各テイクで注入するフレーム数（yaw 中央値に近い順）",
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
    if arguments.gain_frames < 1:
        raise SystemExit("--gain-frames は 1 以上である必要があります。")
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
    if len(arguments.video) < MIN_TAKES:
        print(
            f"  ※ テイクが {len(arguments.video)} 本しかないので、"
            f"セッション間の散らばりは出ません（{MIN_TAKES} 本以上必要）。"
        )


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
        "mode": arguments.mode,
    }

    if not arguments.self_test:
        model_path, model_version = resolve_model(arguments.model)
        report["model_version"] = model_version
        print(f"model   : {model_version}")
        print(f"入力    : {', '.join(path.name for path in arguments.video)}")
        print()
        takes = [
            capture_take(
                path,
                model_path=model_path,
                model_version=model_version,
                mode=arguments.mode,
                max_frames=arguments.max_frames,
                fallback_fps=arguments.fps,
            )
            for path in arguments.video
        ]
        report["takes"] = [
            {
                "name": take.name,
                "frames_seen": take.frames_seen,
                "frames_single_face": len(take.capture.detections),
                "frames_with_distance": take.series.count,
                "fps": take.fps,
                "running_mode": take.capture.running_mode,
                "detector_class": take.capture.detector_class,
                "model_sha256": take.capture.model_sha256,
            }
            for take in takes
        ]
        short = [take.name for take in takes if take.series.count < MIN_USEFUL_FRAMES]
        if short:
            print(f"  ※ {MIN_USEFUL_FRAMES} 枚未満のテイクがある: {', '.join(short)}")
        if "P" in selected:
            section_p(takes, report)
        if "V" in selected:
            section_v(takes, report)
        if "G" in selected:
            section_g(arguments, takes, report)

    failures = 0
    if arguments.self_test:
        heading("S", "健全性確認 — 実写なしで、共変量と散らばりの計算を検証する")
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
    print("解説: docs/symmetry-metric-measurements.md §14 / docs/capture-protocol.md")
    print("=" * 100)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
