"""FR-038: 参考年齢の表示ゲート（`age-gate/1.0.0`）の純粋計算（§10.11, ADR-0005）。

このモジュールは年齢を推定しない。478 点の正規化済みランドマーク・姿勢・
`symmetry-3d/1.0.0` の左右差から 41 次元の幾何特徴を組み立て、
`models/age_gate_model.onnx`（決定木アンサンブル）が「参考年齢を表示してよい
18〜75 歳の範囲内か」を判定するための入力を作るだけである。

特徴量の定義の正本は `research/utkface-symmetry/utkface_age_gate_lab.ipynb` の
cell 6（``geometry_features`` / ``IDX`` / ``GEOMETRY_FEATURE_NAMES``）であり、
本モジュールはそれをそのまま移植する。特徴量の並びが 1 つでもずれると、
モデルは例外を投げずに違う確率を返す（`tests/model/test_age_gate_contract.py`
が実モデルで固定する）。

対応ペア表（``PART_PAIRS`` 相当）は `configs/metrics/symmetry-3d-v1.json`
（`symmetry-3d/1.0.0`）の parts と現時点では同一だが、``age-gate-features/1.0.0``
は学習時に固定されたこの対応表から独立して凍結されている必要があるため
（将来 symmetry-3d 側の bundle が改訂されても、この特徴量の定義は変わっては
ならない）、ここに固定コピーを持つ。domain は adapters（JSON 読み込み）に
依存できないため、実行時に `configs/metrics/symmetry-3d-v1.json` から導くこと
はできない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.landmarks import FACE_LANDMARK_COUNT
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.pose import FacePose

__all__ = [
    "FEATURE_NAMES",
    "GEOMETRY_FEATURE_NAMES",
    "POSE_FEATURE_NAMES",
    "SYMMETRY_FEATURE_NAMES",
    "AgeGateDecision",
    "AgeGateError",
    "AgeGateFeatures",
    "SymmetryDistances",
    "build_features",
    "decide",
]


class AgeGateError(ValueError):
    """年齢ゲートの入力・判定が不正である。"""


class SymmetryDistances(Protocol):
    """`build_features` が `symmetry_3d` へ要求する最小の境界。

    実体は `domain.symmetry_3d.Symmetry3DResult` だが、ここで読むのは
    `distance(name)` と `equal_weight_distance` の 2 つだけであり、それ以外の
    属性（`splits`・`overall` 等）へは触れない。
    """

    @property
    def equal_weight_distance(self) -> float | None: ...

    @property
    def overall_distance(self) -> float | None: ...

    def distance(self, name: str) -> float | None: ...


#: notebook cell 6 の `IDX` をそのまま移植したもの。MediaPipe FaceMesh の
#: 標準 index で、年齢帯の手がかりになる比率を測るために使う 1 点ずつの参照。
IDX: Final[dict[str, int]] = {
    "forehead": 10,  # 額の正中最上部
    "chin": 152,  # 顎の正中最下部
    "cheek_l": 234,  # 左頬の最外
    "cheek_r": 454,  # 右頬の最外
    "jaw_l": 58,  # 左下顎角
    "jaw_r": 288,  # 右下顎角
    "nasion": 168,  # 両眼のあいだ（鼻根）
    "nose_tip": 1,  # 鼻尖
    "subnasale": 2,  # 鼻下
    "nose_wing_l": 98,
    "nose_wing_r": 327,
    "mouth_l": 61,
    "mouth_r": 291,
    "lip_top": 0,  # 上唇の外側正中
    "lip_bottom": 17,  # 下唇の外側正中
    "eye_l_out": 33,
    "eye_l_in": 133,
    "eye_l_top": 159,
    "eye_l_bottom": 145,
    "eye_r_out": 263,
    "eye_r_in": 362,
    "eye_r_top": 386,
    "eye_r_bottom": 374,
    "brow_l": 105,
    "brow_r": 334,
}

#: notebook cell 6 の `GEOMETRY_FEATURE_NAMES` をそのまま移植したもの（32 個）。
GEOMETRY_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "face_height",
    "face_width",
    "face_aspect",
    "forehead_ratio",
    "midface_ratio",
    "lower_face_ratio",
    "chin_ratio",
    "nose_length_ratio",
    "nose_width_ratio",
    "nose_depth",
    "mouth_width_ratio",
    "mouth_position_ratio",
    "lip_height_ratio",
    "jaw_width_ratio",
    "cheek_height_ratio",
    "jaw_taper",
    "eye_width_left",
    "eye_width_right",
    "eye_open_left",
    "eye_open_right",
    "eye_aspect_left",
    "eye_aspect_right",
    "brow_gap_left",
    "brow_gap_right",
    "eye_to_nose_ratio",
    "eye_to_mouth_ratio",
    "depth_forehead",
    "depth_chin",
    "depth_cheek_mean",
    "depth_spread",
    "contour_depth_range",
    "face_convexity",
)

#: `FacePose` を素通しする 3 特徴。度のまま、スケールし直さない。
POSE_FEATURE_NAMES: Final[tuple[str, ...]] = ("yaw_deg", "pitch_deg", "roll_deg")

#: `symmetry-3d/1.0.0` の生距離を素通しする 6 特徴。`s3d_distance` は
#: `Symmetry3DResult.equal_weight_distance`（`s_3d`）、`distance_*` は
#: `PART_NAMES` の順の部位別距離と全体距離（`d_3d`）。
SYMMETRY_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "s3d_distance",
    "distance_contour",
    "distance_eyes",
    "distance_nose",
    "distance_mouth",
    "distance_overall",
)

#: 学習時に固定された 41 次元の特徴量の並び（`configs/age-gate-v1.json` の
#: `feature_names` と一致することを `tests/model/test_age_gate_contract.py` が
#: 実モデルで検証する）。
FEATURE_NAMES: Final[tuple[str, ...]] = (
    *GEOMETRY_FEATURE_NAMES,
    *POSE_FEATURE_NAMES,
    *SYMMETRY_FEATURE_NAMES,
)

#: notebook cell 6 の `PART_PAIRS` をそのまま移植したもの。
#: `configs/metrics/symmetry-3d-v1.json` の `parts` と現時点では一致するが、
#: `age-gate-features/1.0.0` はその bundle の将来の改訂から独立して凍結されて
#: いる必要があるため、実行時に読み込まずここへ固定コピーを持つ。
_PART_PAIRS: Final[dict[str, tuple[tuple[int, int], ...]]] = {
    "contour": (
        (21, 251),
        (54, 284),
        (58, 288),
        (67, 297),
        (93, 323),
        (103, 332),
        (109, 338),
        (127, 356),
        (132, 361),
        (136, 365),
        (148, 377),
        (149, 378),
        (150, 379),
        (162, 389),
        (172, 397),
        (176, 400),
        (234, 454),
    ),
    "eyes": (
        (7, 249),
        (33, 263),
        (46, 276),
        (52, 282),
        (53, 283),
        (55, 285),
        (63, 293),
        (65, 295),
        (66, 296),
        (70, 300),
        (105, 334),
        (107, 336),
        (133, 362),
        (144, 373),
        (145, 374),
        (153, 380),
        (154, 381),
        (155, 382),
        (157, 384),
        (158, 385),
        (159, 386),
        (160, 387),
        (161, 388),
        (163, 390),
        (173, 398),
        (246, 466),
    ),
    "nose": (
        (45, 275),
        (48, 278),
        (64, 294),
        (97, 326),
        (98, 327),
        (115, 344),
        (220, 440),
    ),
    "mouth": (
        (37, 267),
        (39, 269),
        (74, 270),
        (76, 291),
        (78, 306),
        (81, 311),
        (82, 312),
        (84, 314),
        (87, 317),
        (88, 319),
        (90, 321),
        (178, 403),
        (181, 405),
    ),
}

#: `depth_spread` / `contour_depth_range` が読む、対応ペアで使う 126 点
#: （`POOLED_PAIRS` の sorted unique index）。478 点全部ではない
#: （notebook cell 6 の `_USED` と同じ導出）。
_POOLED_LANDMARK_INDICES: Final[tuple[int, ...]] = tuple(
    sorted({index for name in PART_NAMES for pair in _PART_PAIRS[name] for index in pair})
)
_POOLED_INDEX_ARRAY: Final[NDArray[np.int64]] = np.asarray(_POOLED_LANDMARK_INDICES, dtype=np.int64)
_POOLED_POSITION: Final[dict[int, int]] = {
    landmark: order for order, landmark in enumerate(_POOLED_LANDMARK_INDICES)
}
#: `contour_depth_range` が読む、126 点のうち輪郭部位に属する点の行位置
#: （notebook cell 6 の `_PART_ROWS["contour"]` と同じ導出）。
_CONTOUR_POOL_ROWS: Final[NDArray[np.int64]] = np.asarray(
    [_POOLED_POSITION[index] for pair in _PART_PAIRS["contour"] for index in pair],
    dtype=np.int64,
)

#: 顔の寸法・比率の分母がこれを下回ると「退化した顔」として扱う（notebook と同じ）。
_MIN_FACE_DIMENSION: Final[float] = 1e-6
#: 比率の分母を 0 割りから守るための下限（notebook の `max(..., 1e-6)` と同じ）。
_RATIO_EPSILON: Final[float] = 1e-6


@dataclass(frozen=True, slots=True)
class AgeGateFeatures:
    """41 次元の特徴ベクトル。並びは `FEATURE_NAMES` と同じ。"""

    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.vector) != len(FEATURE_NAMES):
            raise AgeGateError(
                f"age-gate の特徴ベクトルは {len(FEATURE_NAMES)} 次元である必要があります。"
            )

    def as_vector(self) -> tuple[float, ...]:
        return self.vector


@dataclass(frozen=True, slots=True)
class AgeGateDecision:
    """凍結された動作点で下した、1 回分の判定（§9.2, §10.11-8）。"""

    #: `probabilities[:, 1]`。生の確率はエクスポートへ記録しない（§9.2）が、
    #: この判定オブジェクト自体は表示可否を決めるために保持する。
    probability_in_range: float
    threshold: float
    in_range: bool
    spec_version: str


def decide(probability: float, *, threshold: float, spec_version: str) -> AgeGateDecision:
    """凍結された閾値で「範囲内」かどうかを決める。

    `probability > threshold` のときだけ範囲内とする（`>=` ではない）。
    境界そのもの（`probability == threshold`）は安全側（範囲外）へ倒す。
    """
    value = float(probability)
    limit = float(threshold)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AgeGateError(f"probability は 0〜1 の有限値である必要があります: {probability!r}")
    return AgeGateDecision(
        probability_in_range=value,
        threshold=limit,
        in_range=value > limit,
        spec_version=spec_version,
    )


def _geometry_features(points: NDArray[np.float64]) -> tuple[float, ...] | None:
    """notebook cell 6 の `geometry_features()` の移植（32 特徴）。

    退化した顔（`face_height` または `face_width` が `_MIN_FACE_DIMENSION` 以下）
    は例外ではなく `None` を返す。
    """

    def point(name: str) -> NDArray[np.float64]:
        return np.asarray(points[IDX[name]], dtype=np.float64)

    def dx(a: str, b: str) -> float:
        return float(abs(point(a)[0] - point(b)[0]))

    def dy(a: str, b: str) -> float:
        return float(abs(point(a)[1] - point(b)[1]))

    face_height = dy("forehead", "chin")
    face_width = dx("cheek_l", "cheek_r")
    if face_height <= _MIN_FACE_DIMENSION or face_width <= _MIN_FACE_DIMENSION:
        return None

    eye_y = 0.0  # 正規化の定義により両眼中点が原点。
    forehead_y = float(point("forehead")[1])
    chin_y = float(point("chin")[1])
    nasion_y = float(point("nasion")[1])
    subnasale_y = float(point("subnasale")[1])
    mouth_y = float((point("lip_top")[1] + point("lip_bottom")[1]) / 2.0)

    eye_width_left = dx("eye_l_out", "eye_l_in")
    eye_width_right = dx("eye_r_out", "eye_r_in")
    eye_open_left = dy("eye_l_top", "eye_l_bottom")
    eye_open_right = dy("eye_r_top", "eye_r_bottom")

    pooled_depth = points[_POOLED_INDEX_ARRAY, 2]
    contour_depth = pooled_depth[_CONTOUR_POOL_ROWS]

    values: dict[str, float] = {
        # --- 顔全体の寸法（両眼間距離を 1 とした比） ---
        "face_height": face_height,
        "face_width": face_width,
        "face_aspect": face_height / face_width,
        # --- 縦方向の配分。合計はおおよそ 1 になる ---
        "forehead_ratio": (eye_y - forehead_y) / face_height,
        "midface_ratio": (subnasale_y - nasion_y) / face_height,
        "lower_face_ratio": (chin_y - subnasale_y) / face_height,
        "chin_ratio": (chin_y - float(point("lip_bottom")[1])) / face_height,
        # --- 鼻 ---
        "nose_length_ratio": (float(point("nose_tip")[1]) - nasion_y) / face_height,
        "nose_width_ratio": dx("nose_wing_l", "nose_wing_r") / face_width,
        "nose_depth": float(abs(point("nose_tip")[2])),
        # --- 口 ---
        "mouth_width_ratio": dx("mouth_l", "mouth_r") / face_width,
        "mouth_position_ratio": (mouth_y - eye_y) / face_height,
        "lip_height_ratio": dy("lip_top", "lip_bottom") / face_height,
        # --- 輪郭 ---
        "jaw_width_ratio": dx("jaw_l", "jaw_r") / face_width,
        "cheek_height_ratio": (float(point("cheek_l")[1] + point("cheek_r")[1]) / 2.0 - eye_y)
        / face_height,
        "jaw_taper": dx("jaw_l", "jaw_r") / max(dx("cheek_l", "cheek_r"), _RATIO_EPSILON),
        # --- 目 ---
        "eye_width_left": eye_width_left,
        "eye_width_right": eye_width_right,
        "eye_open_left": eye_open_left,
        "eye_open_right": eye_open_right,
        "eye_aspect_left": eye_open_left / max(eye_width_left, _RATIO_EPSILON),
        "eye_aspect_right": eye_open_right / max(eye_width_right, _RATIO_EPSILON),
        "brow_gap_left": dy("brow_l", "eye_l_top"),
        "brow_gap_right": dy("brow_r", "eye_r_top"),
        # --- 目を基準にした縦位置 ---
        "eye_to_nose_ratio": (float(point("nose_tip")[1]) - eye_y) / face_height,
        "eye_to_mouth_ratio": (mouth_y - eye_y) / face_height,
        # --- 立体感（z は両眼中点を 0、両眼間距離を 1 とした奥行き） ---
        "depth_forehead": float(point("forehead")[2]),
        "depth_chin": float(point("chin")[2]),
        "depth_cheek_mean": float((point("cheek_l")[2] + point("cheek_r")[2]) / 2.0),
        "depth_spread": float(pooled_depth.std()),
        "contour_depth_range": float(contour_depth.max() - contour_depth.min()),
        "face_convexity": float(abs(point("nose_tip")[2] - point("chin")[2])),
    }
    return tuple(values[name] for name in GEOMETRY_FEATURE_NAMES)


def build_features(
    normalized_landmarks: NDArray[np.float64],
    *,
    pose: FacePose,
    symmetry_3d: SymmetryDistances,
) -> AgeGateFeatures | None:
    """41 次元の特徴ベクトルを組み立てる。

    ``normalized_landmarks`` は `domain.geometry.normalize_landmarks_3d` の出力
    （``(478, 3)``、両眼中点が原点・両眼間距離が 1）でなければならない。ここで
    正規化を再実装することはしない。

    NaN / Inf、退化した顔（``face_height`` または ``face_width`` が 1e-6 以下）、
    478 点でない入力、対称度の部位欠測は、例外ではなく欠測（``None``）を返す
    （不変条件 11・12）。入力配列は読むだけで変更しない。
    """
    points = np.asarray(normalized_landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] != FACE_LANDMARK_COUNT:
        return None
    if not np.isfinite(points).all():
        return None

    geometry = _geometry_features(points)
    if geometry is None:
        return None

    pose_values = (pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    if not all(math.isfinite(value) for value in pose_values):
        return None

    equal_weight = symmetry_3d.equal_weight_distance
    if equal_weight is None or not math.isfinite(equal_weight):
        return None
    # 部位別は `distance(name)`、総合は `overall_distance` から読む。**同じ
    # `distance()` で `"overall"` を引かない** —— 実物の `Symmetry3DResult` は
    # `splits`（部位別）しか引かないため常に `None` になり、その結果ここが必ず
    # 欠測を返して、ゲートが毎回 `UNDETERMINED`（= 参考年齢を出さない）へ倒れて
    # いた。stub だけがその名前を受け付けていたので、テストは通り続けていた。
    part_and_overall: list[float] = []
    for name in PART_NAMES:
        value = symmetry_3d.distance(name)
        if value is None or not math.isfinite(value):
            return None
        part_and_overall.append(value)
    overall = symmetry_3d.overall_distance
    if overall is None or not math.isfinite(overall):
        return None
    part_and_overall.append(overall)

    vector = (*geometry, *pose_values, float(equal_weight), *part_and_overall)
    if len(vector) != len(FEATURE_NAMES):
        return None
    return AgeGateFeatures(vector=vector)
