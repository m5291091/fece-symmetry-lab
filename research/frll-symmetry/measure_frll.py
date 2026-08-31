"""FRLL（Face Research Lab London Set）を symmetry-3d/1.0.0 で測る。

`research/utkface-symmetry/utkface_symmetry_lab.ipynb` と**同じ self-contained 移植**
を使う。domain を import しないのは Colab / ローカルのどちらでも同じコードが動くため
であり、`--self-test` が毎回 domain 実装との一致を確かめる（`symmetry_level3_probe.py`
と同じ規約）。

測る対象は 4 つ。

    M1  102 人の対称度分布と、撮影規約で揃えたはずの写真に残っている姿勢
    M2  検出器の左右バイアス（同じ画像を左右反転して測り直す）
    M3  部位別の split-half 信頼性（「鼻の欄は何を測っているか」）
    M4  人手 189 点（.tem）から独立に測った左右差との突き合わせ

出力は集計値と 1 画像 1 行の測定値だけで、**顔画像とランドマーク座標は書き出さない**
（不変条件 1）。

使い方:

    uv run python research/frll-symmetry/measure_frll.py --self-test
    uv run python research/frll-symmetry/measure_frll.py \
        --images "/path/to/neutral_front" \
        --info   "/path/to/london_faces_info.csv" \
        --out    data/frll

データ: Face Research Lab London Set, DeBruine & Jones (2017), figshare, CC BY 4.0
        https://doi.org/10.6084/m9.figshare.5047666
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# A. 指標定数：configs/metrics/symmetry-3d-v1.json と同一
# ---------------------------------------------------------------------------

METRIC_VERSION = "symmetry-3d/1.0.0"
LANDMARK_MAP_VERSION = "mediapipe-478-mirror/1.0.0"
LANDMARK_COUNT = 478
PART_NAMES = ("contour", "eyes", "nose", "mouth")
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (263, 362)
PART_PAIRS = {
    "contour": (
        (21, 251), (54, 284), (58, 288), (67, 297), (93, 323),
        (103, 332), (109, 338), (127, 356), (132, 361), (136, 365),
        (148, 377), (149, 378), (150, 379), (162, 389), (172, 397),
        (176, 400), (234, 454),
    ),
    "eyes": (
        (7, 249), (33, 263), (46, 276), (52, 282), (53, 283),
        (55, 285), (63, 293), (65, 295), (66, 296), (70, 300),
        (105, 334), (107, 336), (133, 362), (144, 373),
        (145, 374), (153, 380), (154, 381), (155, 382),
        (157, 384), (158, 385), (159, 386), (160, 387),
        (161, 388), (163, 390), (173, 398), (246, 466),
    ),
    "nose": (
        (45, 275), (48, 278), (64, 294), (97, 326), (98, 327),
        (115, 344), (220, 440),
    ),
    "mouth": (
        (37, 267), (39, 269), (74, 270), (76, 291), (78, 306),
        (81, 311), (82, 312), (84, 314), (87, 317), (88, 319),
        (90, 321), (178, 403), (181, 405),
    ),
}
POOLED_PAIRS = tuple(pair for name in PART_NAMES for pair in PART_PAIRS[name])
SERIES = (*PART_NAMES, "overall")

DISPLAY_SCALE_VERSION = "display-scale/1.0.0"
DISTANCE_AT_FULL_SCORE = 0.010
DISTANCE_AT_ZERO_SCORE = 0.050
MIN_EYE_DISTANCE = 1e-9

# 事前計算（全画像で不変）
_USED = sorted({index for pair in POOLED_PAIRS for index in pair})
_USED_ARR = np.asarray(_USED, dtype=np.int64)
_POSITION = {landmark: order for order, landmark in enumerate(_USED)}
_SWAP = np.arange(len(_USED))
for _left, _right in POOLED_PAIRS:
    _SWAP[_POSITION[_left]], _SWAP[_POSITION[_right]] = _POSITION[_right], _POSITION[_left]
_PART_ROWS = {
    name: np.asarray(
        [_POSITION[index] for pair in PART_PAIRS[name] for index in pair], dtype=np.int64
    )
    for name in PART_NAMES
}
# split-half：ペアを交互に 2 群へ分ける（部位内で偏らせない）
_PART_ROWS_HALF = {}
for name in PART_NAMES:
    pairs = PART_PAIRS[name]
    for half, take in (("a", 0), ("b", 1)):
        rows = [
            _POSITION[index]
            for order, pair in enumerate(pairs)
            if order % 2 == take
            for index in pair
        ]
        _PART_ROWS_HALF[(name, half)] = np.asarray(rows, dtype=np.int64)
_LEFT2 = list(LEFT_EYE_CORNERS)
_RIGHT2 = list(RIGHT_EYE_CORNERS)


# ---------------------------------------------------------------------------
# B. 指標本体：domain/geometry.py・domain/symmetry_3d.py と同一
# ---------------------------------------------------------------------------


def normalize_landmarks_3d(points: np.ndarray) -> np.ndarray:
    """geometry.normalize_landmarks_3d と同一。スケールは 3D の両眼間距離。"""
    array = np.asarray(points, dtype=np.float64)
    left2 = array[_LEFT2, :2].mean(axis=0)
    right2 = array[_RIGHT2, :2].mean(axis=0)
    delta = right2 - left2
    distance2d = float(np.hypot(delta[0], delta[1]))
    if distance2d <= MIN_EYE_DISTANCE:
        raise ValueError("eye distance too small")
    cosine = float(delta[0]) / distance2d
    sine = -float(delta[1]) / distance2d
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    center = (left2 + right2) / 2.0
    left3 = array[_LEFT2].mean(axis=0)
    right3 = array[_RIGHT2].mean(axis=0)
    scale = float(np.linalg.norm(right3 - left3))
    if scale <= MIN_EYE_DISTANCE:
        raise ValueError("eye distance too small")
    plane = ((array[:, :2] - center) @ rotation.T) / scale
    depth = (array[:, 2] - float(left3[2] + right3[2]) / 2.0) / scale
    return np.column_stack([plane, depth])


def axis_split(residual: np.ndarray) -> dict:
    """symmetry_3d.axis_split と同一。distance² = plane² + depth² が恒等に成り立つ。"""
    return {
        "distance": float(np.sqrt((residual**2).sum(axis=1).mean())),
        "plane": float(np.sqrt((residual[:, :2] ** 2).sum(axis=1).mean())),
        "depth": float(np.sqrt((residual[:, 2] ** 2).mean())),
    }


def object_residual(points: np.ndarray) -> np.ndarray:
    """鏡映 → ペア名入替 → Procrustes 重ね合わせ → 残差 `(126, 3)`。

    **重ね合わせは全 126 点で 1 回だけ行う。**部位別も split-half も、この同じ
    残差を行で読み替えたものである（§8.6.5）。
    """
    normalized = normalize_landmarks_3d(points)
    original = normalized[_USED_ARR]
    mirrored = original.copy()
    mirrored[:, 0] *= -1.0
    mirrored = mirrored[_SWAP]
    centred = original - original.mean(axis=0)
    mirrored_centred = mirrored - mirrored.mean(axis=0)
    left_vectors, _, right_vectors = np.linalg.svd(mirrored_centred.T @ centred)
    rotation = left_vectors @ right_vectors
    if np.linalg.det(rotation) < 0:
        left_vectors[:, -1] *= -1.0
        rotation = left_vectors @ right_vectors
    return centred - (mirrored_centred @ rotation)


def compute_scores_3d(points: np.ndarray) -> dict:
    """478 点そろった顔 1 つ分。compute_symmetry_3d + equal_weight_distance と同一。"""
    residual = object_residual(points)
    result = {name: axis_split(residual[_PART_ROWS[name]]) for name in PART_NAMES}
    result["overall"] = axis_split(residual)
    result["s_3d"] = math.sqrt(
        sum(result[name]["distance"] ** 2 for name in PART_NAMES) / len(PART_NAMES)
    )
    for name in PART_NAMES:
        for half in ("a", "b"):
            rows = _PART_ROWS_HALF[(name, half)]
            result[f"half_{name}_{half}"] = axis_split(residual[rows])["distance"]
    return result


def symmetry_percent(distance):
    """ui.readings.symmetry_percent と同一。"""
    if distance is None or not math.isfinite(distance):
        return None
    return min(100.0, max(0.0, (1.0 - distance) * 100.0))


def score_points(distance):
    """score_scale.score_points と同一。"""
    if distance is None or not math.isfinite(distance):
        return None
    span = DISTANCE_AT_ZERO_SCORE - DISTANCE_AT_FULL_SCORE
    points = 100.0 * (DISTANCE_AT_ZERO_SCORE - distance) / span
    return min(100.0, max(0.0, points))


def pose_degrees(matrix: np.ndarray):
    """pose.pose_from_matrix と同一の yaw/pitch/roll（度）。"""
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    left, singular_values, right = np.linalg.svd(linear)
    if singular_values.min() <= 1e-7:
        raise ValueError("pose matrix is degenerate")
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        raise ValueError("pose matrix contains reflection")
    sin_pitch = float(np.clip(-rotation[1, 2], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    cos_pitch = math.sqrt(max(0.0, 1.0 - sin_pitch * sin_pitch))
    if cos_pitch > 1e-7:
        yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
        roll = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    else:
        yaw = math.atan2(-float(rotation[2, 0]), float(rotation[0, 0]))
        roll = 0.0
    return tuple(math.degrees(value) for value in (yaw, pitch, roll))


# ---------------------------------------------------------------------------
# C. 統計の小道具（scipy を足さない）
# ---------------------------------------------------------------------------


def mean(values):
    return sum(values) / len(values)


def sd(values):
    if len(values) < 2:
        return float("nan")
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def corr(first, second):
    if len(first) < 3:
        return float("nan")
    mean_a, mean_b = mean(first), mean(second)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(first, second))
    denominator = math.sqrt(
        sum((a - mean_a) ** 2 for a in first) * sum((b - mean_b) ** 2 for b in second)
    )
    return numerator / denominator if denominator > 0 else float("nan")


def ols(design, target):
    """定数項つき重回帰。係数・R²・残差 SD を返す。"""
    width = len(design[0]) + 1
    normal = [[0.0] * width for _ in range(width)]
    right = [0.0] * width
    for row, value in zip(design, target):
        augmented = [1.0, *row]
        for i in range(width):
            right[i] += augmented[i] * value
            for j in range(width):
                normal[i][j] += augmented[i] * augmented[j]
    for i in range(width):
        pivot = max(range(i, width), key=lambda r: abs(normal[r][i]))
        normal[i], normal[pivot] = normal[pivot], normal[i]
        right[i], right[pivot] = right[pivot], right[i]
        if abs(normal[i][i]) < 1e-12:
            return None, float("nan"), float("nan")
        for r in range(i + 1, width):
            factor = normal[r][i] / normal[i][i]
            for c in range(i, width):
                normal[r][c] -= factor * normal[i][c]
            right[r] -= factor * right[i]
    coefficients = [0.0] * width
    for i in reversed(range(width)):
        total = right[i] - sum(normal[i][j] * coefficients[j] for j in range(i + 1, width))
        coefficients[i] = total / normal[i][i]
    predicted = [coefficients[0] + sum(c * x for c, x in zip(coefficients[1:], row)) for row in design]
    residual = [t - p for t, p in zip(target, predicted)]
    average = mean(target)
    total_ss = sum((t - average) ** 2 for t in target)
    r_squared = 1 - sum(e * e for e in residual) / total_ss if total_ss > 0 else float("nan")
    return coefficients, r_squared, sd(residual)


def spearman_brown(half_r):
    """split-half の相関を全長相当へ補正する。"""
    if not math.isfinite(half_r) or half_r <= -1:
        return float("nan")
    return 2 * half_r / (1 + half_r)


# ---------------------------------------------------------------------------
# D. FRLL の読み込み
# ---------------------------------------------------------------------------


def read_info(path: Path) -> dict:
    """london_faces_info.csv（CR 改行のことがある）を face_id で引ける形にする。"""
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    rows = list(csv.DictReader(io.StringIO(text)))
    table = {}
    for row in rows:
        face_id = (row.get("face_id") or "").strip()
        if not face_id:
            continue
        table[face_id] = {
            "face_age": row.get("face_age", "").strip(),
            "face_sex": (row.get("face_sex") or row.get("face_gender") or "").strip(),
            "face_eth": (row.get("face_eth") or "").strip(),
        }
    return table


def read_tem(path: Path) -> np.ndarray | None:
    """Psychomorph テンプレート。1 行目が点数、続く行が `x\\ty`。"""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return None
    try:
        count = int(lines[0].strip())
    except ValueError:
        return None
    points = []
    for line in lines[1 : 1 + count]:
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            return None
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            return None
    if len(points) != count:
        return None
    return np.asarray(points, dtype=np.float64)


def face_id_of(image_path: Path) -> str:
    """`001_03.jpg` → `001`。"""
    stem = image_path.stem
    return stem.split("_")[0]


# ---------------------------------------------------------------------------
# E. 検出（mediapipe はここでだけ使う）
# ---------------------------------------------------------------------------


class Detector:
    """FaceLandmarker を 1 つだけ作って使い回す。ゲートで弾かない（残存姿勢を測る）。"""

    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.IMAGE,
            num_faces=2,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def analyze(self, rgb: np.ndarray):
        """`('ok', 測定値)` か `(除外理由, None)`。姿勢ゲートは適用しない。"""
        mp = self._mp
        try:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            detection = self._landmarker.detect(image)
        except Exception:
            return ("landmark_detection_error", None)

        faces = list(detection.face_landmarks or [])
        if len(faces) == 0:
            return ("no_face", None)
        if len(faces) > 1:
            return ("multiple_faces", None)
        if len(faces[0]) != LANDMARK_COUNT:
            return ("unexpected_landmark_count", None)

        matrices = list(detection.facial_transformation_matrixes or [])
        if len(matrices) != 1:
            return ("pose_missing", None)
        try:
            yaw, pitch, roll = pose_degrees(np.asarray(matrices[0]))
        except (ValueError, np.linalg.LinAlgError):
            return ("pose_missing", None)

        height, width = rgb.shape[:2]
        points = np.asarray(
            [(p.x * width, p.y * height, p.z * width) for p in faces[0]], dtype=np.float64
        )
        if not np.isfinite(points).all():
            return ("symmetry_error", None)
        try:
            scores = compute_scores_3d(points)
        except (ValueError, np.linalg.LinAlgError):
            return ("symmetry_error", None)

        record = {
            "image_width": int(width),
            "image_height": int(height),
            "yaw_deg": float(yaw),
            "pitch_deg": float(pitch),
            "roll_deg": float(roll),
            "s3d_distance": float(scores["s_3d"]),
            "score_points": score_points(scores["s_3d"]),
        }
        for name in SERIES:
            record[f"distance_{name}"] = scores[name]["distance"]
            record[f"plane_{name}"] = scores[name]["plane"]
            record[f"depth_{name}"] = scores[name]["depth"]
            record[f"symmetry_percent_{name}"] = symmetry_percent(scores[name]["distance"])
        for name in PART_NAMES:
            record[f"half_{name}_a"] = scores[f"half_{name}_a"]
            record[f"half_{name}_b"] = scores[f"half_{name}_b"]
        if any(value is None or not math.isfinite(value) for value in record.values()):
            return ("non_finite_score", None)
        return ("ok", record)


def load_rgb(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# F. M4 — 人手 189 点の鏡映対応を経験的に導出する
# ---------------------------------------------------------------------------


def _align_by_pupils(points: np.ndarray) -> np.ndarray:
    """点 0 / 1（左右の瞳）で並進・回転・スケールを揃える。"""
    left, right = points[0], points[1]
    delta = right - left
    scale = float(np.hypot(delta[0], delta[1]))
    if scale <= MIN_EYE_DISTANCE:
        raise ValueError("pupil distance too small")
    cosine, sine = float(delta[0]) / scale, -float(delta[1]) / scale
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    center = (left + right) / 2.0
    return ((points - center) @ rotation.T) / scale


def derive_mirror_map(templates: list[np.ndarray]):
    """平均テンプレートを鏡映して最近傍対応を取り、対合であることを検証する。

    返り値は `(map, 診断)`。検証に落ちたら `map` は `None`。**通らなかったら
    使わない**（推測で埋めない）。
    """
    aligned = []
    for template in templates:
        try:
            aligned.append(_align_by_pupils(template))
        except ValueError:
            continue
    if len(aligned) < 10:
        return None, {"status": "too_few_templates", "count": len(aligned)}
    average = np.mean(np.stack(aligned), axis=0)

    mirrored = average.copy()
    mirrored[:, 0] *= -1.0
    distances = np.linalg.norm(average[:, None, :] - mirrored[None, :, :], axis=2)
    mapping = distances.argmin(axis=1)
    match_distance = distances[np.arange(len(average)), mapping]

    # 対合性：map(map(i)) == i
    involution = bool(np.array_equal(mapping[mapping], np.arange(len(average))))
    # 分離度：対応先までの距離 < 2 番目に近い点までの距離
    ordered = np.sort(distances, axis=1)
    separation = float(np.min(ordered[:, 1] - ordered[:, 0])) if distances.shape[1] > 1 else 0.0
    spacing = float(np.median(np.sort(
        np.linalg.norm(average[:, None, :] - average[None, :, :], axis=2) + np.eye(len(average)) * 1e9,
        axis=1,
    )[:, 0]))
    worst = float(match_distance.max())

    diagnostics = {
        "status": "ok" if involution and worst < spacing else "rejected",
        "involution": involution,
        "worst_match_distance_iod": worst,
        "median_nearest_point_spacing_iod": spacing,
        "min_separation_iod": separation,
        "self_paired_points": int(np.sum(mapping == np.arange(len(average)))),
    }
    if diagnostics["status"] != "ok":
        return None, diagnostics
    return mapping, diagnostics


def tem_object_symmetry(points: np.ndarray, mapping: np.ndarray) -> float:
    """人手 2D 点の object symmetry 残差 RMS（両瞳間距離 = 1 で正規化）。"""
    aligned = _align_by_pupils(points)
    mirrored = aligned.copy()
    mirrored[:, 0] *= -1.0
    mirrored = mirrored[mapping]
    centred = aligned - aligned.mean(axis=0)
    mirrored_centred = mirrored - mirrored.mean(axis=0)
    left_vectors, _, right_vectors = np.linalg.svd(mirrored_centred.T @ centred)
    rotation = left_vectors @ right_vectors
    if np.linalg.det(rotation) < 0:
        left_vectors[:, -1] *= -1.0
        rotation = left_vectors @ right_vectors
    residual = centred - (mirrored_centred @ rotation)
    return float(np.sqrt((residual**2).sum(axis=1).mean()))


# ---------------------------------------------------------------------------
# G. self-test — domain 実装と一致するか
# ---------------------------------------------------------------------------


def self_test() -> int:
    """`domain.symmetry_3d.compute_symmetry_3d` と同じ数字を出すか確かめる。

    合成顔は**非対称かつ奥行きが平坦でない**ものを使う。「常に 0 を返す実装」が
    通ってしまう入力では意味がない。
    """
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    try:
        from face_symmetry_lab.adapters.io.metric_config import load_metric_config
        from face_symmetry_lab.domain.symmetry_3d import compute_symmetry_3d
    except ImportError as error:
        print(f"NG: domain を import できない: {error}")
        return 1

    config = load_metric_config(root / "configs" / "metrics" / "symmetry-3d-v1.json")
    if config.metric_version != METRIC_VERSION:
        print(f"NG: metric_version 不一致 {config.metric_version} != {METRIC_VERSION}")
        return 1

    rng = np.random.default_rng(20260825)
    worst = 0.0
    for trial in range(20):
        base = rng.normal(0.0, 60.0, size=(LANDMARK_COUNT, 3))
        base[:, 2] *= 0.4
        base[:, 0] += 320.0
        base[:, 1] += 240.0
        base[:, 2] += rng.normal(0.0, 8.0, size=LANDMARK_COUNT)
        # 眼アンカーが潰れないよう最低限の配置を与える
        base[LEFT_EYE_CORNERS[0]] = [260.0, 230.0, 5.0]
        base[LEFT_EYE_CORNERS[1]] = [290.0, 232.0, 3.0]
        base[RIGHT_EYE_CORNERS[0]] = [400.0, 228.0, 6.0]
        base[RIGHT_EYE_CORNERS[1]] = [370.0, 231.0, 4.0]

        mine = compute_scores_3d(base)
        theirs = compute_symmetry_3d(base, config)
        if theirs.overall is None:
            print(f"NG: trial {trial} で domain 側が欠測を返した")
            return 1
        checks = [("overall", mine["overall"]["distance"], theirs.overall_distance)]
        checks += [
            (name, mine[name]["distance"], theirs.distance(name)) for name in PART_NAMES
        ]
        checks.append(("s_3d", mine["s_3d"], theirs.equal_weight_distance))
        for label, a, b in checks:
            if b is None:
                print(f"NG: trial {trial} の {label} が domain 側で欠測")
                return 1
            worst = max(worst, abs(a - b))
            if abs(a - b) > 1e-9:
                print(f"NG: trial {trial} の {label} が不一致 {a!r} != {b!r}")
                return 1

    # split-half の行集合が部位の行集合を過不足なく二分しているか
    for name in PART_NAMES:
        rows_a = set(_PART_ROWS_HALF[(name, "a")].tolist())
        rows_b = set(_PART_ROWS_HALF[(name, "b")].tolist())
        whole = set(_PART_ROWS[name].tolist())
        if rows_a & rows_b or rows_a | rows_b != whole:
            print(f"NG: {name} の split-half が部位の行を二分していない")
            return 1

    print(f"OK: domain 実装と一致（20 試行・最大差 {worst:.3e}）")
    print(f"    metric_version={METRIC_VERSION} landmark_map_version={LANDMARK_MAP_VERSION}")
    return 0


# ---------------------------------------------------------------------------
# H. 本体
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args) -> int:
    images_dir = Path(args.images).expanduser()
    info_path = Path(args.info).expanduser()
    out_dir = Path(args.out).expanduser()
    model_path = Path(args.model).expanduser()

    for path, label in ((images_dir, "--images"), (info_path, "--info"), (model_path, "--model")):
        if not path.exists():
            print(f"NG: {label} が見つからない: {path}")
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    info = read_info(info_path)
    images = sorted(p for p in images_dir.glob("*.jpg"))
    if not images:
        print(f"NG: {images_dir} に .jpg がない")
        return 1
    print(f"画像 {len(images)} 枚 / 属性 {len(info)} 件")

    detector = Detector(model_path)
    records = []
    excluded = {}
    flips = []
    templates = []
    template_ids = []

    for index, image_path in enumerate(images, start=1):
        face_id = face_id_of(image_path)
        rgb = load_rgb(image_path)
        status, record = detector.analyze(rgb)
        if status != "ok":
            excluded[status] = excluded.get(status, 0) + 1
            continue
        attributes = info.get(face_id, {})
        record = {
            "face_id": face_id,
            "filename": image_path.name,
            "face_age": attributes.get("face_age", ""),
            "face_sex": attributes.get("face_sex", ""),
            "face_eth": attributes.get("face_eth", ""),
            **record,
        }

        # M2: 左右反転して測り直す
        flip_status, flipped = detector.analyze(np.ascontiguousarray(rgb[:, ::-1, :]))
        if flip_status == "ok":
            record["flip_ok"] = 1
            for name in SERIES:
                record[f"flip_symmetry_percent_{name}"] = flipped[f"symmetry_percent_{name}"]
            record["flip_yaw_deg"] = flipped["yaw_deg"]
            flips.append((record, flipped))
        else:
            record["flip_ok"] = 0

        tem_points = read_tem(image_path.with_suffix(".tem"))
        if tem_points is not None and len(tem_points) >= 2:
            templates.append(tem_points)
            template_ids.append(face_id)
        records.append(record)
        if index % 20 == 0 or index == len(images):
            print(f"  {index}/{len(images)} 解析済み")

    if not records:
        print("NG: 1 枚も解析できなかった")
        return 1

    report = analyze(records, flips, templates, template_ids)

    # --- 出力 -------------------------------------------------------------
    fieldnames = list(records[0].keys())
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    records_path = out_dir / "frll_symmetry3d_records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    summary = {
        "schema_version": "frll-symmetry-3d/1.0.0",
        "status": "exploratory_descriptive_statistics_not_a_norm_or_target",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Face Research Lab London Set",
            "doi": "10.6084/m9.figshare.5047666",
            "subset": "neutral_front",
            "license": "CC BY 4.0",
            "citation": "DeBruine, L., & Jones, B. (2017). Face Research Lab London Set. figshare.",
        },
        "reproducibility": {
            "implementation": "self-contained script (ported from symmetry-3d/1.0.0)",
            "metric_version": METRIC_VERSION,
            "landmark_map_version": LANDMARK_MAP_VERSION,
            "display_scale_version": DISPLAY_SCALE_VERSION,
            "landmark_model_sha256": sha256_of(model_path),
            "landmark_delegate": "CPU",
            "pose_gate": "not applied (residual pose is what we are measuring)",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "counts": {
            "images": len(images),
            "analyzed": len(records),
            "excluded": excluded,
            "flip_analyzed": len(flips),
            "templates_read": len(templates),
        },
        **report,
        "warning": (
            "FRLL と固定した処理条件に限定した記述統計であり、探索値である。"
            "正常値・目標値・個人評価には使用できない。"
        ),
    }
    summary_path = out_dir / "frll_symmetry3d_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n書き出し: {records_path}\n          {summary_path}")
    return 0


def analyze(records, flips, templates, template_ids) -> dict:
    """M1〜M4 を集計し、標準出力へも要約を出す。"""
    out = {}
    line = "-" * 72

    # --- M1 対称度分布と残存姿勢 -----------------------------------------
    print(f"\n{line}\nM1  対称度分布と残存姿勢（n={len(records)}）\n{line}")
    poses = {axis: [abs(r[f"{axis}_deg"]) for r in records] for axis in ("yaw", "pitch", "roll")}
    print(f"{'姿勢(絶対値)':<14}{'中央値':>9}{'p90':>9}{'最大':>9}")
    pose_stats = {}
    for axis, values in poses.items():
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        p90 = ordered[int(len(ordered) * 0.9)]
        pose_stats[axis] = {"median": median, "p90": p90, "max": max(values)}
        print(f"{axis:<14}{median:9.3f}{p90:9.3f}{max(values):9.3f}")
    out["pose_residual_deg"] = pose_stats

    design = [[abs(r["yaw_deg"]), abs(r["pitch_deg"]), abs(r["roll_deg"])] for r in records]
    print(f"\n{'部位':<9}{'SD(pt)':>9}{'平均':>9}{'r|yaw|':>9}{'姿勢R2':>9}{'残差SD':>9}")
    distribution = {}
    for name in SERIES:
        values = [r[f"symmetry_percent_{name}"] for r in records]
        coefficients, r_squared, residual_sd = ols(design, values)
        yaw_r = corr([row[0] for row in design], values)
        distribution[name] = {
            "sd": sd(values),
            "mean": mean(values),
            "corr_abs_yaw": yaw_r,
            "pose_r_squared": r_squared,
            "residual_sd": residual_sd,
            "coef_yaw_pt_per_deg": coefficients[1] if coefficients else None,
            "coef_pitch_pt_per_deg": coefficients[2] if coefficients else None,
            "coef_roll_pt_per_deg": coefficients[3] if coefficients else None,
        }
        print(
            f"{name:<9}{sd(values):9.3f}{mean(values):9.2f}"
            f"{yaw_r:9.3f}{r_squared:9.3f}{residual_sd:9.3f}"
        )
    out["distribution"] = distribution

    # --- M2 検出器の左右バイアス ----------------------------------------
    print(f"\n{line}\nM2  検出器の左右バイアス（同じ画像を左右反転・n={len(flips)}）\n{line}")
    print("完全な検出器なら反転しても対称度は変わらない。差はすべて検出器側の非対称。")
    bias = {}
    if flips:
        print(f"\n{'部位':<9}{'平均|Δ|':>10}{'中央値|Δ|':>11}{'最大|Δ|':>10}{'人物間SD':>10}{'比':>8}")
        for name in SERIES:
            deltas = [abs(r[f"symmetry_percent_{name}"] - f[f"symmetry_percent_{name}"])
                      for r, f in flips]
            between = sd([r[f"symmetry_percent_{name}"] for r, _ in flips])
            ordered = sorted(deltas)
            ratio = mean(deltas) / between if between > 0 else float("nan")
            bias[name] = {
                "mean_abs_delta": mean(deltas),
                "median_abs_delta": ordered[len(ordered) // 2],
                "max_abs_delta": max(deltas),
                "between_person_sd": between,
                "ratio_to_between_person_sd": ratio,
            }
            print(
                f"{name:<9}{mean(deltas):10.3f}{ordered[len(ordered)//2]:11.3f}"
                f"{max(deltas):10.3f}{between:10.3f}{ratio:8.2f}"
            )
        yaw_sum = [abs(r["yaw_deg"] + f["yaw_deg"]) for r, f in flips]
        bias["yaw_sign_flip_residual_deg"] = {"mean": mean(yaw_sum), "max": max(yaw_sum)}
        print(f"\n反転で yaw の符号が反転するか（|yaw + yaw_flip|）: 平均 {mean(yaw_sum):.3f}° / 最大 {max(yaw_sum):.3f}°")
    out["detector_lr_bias"] = bias

    # --- M3 split-half 信頼性 -------------------------------------------
    print(f"\n{line}\nM3  部位別 split-half 信頼性（その欄は一貫した何かを測っているか）\n{line}")
    print("同じ部位のペアを半分ずつに分け、102 人で相関を取る。")
    print("低ければ、その欄が動く理由は顔の形ではなく、点ごとに独立な雑音である。")
    reliability = {}
    print(f"\n{'部位':<9}{'ペア数':>7}{'r(half)':>10}{'r(補正後)':>11}")
    for name in PART_NAMES:
        half_a = [r[f"half_{name}_a"] for r in records]
        half_b = [r[f"half_{name}_b"] for r in records]
        half_r = corr(half_a, half_b)
        full_r = spearman_brown(half_r)
        reliability[name] = {
            "pairs": len(PART_PAIRS[name]),
            "split_half_r": half_r,
            "spearman_brown_r": full_r,
        }
        print(f"{name:<9}{len(PART_PAIRS[name]):7}{half_r:10.3f}{full_r:11.3f}")
    out["split_half_reliability"] = reliability

    # --- M4 人手 189 点との突き合わせ -----------------------------------
    print(f"\n{line}\nM4  人手 189 点（.tem）との突き合わせ\n{line}")
    tem_block = {"templates": len(templates)}
    if len(templates) >= 10:
        mapping, diagnostics = derive_mirror_map(templates)
        tem_block["mirror_map"] = diagnostics
        print(f"鏡映対応の導出: {diagnostics['status']}")
        print(f"  対合性 {diagnostics['involution']} / 最悪の対応距離 {diagnostics['worst_match_distance_iod']:.4f} IOD"
              f" / 点間隔の中央値 {diagnostics['median_nearest_point_spacing_iod']:.4f} IOD")
        if mapping is None:
            print("  → 検証に落ちたので .tem からの左右差は計算しない（推測で埋めない）")
        else:
            by_id = {r["face_id"]: r for r in records}
            human, machine = [], []
            for face_id, template in zip(template_ids, templates):
                record = by_id.get(face_id)
                if record is None:
                    continue
                try:
                    human_distance = tem_object_symmetry(template, mapping)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                human.append(human_distance * 100.0)
                machine.append(record["plane_overall"] * 100.0)
            if len(human) >= 10:
                r_value = corr(human, machine)
                tem_block["human_vs_machine"] = {
                    "n": len(human),
                    "corr": r_value,
                    "human_mean_pt": mean(human),
                    "human_sd_pt": sd(human),
                    "machine_plane_mean_pt": mean(machine),
                    "machine_plane_sd_pt": sd(machine),
                }
                print(f"\n  n={len(human)}  人手 {mean(human):.3f}±{sd(human):.3f}pt"
                      f"  /  MediaPipe(面内) {mean(machine):.3f}±{sd(machine):.3f}pt")
                print(f"  相関 r = {r_value:.3f}")
                print("  低ければ、機械が測っている左右差は人手が見ている左右差と別物である。")
    else:
        print("  .tem が足りない")
    out["tem_cross_check"] = tem_block
    print()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FRLL を symmetry-3d/1.0.0 で測る")
    parser.add_argument("--self-test", action="store_true", help="domain 実装との一致だけ確かめる")
    parser.add_argument("--images", default="", help="neutral_front ディレクトリ")
    parser.add_argument("--info", default="", help="london_faces_info.csv")
    parser.add_argument("--out", default="data/frll", help="出力先ディレクトリ")
    parser.add_argument(
        "--model",
        default="models/face_landmarker.task",
        help="Face Landmarker のモデルファイル",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.images or not args.info:
        parser.error("--images と --info が要る（または --self-test）")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
