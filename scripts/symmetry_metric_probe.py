"""対称度指標の感度・分解能を測る診断スクリプト（docs/symmetry-metric-analysis.md の裏付け）。

`domain/symmetry.py` と凍結された metric bundle をそのまま読み込み、完全に
左右対称な合成顔へ既知量のズレを注入して、表示値がどう動くかを測る。実写も
モデルも不要なので、CI で回して指標の感度が退行していないことを確認できる。

    uv run python scripts/symmetry_metric_probe.py
    uv run python scripts/symmetry_metric_probe.py --section E --section J   # 一部だけ

合成顔は「正しい顔」ではなく、**ペア構造を持った既知の基準物**である。目的は
顔の再現ではなく「d をどれだけ動かすのに何 mm 必要か」を測ることなので、点の
配置が実顔と多少違っても結論は変わらない。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.domain.geometry import (
    fit_midline,
    normalize_landmarks,
    reflect_across_midline,
)
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig
from face_symmetry_lab.domain.symmetry import compute_symmetry

CONFIG: MetricConfig = load_metric_config()

#: 成人の両眼間距離のおよその値。d は「両眼間距離 = 1」で正規化されているので、
#: これを掛けると mm 換算になる。実測 IPD があればそちらを使うこと。
IPD_MM: float = 63.0

#: Prog Orthod 2015 (3D stereophotogrammetry, n=28/27) の顔全体 RMS 偏差。
#: 「対称」と「非対称」を分ける差が、いかに狭いかを示す基準として使う。
NORMAL_RMS_MM: float = 1.52
CLINICAL_RMS_MM: float = 2.85

#: MediaPipe Face Mesh 論文 (arXiv:1907.06724) の IOD 正規化 MAD。
#: 論文の IOD の定義（目頭・目尻の中点どうしの距離）は §8.3 の正規化スケールと
#: 同じなので、d と直接比較できる。
MEDIAPIPE_FULL_MAD: float = 0.0396
HUMAN_ANNOTATOR_MAD: float = 0.0256
MEDIAPIPE_LIGHT_MAD: float = 0.0515


# --------------------------------------------------------------------------
# 合成した基準の顔
# --------------------------------------------------------------------------
def build_symmetric_face() -> NDArray[np.float64]:
    """厳密に左右対称な landmark 集合。単位は両眼間距離。

    各ペア ``(left, right)`` を ``(-x, y)`` / ``(x, y)`` へ置くので、注入した
    ズレだけが d に現れる。パーツの配置は実顔のおよその比率に合わせてあるが、
    感度の測定に効くのは「注入量 ÷ 両眼間距離」だけである。
    """
    points = np.full((CONFIG.landmark_count, 2), np.nan, dtype=np.float64)

    def place(
        pairs: tuple[tuple[int, int], ...],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        for index, (left, right) in enumerate(pairs):
            t = (index + 0.5) / len(pairs)
            x = x_range[0] + (x_range[1] - x_range[0]) * t
            y = y_range[0] + (y_range[1] - y_range[0]) * np.sin(np.pi * t)
            points[left] = (-x, y)
            points[right] = (x, y)

    place(CONFIG.parts["eyes"].pairs, (0.24, 0.76), (-0.10, 0.10))
    place(CONFIG.parts["contour"].pairs, (0.55, 1.10), (-0.45, 1.30))
    place(CONFIG.parts["nose"].pairs, (0.06, 0.30), (0.25, 0.68))
    place(CONFIG.parts["mouth"].pairs, (0.05, 0.42), (0.85, 1.08))

    midline = CONFIG.midline_indices
    for index, landmark in enumerate(midline):
        points[landmark] = (0.0, -0.70 + 2.05 * index / (len(midline) - 1))

    # 測定に入らない残りの点も、配列の形を保つために埋める。
    unset = np.isnan(points[:, 0])
    points[unset] = np.random.default_rng(0).normal(0.0, 0.5, size=(int(unset.sum()), 2))

    # 眼アンカーだけは正規化が意味を持つ位置へ厳密に置く。
    points[CONFIG.left_eye_corners[0]] = (-0.55, 0.0)
    points[CONFIG.left_eye_corners[1]] = (-0.45, 0.0)
    points[CONFIG.right_eye_corners[0]] = (0.55, 0.0)
    points[CONFIG.right_eye_corners[1]] = (0.45, 0.0)
    return points


BASE: NDArray[np.float64] = build_symmetric_face()
LEFT_INDICES = [left for left, _ in CONFIG.all_pairs]
RIGHT_INDICES = [right for _, right in CONFIG.all_pairs]


def symmetry_percent(distance: float | None) -> str:
    """UI の `format_symmetry` と同じ読み替え。"""
    if distance is None:
        return "  ——  "
    return f"{min(100.0, max(0.0, (1.0 - distance) * 100.0)):6.2f}%"


def overall(points: NDArray[np.float64]) -> float:
    result = compute_symmetry(points, CONFIG)
    assert result.overall is not None
    return result.overall


def report(label: str, points: NDArray[np.float64]) -> None:
    result = compute_symmetry(points, CONFIG)
    parts = "  ".join(f"{name}={symmetry_percent(result.distances[name])}" for name in PART_NAMES)
    print(
        f"  {label:<38} 全体={symmetry_percent(result.overall)} (d={result.overall:.4f})   {parts}"
    )


def mm(value_mm: float) -> float:
    """mm を両眼間距離の単位へ。"""
    return value_mm / IPD_MM


def heading(letter: str, title: str) -> None:
    print()
    print("=" * 108)
    print(f"[{letter}] {title}")
    print("=" * 108)


def shifted(indices: list[int], delta_mm: tuple[float, float]) -> NDArray[np.float64]:
    points = BASE.copy()
    points[indices] = points[indices] + (mm(delta_mm[0]), mm(delta_mm[1]))
    return points


def part_indices(name: str, side: str = "both") -> list[int]:
    pairs = CONFIG.parts[name].pairs
    if side == "left":
        return [left for left, _ in pairs]
    if side == "right":
        return [right for _, right in pairs]
    return [index for pair in pairs for index in pair]


def yawed(degrees: float) -> NDArray[np.float64]:
    """完全対称な顔を yaw させたときの 2D 投影（顔を円筒で近似）。"""
    theta = np.deg2rad(degrees)
    points = BASE.copy()
    depth = 0.45 * np.cos(np.clip(points[:, 0] / 1.2, -1.0, 1.0) * np.pi / 2)
    points[:, 0] = points[:, 0] * np.cos(theta) + depth * np.sin(theta)
    return points


def pair_vectors(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """ペアごとのズレベクトル（正規化フレーム）。DA/FA 分解に使う。"""
    normalized = normalize_landmarks(
        points,
        left_eye_corners=CONFIG.left_eye_corners,
        right_eye_corners=CONFIG.right_eye_corners,
    )
    midline = fit_midline(normalized, CONFIG.midline_indices)
    mirrored = reflect_across_midline(normalized, midline)
    return normalized[LEFT_INDICES] - mirrored[RIGHT_INDICES]


# --------------------------------------------------------------------------
# 各セクション
# --------------------------------------------------------------------------
def section_a() -> None:
    heading("A", "出発点 — 完全に左右対称な顔")
    report("完全対称（数学上の上限）", BASE)
    print()
    print(f"  d の単位は両眼間距離。IPD {IPD_MM:.0f}mm とすると d=0.01 ≒ {IPD_MM * 0.01:.2f}mm。")


def section_b() -> None:
    heading("B", "感度 — 1 点だけ動かす（口角）")
    corner = CONFIG.parts["mouth"].pairs[-1][1]
    for delta in (1, 2, 3, 5, 8, 10):
        report(f"口角 1 点を {delta:>2}mm 外へ", shifted([corner], (delta, 0.0)))
    print()
    print("  → 口角が 1cm ずれた顔でも 98%。1 点のズレはほとんど数値に出ない。")


def section_c() -> None:
    heading("C", "感度 — 部位まるごと動かす")
    for name, label in (("mouth", "口"), ("nose", "鼻")):
        for delta in (1, 2, 3, 5, 8):
            report(f"{label}全体を {delta:>2}mm 横へ", shifted(part_indices(name), (delta, 0.0)))
        print()
    print("  → ここでようやく数値が動くが、鼻が 5mm 曲がるのは形成外科の水準。")


def section_d() -> None:
    heading("D", "信号の幅 — 文献実測値を現行スケールへ写す")
    literature = (
        ("対称と判定された成人", NORMAL_RMS_MM),
        ("非対称と判定された成人", CLINICAL_RMS_MM),
    )
    for label, value in literature:
        distance = mm(value)
        print(
            f"  {label:<26} RMS {value:.2f}mm → d={distance:.4f} → "
            f"対称度 {symmetry_percent(distance)}"
        )
    span = (CLINICAL_RMS_MM - NORMAL_RMS_MM) / IPD_MM * 100
    print()
    print(f"  → 「対称」と「非対称」を分ける差は、この表示でわずか {span:.2f} ポイント。")
    print("    0〜100% のうち、意味を持つのは 2 ポイントだけ。")


def section_e() -> None:
    heading("E", "雑音1 — 撮影姿勢（品質ゲートは yaw ±15° まで通す）")
    signal_span = (CLINICAL_RMS_MM - NORMAL_RMS_MM) / IPD_MM
    for degrees in (3, 5, 8, 12, 15):
        distance = overall(yawed(degrees))
        print(
            f"  完全対称な顔を yaw {degrees:>2}° で撮る → d={distance:.4f} → "
            f"対称度 {symmetry_percent(distance)}  "
            f"(mm換算 {distance * IPD_MM:.2f}mm / 信号全幅の {distance / signal_span:.1f} 倍)"
        )
    print()
    print("  → ゲートを通る姿勢のブレだけで、測りたい信号の 4 倍のノイズが入る。")


def section_f() -> None:
    heading("F", "雑音2 — landmark 検出誤差（MediaPipe 公表値）")
    print(
        f"  測りたい量: 対称な成人 {mm(NORMAL_RMS_MM) * 100:.2f}% IOD "
        f"〜 非対称な成人 {mm(CLINICAL_RMS_MM) * 100:.2f}% IOD"
    )
    print()
    print(f"  {'誤差の水準':<26} {'d の平均':>10} {'対称度':>10} {'95% 区間':>22}")
    print("  " + "-" * 74)
    for label, mad in (
        (f"人間アノテータ {HUMAN_ANNOTATOR_MAD:.2%}", HUMAN_ANNOTATOR_MAD),
        (f"MediaPipe Full {MEDIAPIPE_FULL_MAD:.2%}", MEDIAPIPE_FULL_MAD),
        (f"MediaPipe Light {MEDIAPIPE_LIGHT_MAD:.2%}", MEDIAPIPE_LIGHT_MAD),
    ):
        # 1 次元ガウスの MAD = sigma * sqrt(2/pi)、2 次元へ等方に配る。
        sigma = mad * np.sqrt(np.pi / 2) / np.sqrt(2)
        values = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            points = BASE.copy()
            points[LEFT_INDICES] += rng.normal(0.0, sigma, size=(len(LEFT_INDICES), 2))
            points[RIGHT_INDICES] += rng.normal(0.0, sigma, size=(len(RIGHT_INDICES), 2))
            values.append(overall(points))
        array = np.array(values)
        low, high = np.percentile(array, [2.5, 97.5])
        print(
            f"  {label:<26} {array.mean():10.4f} {symmetry_percent(float(array.mean())):>10} "
            f"  {symmetry_percent(high)} 〜 {symmetry_percent(low)}"
        )
    print()
    print("  → 顔が完全に対称でも、検出誤差だけで「臨床的に非対称」と同じ d が出る。")
    print("    ただしこの試算は誤差が landmark ごとに独立という仮定。実際の")
    print("    MediaPipe は対称なテンプレートに拘束されるため、逆に d を")
    print("    押し下げる（実物より対称に見せる）方向にも働く。どちらにせよ")
    print("    表示値の大半が顔の形ではなく検出器の性質でできている。")


def section_g() -> None:
    heading("G", "構造 — 何を測っていて、何を測っていないか")
    paired = {index for pair in CONFIG.all_pairs for index in pair}
    midline = set(CONFIG.midline_indices)
    print(
        f"  landmark {CONFIG.landmark_count} 点のうち、距離計算に使うのは "
        f"{len(paired)} 点（{len(CONFIG.all_pairs)} ペア）だけ"
    )
    print(f"  正中線 {len(midline)} 点とペア点の重なり: {sorted(midline & paired) or '空集合'}")
    print("    → 鼻筋・鼻先・人中・顎先そのものは一度も測っていない（基準線専用）")
    anchors = set(CONFIG.left_eye_corners + CONFIG.right_eye_corners)
    print(
        f"  眼アンカー {sorted(anchors)} と eyes ペアの重なり: "
        f"{sorted(anchors & {i for pair in CONFIG.parts['eyes'].pairs for i in pair})}"
    )
    print("    → 正規化で対称位置に固定される点を、対称性の測定にも使っている")
    print()
    print("  全体値へのパーツ寄与（ペア数のプールなので、ペア数がそのまま重みになる）")
    total = len(CONFIG.all_pairs)
    for name in PART_NAMES:
        count = len(CONFIG.parts[name].pairs)
        print(f"    {name:<10} {count:>3} ペア  →  {count / total:5.1%}")
    print()
    print("  同じ 5mm のズレでも、パーツによって全体への効き方が違う:")
    for name in PART_NAMES:
        points = shifted(part_indices(name, "right"), (5.0, 0.0))
        result = compute_symmetry(points, CONFIG)
        print(
            f"    {name:<10} 片側 5mm → 全体 {symmetry_percent(result.overall)}  "
            f"({name} 単体では {symmetry_percent(result.distances[name])})"
        )


def section_h() -> None:
    heading("H", "構造 — 測っていない正中線点が、測っている全てを動かす")
    lower_midline = [index for index in CONFIG.midline_indices if BASE[index][1] > 0.3]
    for delta in (2, 5):
        report(f"顎〜鼻先の正中線を {delta}mm 横へ", shifted(lower_midline, (delta, 0.0)))
    print()
    print("  → 正中線点は自分ではスコアを持たないのに、基準線を通じて")
    print("    全パーツのスコアを同時に動かす。レバーが効きすぎている。")


def section_i() -> None:
    heading("I", "集計 — RMS が同じでも中身は違う")
    case_a = shifted(part_indices("mouth"), (4.0, 0.0))
    case_b = BASE.copy()
    rng = np.random.default_rng(3)
    case_b[LEFT_INDICES] += rng.normal(0.0, mm(2.45), size=(len(LEFT_INDICES), 2))

    print(f"  {'':<28} {'全体':>10} {'中央値':>10} {'上位10%':>10} {'最大':>10}")
    print("  " + "-" * 72)
    for label, points in (("A: 口だけ 4mm ズレ", case_a), ("B: 全体に 2.45mm ばらつき", case_b)):
        distances = np.linalg.norm(pair_vectors(points), axis=1)
        print(
            f"  {label:<28} {symmetry_percent(overall(points)):>10} "
            f"{np.median(distances) * IPD_MM:9.2f}mm "
            f"{np.percentile(distances, 90) * IPD_MM:9.2f}mm "
            f"{distances.max() * IPD_MM:9.2f}mm"
        )
    print()
    print("  → 表示上は同じ値。中央値と最大値を並べるだけで「1 か所が大きい」のか")
    print("    「全体が緩い」のかが分かれる。ペアごとの距離はすでに計算済み。")

    print()
    print("  DA/FA 分解（形態計測学の標準）— DA に姿勢、FA に本来の左右差が乗る:")
    for label, points in (("A: 口だけ 4mm ズレ", case_a), ("B: 全体に 2.45mm ばらつき", case_b)):
        vectors = pair_vectors(points)
        directional = float(np.linalg.norm(vectors.mean(axis=0)))
        fluctuating = float(np.sqrt(((vectors - vectors.mean(axis=0)) ** 2).sum(axis=1).mean()))
        print(f"    {label:<28} DA={directional * IPD_MM:5.2f}mm  FA={fluctuating * IPD_MM:5.2f}mm")


def section_j() -> None:
    heading("J", "表示 — 測定値を変えずに、比べる相手だけを変える")
    rng = np.random.default_rng(7)
    population = np.clip(rng.lognormal(np.log(NORMAL_RMS_MM), 0.35, 3000) / IPD_MM, 0.002, None)
    noise_floor = overall(yawed(5.0))  # 撮影のばらつきを yaw 5° 相当とみなす

    samples = [
        ("かなり対称な人", float(np.percentile(population, 10))),
        ("平均的な人", float(np.percentile(population, 50))),
        ("やや非対称な人", float(np.percentile(population, 90))),
        ("臨床的に非対称", mm(CLINICAL_RMS_MM)),
    ]
    print(f"  {'':<18} {'現行 (1-d)%':>13} {'mm 換算':>11} {'percentile':>13} {'ノイズ比':>10}")
    print("  " + "-" * 72)
    for label, distance in samples:
        percentile = float((population < distance).mean() * 100)
        print(
            f"  {label:<18} {symmetry_percent(distance):>13} "
            f"{distance * IPD_MM:9.2f}mm {percentile:12.0f}位 "
            f"{distance / noise_floor:9.2f}x"
        )
    print()
    print("  → 現行の列は 4 人が 95〜98% に固まる。percentile 列は 10 位〜96 位まで開く。")
    print("    測定値 d は一切変えていない。変えたのは「何と比べるか」だけ。")


SECTIONS: dict[str, tuple[str, Callable[[], None]]] = {
    "A": ("出発点", section_a),
    "B": ("感度・1 点", section_b),
    "C": ("感度・部位", section_c),
    "D": ("信号の幅", section_d),
    "E": ("雑音・姿勢", section_e),
    "F": ("雑音・検出誤差", section_f),
    "G": ("構造・測定範囲", section_g),
    "H": ("構造・正中線", section_h),
    "I": ("集計・RMS", section_i),
    "J": ("表示・比較対象", section_j),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        action="append",
        choices=sorted(SECTIONS),
        help="実行するセクション（省略時は全部）",
    )
    arguments = parser.parse_args()
    selected = arguments.section or sorted(SECTIONS)

    print(f"metric  : {CONFIG.metric_version}")
    print(f"landmark: {CONFIG.landmark_map_version}")
    print(f"IPD     : {IPD_MM:.0f}mm （d × {IPD_MM:.0f} で mm 換算）")
    for letter in selected:
        SECTIONS[letter][1]()
    print()
    print("=" * 108)
    print("解説: docs/symmetry-metric-analysis.md")
    print("=" * 108)


if __name__ == "__main__":
    main()
