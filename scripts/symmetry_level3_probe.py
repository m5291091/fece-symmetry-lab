"""レベル3（測定を締める）がどこまで効くかを測る（docs/symmetry-metric-level3.md の裏付け）。

`symmetry_metric_probe.py` は 2D の合成顔で「指標の感度」を測った。こちらは
奥行きを持つ 3D 顔モデルを作り、姿勢・フレーム平均・3D 化という 3 つの対策が
それぞれどれだけ効くかを測る。

    uv run python scripts/symmetry_level3_probe.py
    uv run python scripts/symmetry_level3_probe.py --section 3   # 対策10 だけ

奥行きモデル `face_depth()` が本スクリプト唯一の仮定である。実測値におよそ
合わせてあるが、実写で確認する価値がある（docs のテストD の副産物で得られる）。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.symmetry import compute_symmetry

CONFIG: MetricConfig = load_metric_config()

IPD_MM: float = 63.0

#: Prog Orthod 2015 の顔全体 RMS 偏差。この 2 つの幅が「測りたい信号」。
NORMAL_D: float = 1.52 / IPD_MM
CLINICAL_D: float = 2.85 / IPD_MM
SIGNAL_SPAN: float = CLINICAL_D - NORMAL_D

#: MediaPipe Face Mesh Full の IOD 正規化 MAD (arXiv:1907.06724)。
#: 1 次元ガウスの MAD = sigma * sqrt(2/pi) なので、等方 2 次元へ配ると下の sigma。
MEDIAPIPE_FULL_MAD: float = 0.0396
LANDMARK_SIGMA: float = MEDIAPIPE_FULL_MAD * np.sqrt(np.pi / 2) / np.sqrt(2)


# --------------------------------------------------------------------------
# 奥行きを持つ 3D 顔モデル
# --------------------------------------------------------------------------
def face_depth(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """顔の奥行き z。原点は両眼中心、+z がカメラ側。単位は両眼間距離。

    頭の丸み（後ろへ引く二次曲面）と鼻の隆起の重ね合わせ。実顔の実測値に
    およそ合わせてある: 目尻 -20mm / 輪郭 -42mm / 鼻先 +19mm / 顎 -10mm。
    """
    head = -0.55 * x**2 - 0.15 * (y - 0.30) ** 2
    nose = 0.30 * np.exp(-((x / 0.25) ** 2 + ((y - 0.45) / 0.35) ** 2))
    return head + nose


def build_face_3d() -> NDArray[np.float64]:
    """厳密に左右対称な 3D landmark 集合。注入したズレだけが d に現れる。"""
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

    unset = np.isnan(points[:, 0])
    points[unset] = np.random.default_rng(0).normal(0.0, 0.5, size=(int(unset.sum()), 2))

    points[CONFIG.left_eye_corners[0]] = (-0.55, 0.0)
    points[CONFIG.left_eye_corners[1]] = (-0.45, 0.0)
    points[CONFIG.right_eye_corners[0]] = (0.55, 0.0)
    points[CONFIG.right_eye_corners[1]] = (0.45, 0.0)

    depth = face_depth(points[:, 0], points[:, 1])
    return np.column_stack([points, depth])


FACE3D: NDArray[np.float64] = build_face_3d()


def yaw_matrix(degrees: float) -> NDArray[np.float64]:
    theta = np.deg2rad(degrees)
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def project(points3d: NDArray[np.float64], yaw_deg: float = 0.0) -> NDArray[np.float64]:
    """頭を yaw させ、正射影で 2D にする（いまのパイプラインが見ている像）。"""
    return np.asarray((points3d @ yaw_matrix(yaw_deg).T)[:, :2], dtype=np.float64)


def distance_2d(points2d: NDArray[np.float64]) -> float:
    result = compute_symmetry(points2d, CONFIG)
    assert result.overall is not None
    return result.overall


def distance_3d(points3d: NDArray[np.float64]) -> float:
    """3D object symmetry 解析。剛体変換に対して不変。

    鏡映してペア名を入れ替えた配置を、元の配置へ Procrustes 重ね合わせする。
    「どの平面で折るか」を先に決めないので、正中線のあてはめが要らず、姿勢に
    依存しない。形態計測学の標準手法。
    """
    used = sorted({index for pair in CONFIG.all_pairs for index in pair})
    position = {landmark: order for order, landmark in enumerate(used)}
    original = points3d[used]

    mirrored = original.copy()
    mirrored[:, 0] *= -1.0
    swap = np.arange(len(used))
    for left, right in CONFIG.all_pairs:
        swap[position[left]], swap[position[right]] = position[right], position[left]
    mirrored = mirrored[swap]

    # 並進と回転のみ（スケールは §8.3 で正規化済みなので触らない）
    centred = original - original.mean(axis=0)
    mirrored_centred = mirrored - mirrored.mean(axis=0)
    left_vectors, _, right_vectors = np.linalg.svd(mirrored_centred.T @ centred)
    rotation = left_vectors @ right_vectors
    if np.linalg.det(rotation) < 0:
        left_vectors[:, -1] *= -1.0
        rotation = left_vectors @ right_vectors
    aligned = mirrored_centred @ rotation
    return float(np.sqrt(((centred - aligned) ** 2).sum(axis=1).mean()))


def percent(distance: float) -> str:
    return f"{min(100.0, max(0.0, (1.0 - distance) * 100.0)):6.2f}%"


def heading(number: str, title: str) -> None:
    print()
    print("=" * 100)
    print(f"【{number}】{title}")
    print("=" * 100)


# --------------------------------------------------------------------------
# 各セクション
# --------------------------------------------------------------------------
def section_1() -> None:
    heading("1", "姿勢が偽の非対称をつくる仕組み — 2·z·sinθ")
    print("""
  顔を角度 θ だけ横に向けると、奥行き z にある点は横へ z·sinθ ずれて写る。
  左右のペアは同じ奥行きを持つので、両方とも同じ向きにずれる。
  鏡に映すと向きが反転するので、差は 2·z·sinθ になる。

      左の点 (-x, z) →  -x·cosθ + z·sinθ
      右の点 (+x, z) →  +x·cosθ + z·sinθ
      右を鏡映       →  -x·cosθ - z·sinθ
      ─────────────────────────────────────
      差             =           2·z·sinθ    ← 顔の形と無関係に湧く

  x（横位置）の項は消え、z（奥行き）だけが残る。
""")
    print(f"  {'部位':<10} {'平均の奥行き z':>16} {'yaw 5° の偽ズレ':>18}")
    print("  " + "-" * 48)
    for name in ("contour", "eyes", "nose", "mouth"):
        indices = [index for pair in CONFIG.parts[name].pairs for index in pair]
        depth = float(np.abs(FACE3D[indices, 2]).mean())
        fake = 2 * depth * np.sin(np.deg2rad(5.0))
        print(f"  {name:<10} {depth * IPD_MM:13.1f}mm {fake * IPD_MM:15.2f}mm")
    print()
    print(f"  比較: 「対称な成人」と「非対称な成人」を分ける差 = {SIGNAL_SPAN * IPD_MM:.2f}mm")
    print("  → 輪郭は yaw 5° だけで、その差の 3.7 倍の偽ズレを出す。")


def section_2() -> None:
    heading("2", "対策9: yaw ゲートを絞る — どこまで絞れば足りるか")
    print(f"  {'yaw':>6} {'d':>10} {'表示':>10} {'mm換算':>10} {'信号全幅の何倍':>16}")
    print("  " + "-" * 58)
    for degrees in (0, 1, 2, 3, 5, 8, 15):
        distance = distance_2d(project(FACE3D, degrees))
        print(
            f"  {degrees:>4}° {distance:10.4f} {percent(distance):>10} "
            f"{distance * IPD_MM:8.2f}mm {distance / SIGNAL_SPAN:14.1f}x"
        )
    print()
    print("  → 信号と同じ大きさに収めるには ±1° が必要。人間には守れない。")
    print("  → 代案: ゲートを絞る代わりに、yaw から誤差バーを計算して併記する。")
    print("    偽ズレ ≈ 2 × RMS(z) × sin(yaw)。z は既知、yaw は pose.py が出す。")


def section_3() -> None:
    heading("3", "対策10: 複数フレーム平均 — 「何を」平均するかで結果が正反対")
    print("""
  d = ||左 - 鏡映(右)|| は距離なので必ず 0 以上。だから検出誤差は d を
  押し上げる方向にしか働かない（正のバイアス）。

    × d を平均      → 押し上がった値を平均するだけ。バイアスは消えない
    ○ 座標を平均    → 符号があるので打ち消し合い、1/√N で消える

  いまの domain/smoothing.py (TimeBasedEma) は d を平滑化している。効かない側。
""")
    truth = project(FACE3D, 0.0)
    print("  完全対称な顔・yaw 0°・MediaPipe Full 相当の誤差（真の d = 0.0000）")
    print(f"  {'N':>5} {'d を平均':>22} {'座標を平均':>24}")
    print("  " + "-" * 54)
    for count in (1, 5, 10, 30, 100):
        by_distance, by_coordinate = [], []
        for trial in range(60):
            rng = np.random.default_rng(trial * 1000 + count)
            frames = truth[None, :, :] + rng.normal(
                0.0, LANDMARK_SIGMA, size=(count, len(truth), 2)
            )
            by_distance.append(float(np.median([distance_2d(frame) for frame in frames])))
            by_coordinate.append(distance_2d(frames.mean(axis=0)))
        mean_distance = float(np.mean(by_distance))
        mean_coordinate = float(np.mean(by_coordinate))
        print(
            f"  {count:>5} {mean_distance:14.4f} ({percent(mean_distance)}) "
            f"{mean_coordinate:14.4f} ({percent(mean_coordinate)})"
        )
    print()
    print("  → d を平均: N を増やしても 0.073 のまま。100 倍がんばっても改善しない。")
    print("  → 座標を平均: 10 フレームで 3.2 倍、100 フレームで 10 倍の改善。")
    print()
    print("  ただし系統誤差（毎回同じ方向にズレる誤差）は平均では減らない:")
    for degrees in (3, 5):
        biased = project(FACE3D, degrees)
        values = []
        for trial in range(40):
            rng = np.random.default_rng(trial)
            frames = biased[None, :, :] + rng.normal(
                0.0, LANDMARK_SIGMA, size=(100, len(biased), 2)
            )
            values.append(distance_2d(frames.mean(axis=0)))
        mean = float(np.mean(values))
        print(
            f"    yaw {degrees}° で座り続けると、100 フレーム平均しても "
            f"d={mean:.4f} ({percent(mean)}) のまま"
        )
    print("    → カメラが最初から傾いているのと同じ。何枚撮っても全部傾いている。")


def section_4() -> None:
    heading("4", "対策11: 3D で測る — 姿勢の影響が原理的に消える")
    print("""
  2D では回転すると形が変わって見えるので、姿勢と非対称を区別できない。
  3D なら回転は「向きが変わっただけで形は同じ」と扱える。

  object symmetry 解析（形態計測学の標準手法）:
    1. 3D 点群 X を用意する
    2. X を鏡映し、左右のペア名を入れ替えた M(X) をつくる
    3. M(X) を X へ最小二乗で重ね合わせる（Procrustes 重ね合わせ）
    4. 重ねきれずに残った距離が、本当の左右差

  折り目を先に決めない。重ね合わせが最適な折り方を自動で見つける。
  X を回すと M(X) も同じだけ回るので、結果は姿勢に依存しない。
""")
    print(f"  {'yaw':>6} {'現行 2D':>12} {'表示':>10}  |{'3D':>12} {'表示':>10}")
    print("  " + "-" * 60)
    for degrees in (0, 3, 5, 8, 15):
        flat = distance_2d(project(FACE3D, degrees))
        solid = distance_3d(FACE3D @ yaw_matrix(degrees).T)
        print(
            f"  {degrees:>4}° {flat:12.4f} {percent(flat):>10}  |{solid:12.4f} {percent(solid):>10}"
        )
    print()
    print("  → 3D 側は yaw が何度でも d=0.0000。副次的に §5.2 の正中線問題も消える。")
    print()
    print("  ただし代償がある。z にはまた別の誤差が乗る:")
    for error_percent in (2, 5, 10):
        values = []
        for trial in range(40):
            rng = np.random.default_rng(trial)
            noisy = FACE3D.copy()
            noisy[:, 2] += rng.normal(0.0, error_percent / 100.0, size=len(noisy))
            values.append(distance_3d(noisy))
        mean = float(np.mean(values))
        print(
            f"    z に {error_percent:>2}% IOD の誤差 → 完全対称な顔でも "
            f"d={mean:.4f} ({percent(mean)})"
        )
    print("    → z の誤差が 5% IOD あると yaw 8° 相当。消した分をそのまま失う。")
    print("      MediaPipe の z 精度は未公表なので、実測してから決めること。")


def section_5() -> None:
    heading("5", "レベル3 を全部やった後、画面には何が出るか")
    print("  仮に測定を完璧にできたとする（検出誤差ゼロ・姿勢の影響ゼロ）。")
    print()
    print(f"  {'':<26} {'真の d':>10} {'現行の表示':>14}")
    print("  " + "-" * 52)
    for label, distance in (
        ("対称な成人（文献値）", NORMAL_D),
        ("非対称な成人（文献値）", CLINICAL_D),
    ):
        print(f"  {label:<26} {distance:10.4f} {percent(distance):>14}")
    print()
    print("  → 測定が完璧でも、画面の数字は 97.59% と 95.48%。")
    print("    レベル3 は「正しく測れるようにする」だけ。")
    print("    「読めるように見せる」（レベル1）を直さない限り画面に出てこない。")


SECTIONS: dict[str, tuple[str, Callable[[], None]]] = {
    "1": ("姿勢の仕組み", section_1),
    "2": ("対策9 yaw ゲート", section_2),
    "3": ("対策10 フレーム平均", section_3),
    "4": ("対策11 3D 化", section_4),
    "5": ("レベル3 の限界", section_5),
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

    print(f"metric  : {CONFIG.metric_version}")
    print(f"landmark: {CONFIG.landmark_map_version}")
    print(f"信号全幅: {SIGNAL_SPAN * IPD_MM:.2f}mm （対称な成人 ↔ 非対称な成人）")
    for letter in arguments.section or sorted(SECTIONS):
        SECTIONS[letter][1]()
    print()
    print("=" * 100)
    print("解説: docs/symmetry-metric-level3.md")
    print("=" * 100)


if __name__ == "__main__":
    main()
