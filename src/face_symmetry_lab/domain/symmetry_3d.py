"""3D object symmetry 解析（§8.5.1, §8.6.5, FR-036）。

§8.5 は「正中線で折って左右を比べる」。本 module はそれをやめて、**鏡映して
ペア名を入れ替えた配置を、元の配置へ剛体で重ね合わせる**。折る平面を先に決め
ないので、重ね合わせが最適な折り方を自動で見つける。

    X       = 対応ペア 126 点の 3D 座標
    M(X)    = sigma( mirror_x(X) )            sigma はペア表そのもの
    R, t    = argmin ||X - (M(X)·R + t)||²    R は回転、スケール自由度なし
    d_3d    = sqrt( mean_i ||X_i - (M(X)·R + t)_i||² )

`X` を剛体で動かすと `M(X)` も同じだけ動くので、`d_3d` は頭部の並進・回転
（yaw / pitch / roll のすべて）に対して不変である。この不変性が、§8.6.6 に
記録した現行 2D 指標の 2 つの欠陥——正中線あてはめが引きずられること、眼
anchor が正規化基準そのものであること——を同時に消す。**どちらも「折り方を
先に決める」ことから来ていた。**

`symmetry-2d/2.0.0` は退役していない。本 module はそれと**並走**し、この版では
表示・EMA・エクスポートのいずれへも接続しない（FR-036）。

部位別と xy / z 分解は §8.6 と同じ位置づけの**並列の診断出力**である。総合値へ
寄与しない。ただし §8.5 と違い、部位別は**全 126 点で 1 回だけ行った重ね合わせ
の残差を部位ごとに読んだもの**であり、部位ごとに Procrustes を走らせ直したもの
ではない（理由は §8.6.5 に 4 点記録した）。この定義の帰結として、重ね合わせが
成立しなければ部位別も定義されない。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.geometry import GeometryError, normalize_landmarks_3d
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig


class Symmetry3DError(ValueError):
    """入力からそもそも 3D の左右差距離を求められない。"""


def equal_weight_distance_from_parts(distances: Mapping[str, float | None]) -> float | None:
    """§8.6.5 の `s_3d` を、部位別距離の辞書から直接求める純関数。

        s_3d = sqrt( Σ_p d_p² / P )        P = 部位数（現行は 4）

    `Symmetry3DResult.equal_weight_distance` が自身の `splits`（生値）から呼ぶ
    のと同じ式であり、そちらから切り出した。部位別距離を単独で持つ場面
    （`ui.scores` の「全体のスコア」計算過程の開示、H2）でも同じ式を再利用でき、
    domain 層の外（ui）で計算式を複製しないための境界である（CLAUDE.md §5・§6）。

    **部位が 1 つでも欠測（`None`）、またはキー自体が無ければ欠測を返す。**
    使える部位だけで平均しない（§8.6.5 と同じ規約、不変条件 11）。
    """
    values: list[float] = []
    for name in PART_NAMES:
        value = distances.get(name)
        if value is None:
            return None
        values.append(value)
    return math.sqrt(sum(value**2 for value in values) / len(PART_NAMES))


@dataclass(frozen=True, slots=True)
class AxisSplit:
    """残差 RMS と、その面内 / 奥行き分解（§8.6.5）。

    残差はベクトルなので直交分解でき、`distance² = plane² + depth²` が恒等的に
    成り立つ。分解は新しい指標ではなく、同じ残差を軸方向へ読み替えたものである。
    """

    #: 残差ノルムの RMS。これが d_3d（部位別なら d_3d,p）である。
    distance: float
    #: 面内（xy）成分の RMS。
    plane: float
    #: 奥行き（z）成分の RMS。
    depth: float


@dataclass(frozen=True, slots=True)
class Symmetry3DResult:
    """§8.6.5 の出力一式。部位別は総合値の内訳ではなく、並列の診断値である。"""

    splits: dict[str, AxisSplit | None]
    overall: AxisSplit | None
    metric_version: str
    landmark_map_version: str

    def split(self, name: str) -> AxisSplit | None:
        return self.splits.get(name)

    def distance(self, name: str) -> float | None:
        split = self.splits.get(name)
        return None if split is None else split.distance

    @property
    def is_complete(self) -> bool:
        return self.overall is not None and all(
            self.splits.get(name) is not None for name in PART_NAMES
        )

    @property
    def equal_weight_distance(self) -> float | None:
        """§8.6.5 の `s_3d`。部位を等しく 1 票として扱う、もう 1 つの総合距離。

            s_3d = sqrt( Σ_p d_3d,p² / P )        P = 部位数（現行は 4）

        `overall`（`d_3d`）が対応点をそれぞれ 1 票として数えるのに対し、これは
        部位をそれぞれ 1 票として数える。**較正された重みではない** —— 全部位
        について 1 に固定され、調整の余地を持たない。§8.6 が禁じるのは固定評価
        セットで較正されていない自由パラメータであり、この固定 1 は対象外である。

        `is_complete` と同じ条件（`overall` と全部位の `splits` がすべて揃って
        いること）を使う。**部位が 1 つでも欠測なら欠測にし、使える部位だけで
        平均しない。** この定義は部位集合が固定であることに依存しており、3 部位
        で平均した値と 4 部位で平均した値を同じ名前で扱うと、部位が欠測した瞬間
        に値が跳ぶ。`overall` 自身は §8.5.1 のとおり、重ね合わせさえ成立すれば
        部位の内訳なしに定義され続けるため、この条件の対象外である。
        """
        if not self.is_complete:
            return None
        return equal_weight_distance_from_parts({name: self.distance(name) for name in PART_NAMES})


def _usable(pairs: tuple[tuple[int, int], ...], available: int) -> tuple[tuple[int, int], ...]:
    return tuple((left, right) for left, right in pairs if left < available and right < available)


def object_symmetry_residual(
    normalized: NDArray[np.float64],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[NDArray[np.float64], dict[int, int]]:
    """§8.5.1 の手順 1〜4。残差 `(M, 3)` と、landmark index → 行番号の対応を返す。

    行の並びは使用する landmark index の昇順である。部位別（§8.6.5）はこの残差
    の行を選び直すだけで、重ね合わせをやり直さない。

    **入力は両眼間距離を 1 とした正規化座標であること。** ここではスケールを
    決めない。`compute_symmetry_3d` は `normalize_landmarks_3d` を通してから
    この関数を呼ぶ。生の画素座標を直接渡すと、残差が画素のまま出る。

    公開しているのは、診断（`scripts/measure_metric_noise.py`）が残差そのものを
    軸ごとに読む必要があるためである。同じ手順の写しをスクリプト側へ置くと、
    片方だけが静かにずれる。実装は 1 つに保つ。
    """
    used = sorted({index for pair in pairs for index in pair})
    position = {landmark: order for order, landmark in enumerate(used)}
    original = normalized[used]

    # 2. 鏡映（x を反転）。
    mirrored = original.copy()
    mirrored[:, 0] *= -1.0

    # 3. sigma でペア名を入れ替える。ペア表そのものであり、ここで新しい対応を
    #    推測しない。ペアに現れない点は使っていないので、恒等のまま残らない。
    swap = np.arange(len(used))
    for left, right in pairs:
        swap[position[left]], swap[position[right]] = position[right], position[left]
    mirrored = mirrored[swap]

    # 4. 並進（重心そろえ）と回転だけで重ね合わせる。スケールは §8.5.1 の
    #    正規化で決まっており、ここでは触らない。
    centred = original - original.mean(axis=0)
    mirrored_centred = mirrored - mirrored.mean(axis=0)
    left_vectors, _, right_vectors = np.linalg.svd(mirrored_centred.T @ centred)
    rotation = left_vectors @ right_vectors
    if np.linalg.det(rotation) < 0:
        # 鏡映へ落ちた。最後の特異ベクトルを反転して真の回転へ戻す。
        left_vectors[:, -1] *= -1.0
        rotation = left_vectors @ right_vectors
    aligned = mirrored_centred @ rotation

    return np.asarray(centred - aligned, dtype=np.float64), position


def axis_split(residual: NDArray[np.float64]) -> AxisSplit:
    """残差 `(M, 3)` の RMS と、その xy / z 分解（§8.6.5）。

    恒等式 `distance² = plane² + depth²` は構成そのものから従う。両者を別々に
    集計しているのではなく、同じ二乗和を 2 つに割っているだけである。
    """
    distance = float(np.sqrt((residual**2).sum(axis=1).mean()))
    plane = float(np.sqrt((residual[:, :2] ** 2).sum(axis=1).mean()))
    depth = float(np.sqrt((residual[:, 2] ** 2).mean()))
    return AxisSplit(distance=distance, plane=plane, depth=depth)


def compute_symmetry_3d(
    points: NDArray[np.float64],
    config: MetricConfig,
) -> Symmetry3DResult:
    """3 軸そろった顔 1 つ分の左右差を、object symmetry 解析で測る（§8.5.1）。

    `points` は §8.1.1 の画素相当 `(N, 3)` 座標である。`z` が欠測のフレームは
    そもそもここへ届かない（`to_pixel_array_3d` が拒否する）。
    """
    try:
        normalized = normalize_landmarks_3d(
            points,
            left_eye_corners=config.left_eye_corners,
            right_eye_corners=config.right_eye_corners,
        )
    except GeometryError as error:
        raise Symmetry3DError(str(error)) from error

    available = len(normalized)
    usable = {name: _usable(config.parts[name].pairs, available) for name in PART_NAMES}
    pooled = tuple(pair for name in PART_NAMES for pair in usable[name])

    missing = Symmetry3DResult(
        splits=dict.fromkeys(PART_NAMES),
        overall=None,
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
    if len(pooled) < config.overall_min_pairs:
        # §8.5.1: 重ね合わせを行わない。部位別も定義されない。0 で埋めない。
        return missing

    residual, position = object_symmetry_residual(normalized, pooled)

    splits: dict[str, AxisSplit | None] = {}
    for name in PART_NAMES:
        part_pairs = usable[name]
        if len(part_pairs) < config.parts[name].min_pairs:
            splits[name] = None
            continue
        rows = [position[index] for pair in part_pairs for index in pair]
        splits[name] = axis_split(residual[rows])

    return Symmetry3DResult(
        splits=splits,
        overall=axis_split(residual),
        metric_version=config.metric_version,
        landmark_map_version=config.landmark_map_version,
    )
