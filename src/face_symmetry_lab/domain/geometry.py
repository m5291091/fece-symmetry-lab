"""左右差指標のための純粋な幾何（§8.3, §8.4, §8.5.1）。

正規化は roll・スケール・平行移動を取り除き、左右の形状差だけを残す。正中線は
``x = ay + b`` の回帰ではなく全最小二乗（PCA）で推定する。そうすることで、
ほぼ水平な正中線でも推定が破綻しない。

正中線と鏡映は 2D の `symmetry-2d/2.0.0` だけが使う。`normalize_landmarks_3d`
は §8.5.1 の object symmetry 解析へ 3 軸そろった座標を渡すためのもので、割る
スケールが **3D の**両眼間距離である点だけが 2D と異なる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: 正規化後の両眼距離がこの値を下回る入力は、使用不能として扱う（§8.3）。
MIN_EYE_DISTANCE: float = 1e-9


class GeometryError(ValueError):
    """幾何入力が不正である。"""


@dataclass(frozen=True, slots=True)
class Midline:
    """``origin`` を通り、単位ベクトル ``direction`` の向きを持つ直線。"""

    origin: NDArray[np.float64]
    direction: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NormalizationTransform:
    """§8.3 が適用する相似変換と、その厳密な逆変換。

    順方向は ``((p - center) @ rotation.T) / scale`` である。``rotation`` は
    正規直交なので、逆変換は同じ 3 手順を逆順にほどくだけで得られる。§8.6.2 が
    対称化した点を近似ではなく厳密に元画像の画素格子へ戻すために、これが必要
    である。
    """

    #: 2 つの眼のアンカーの中点。入力の座標系で持つ。
    center: NDArray[np.float64]
    #: 2x2 の roll 補正。``@ rotation.T`` として右から適用する。
    rotation: NDArray[np.float64]
    #: 両眼間隔。順方向の変換でこれが 1 になる。
    scale: float

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """入力座標を正規化フレームへ移す。"""
        array = _as_points(points)
        return ((array - self.center) @ self.rotation.T) / self.scale

    def invert(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """正規化座標を入力のフレームへ戻す。"""
        array = _as_points(points)
        return (array * self.scale) @ self.rotation + self.center


def _as_points(points: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise GeometryError("座標は (N, 2) の配列である必要があります。")
    if not np.isfinite(array).all():
        raise GeometryError("座標は有限値である必要があります。")
    return array


def _as_points_3d(points: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise GeometryError("座標は (N, 3) の配列である必要があります。")
    if not np.isfinite(array).all():
        raise GeometryError("座標は有限値である必要があります。")
    return array


def _anchor(points: NDArray[np.float64], corners: tuple[int, int]) -> NDArray[np.float64]:
    count = len(points)
    for index in corners:
        if not 0 <= index < count:
            raise GeometryError(f"眼の index が範囲外です: {index}")
    return points[list(corners)].mean(axis=0)


def normalization_transform(
    points: NDArray[np.float64],
    *,
    left_eye_corners: tuple[int, int],
    right_eye_corners: tuple[int, int],
) -> NormalizationTransform:
    """§8.3 の変換を導く。眼を結ぶ線を水平にし、長さを 1 にし、中心へ寄せる。

    適用結果ではなく変換自体を返す。逆方向へ戻す必要がある呼び出し側——正規化
    フレームで計算されるが元画像へ描かれる §8.6.2 の対称化点——が、両方向で
    まったく同じ変換を使えるようにするためである。
    """
    array = _as_points(points)
    left = _anchor(array, left_eye_corners)
    right = _anchor(array, right_eye_corners)

    delta = right - left
    distance = float(np.hypot(delta[0], delta[1]))
    if distance <= MIN_EYE_DISTANCE:
        raise GeometryError("両眼の距離が小さすぎるため正規化できません。")

    cosine = float(delta[0]) / distance
    sine = -float(delta[1]) / distance
    return NormalizationTransform(
        center=(left + right) / 2.0,
        rotation=np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64),
        scale=distance,
    )


def normalize_landmarks(
    points: NDArray[np.float64],
    *,
    left_eye_corners: tuple[int, int],
    right_eye_corners: tuple[int, int],
) -> NDArray[np.float64]:
    """眼を結ぶ線を水平にし、両眼距離を 1 にし、両眼の中点を原点にする。"""
    transform = normalization_transform(
        points,
        left_eye_corners=left_eye_corners,
        right_eye_corners=right_eye_corners,
    )
    return transform.apply(points)


def normalize_landmarks_3d(
    points: NDArray[np.float64],
    *,
    left_eye_corners: tuple[int, int],
    right_eye_corners: tuple[int, int],
) -> NDArray[np.float64]:
    """§8.3 の相似変換を 3 軸へ広げる（§8.5.1）。単位は 3D の両眼間距離である。

    面内の扱いは §8.3 と同じで、眼 anchor を結ぶ線を水平にし、両眼中点を原点へ
    寄せる。奥行きも同じ 4 点から取った中点を原点にする。

    **割るスケールは 3 次元で測った両眼 anchor 間距離である。** §8.3 の 2D 投影
    距離を使ってはならない。yaw が ``θ`` のとき投影距離は ``cos θ`` 倍に縮むので、
    それで割ると距離が ``1/cos θ`` 倍に膨らみ、object symmetry が剛体不変性で
    追い出したはずの姿勢依存が、正規化という裏口から戻ってくる。``θ = 0`` では
    両者が一致するため、正面の合成顔だけでは見つからない種類の誤りである。

    §8.5.1 の重ね合わせは並進と回転を吸収するので、この変換のうち距離へ効くのは
    スケールだけである。原点の取り方と面内回転は 3D の剛体変換であり、残差を
    変えない。それでも同じ座標系へ揃えておくのは、xy / z 分解（§8.6.5）と、
    2D 側の正規化座標との突き合わせを読める形に保つためである。
    """
    array = _as_points_3d(points)
    transform = normalization_transform(
        array[:, :2],
        left_eye_corners=left_eye_corners,
        right_eye_corners=right_eye_corners,
    )
    left = _anchor(array, left_eye_corners)
    right = _anchor(array, right_eye_corners)

    scale = float(np.linalg.norm(right - left))
    if scale <= MIN_EYE_DISTANCE:
        raise GeometryError("両眼の距離が小さすぎるため正規化できません。")

    plane = ((array[:, :2] - transform.center) @ transform.rotation.T) / scale
    depth = (array[:, 2] - float(left[2] + right[2]) / 2.0) / scale
    return np.column_stack([plane, depth])


def fit_midline(points: NDArray[np.float64], indices: tuple[int, ...]) -> Midline:
    """与えられた点群に対し、直交距離最小（TLS）で正中線を推定する。"""
    array = _as_points(points)
    count = len(array)
    for index in indices:
        if not 0 <= index < count:
            raise GeometryError(f"正中点の index が範囲外です: {index}")

    selected = array[list(indices)]
    if len(selected) < 3:
        raise GeometryError("正中線の推定には 3 点以上が必要です。")

    origin = selected.mean(axis=0)
    centred = selected - origin
    if float(np.abs(centred).max()) <= MIN_EYE_DISTANCE:
        raise GeometryError("正中点が退化しているため直線を推定できません。")

    # 第 1 主成分が、全最小二乗による直線の方向である。
    _, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)
    if not np.isfinite(singular_values).all() or float(singular_values[0]) <= MIN_EYE_DISTANCE:
        raise GeometryError("正中点が退化しているため直線を推定できません。")

    direction = np.asarray(right_vectors[0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= MIN_EYE_DISTANCE:
        raise GeometryError("正中線の方向を決定できません。")
    return Midline(origin=origin, direction=direction / norm)


def reflect_across_midline(
    points: NDArray[np.float64],
    midline: Midline,
) -> NDArray[np.float64]:
    """すべての点を正中線で鏡映する。この操作は対合（2 回で恒等）である。"""
    array = _as_points(points)
    direction = np.asarray(midline.direction, dtype=np.float64)
    relative = array - midline.origin
    projection = np.outer(relative @ direction, direction)
    return midline.origin + 2.0 * projection - relative
