"""顔の変換行列から求める頭部姿勢（§8.2.1）。

規約は次のとおり。右手系のカメラ座標で +X が右、+Y が上、+Z がカメラ方向、
オイラー角は intrinsic Y-X-Z。``yaw > 0`` は鏡像でない画像の右へ鼻を向け、
``pitch > 0`` は鼻を上げ、``roll > 0`` は時計回りである。角度を読む前に極分解で
スケールとせん断を取り除くので、スケールを含む行列でも同じ回転が得られる。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import asin, atan2, degrees

import numpy as np
from numpy.typing import NDArray

#: |cos(pitch)| がこれを下回ると Y 軸と Z 軸が退化する（ジンバルロック）。
GIMBAL_EPSILON: float = 1e-7


class PoseError(ValueError):
    """その行列を頭部の回転として解釈できない。"""


@dataclass(frozen=True, slots=True)
class FacePose:
    """このアプリの規約による、度数法での頭部の向き。"""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def _rotation_part(matrix: NDArray[np.float64] | Sequence[float]) -> NDArray[np.float64]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        if array.size == 16:
            array = array.reshape(4, 4)
        elif array.size == 9:
            array = array.reshape(3, 3)
        else:
            raise PoseError("変換行列の形状が不正です。")
    if array.ndim != 2 or array.shape not in {(3, 3), (4, 4)}:
        raise PoseError("変換行列の形状が不正です。")
    if not np.isfinite(array).all():
        raise PoseError("変換行列は有限値である必要があります。")
    return np.array(array[:3, :3], dtype=np.float64)


def _orthonormalize(linear: NDArray[np.float64]) -> NDArray[np.float64]:
    """極分解によって求めた最も近い回転行列（§8.2.1）。"""
    try:
        left, singular_values, right = np.linalg.svd(linear)
    except np.linalg.LinAlgError as error:
        raise PoseError("変換行列を分解できません。") from error
    if float(singular_values.min()) <= GIMBAL_EPSILON:
        raise PoseError("変換行列が退化しているため回転を取り出せません。")

    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0:
        raise PoseError("回転ではなく鏡映を含む行列です。")
    return np.asarray(rotation, dtype=np.float64)


def pose_from_matrix(matrix: NDArray[np.float64] | Sequence[float]) -> FacePose:
    """顔の変換行列から intrinsic Y-X-Z の角度を取り出す。"""
    rotation = _orthonormalize(_rotation_part(matrix))

    # R = Ry(yaw) @ Rx(pitch) @ Rz(roll) の分解を行う
    sin_pitch = float(min(1.0, max(-1.0, -rotation[1, 2])))
    pitch = asin(sin_pitch)
    cos_pitch = float(np.sqrt(max(0.0, 1.0 - sin_pitch * sin_pitch)))

    if cos_pitch > GIMBAL_EPSILON:
        yaw = atan2(float(rotation[0, 2]), float(rotation[2, 2]))
        roll = atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    else:
        # ジンバルロック時は yaw と roll を分離できないので、roll を 0 に固定する。
        yaw = atan2(-float(rotation[2, 0]), float(rotation[0, 0]))
        roll = 0.0

    return FacePose(
        yaw_deg=degrees(yaw),
        pitch_deg=degrees(pitch),
        roll_deg=degrees(roll),
    )
