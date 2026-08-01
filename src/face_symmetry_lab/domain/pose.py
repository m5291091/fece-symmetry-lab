"""Head pose from a facial transformation matrix (§8.2.1).

Convention: right-handed camera frame with +X right, +Y up, +Z toward the
camera, intrinsic Y-X-Z Euler angles. ``yaw > 0`` turns the nose toward the
right of the non-mirrored image, ``pitch > 0`` lifts it, ``roll > 0`` is
clockwise. Scale and shear are removed by a polar decomposition before the
angles are read, so a matrix carrying scale still yields the same rotation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import asin, atan2, degrees

import numpy as np
from numpy.typing import NDArray

#: Below this |cos(pitch)| the Y and Z axes are degenerate (gimbal lock).
GIMBAL_EPSILON: float = 1e-7


class PoseError(ValueError):
    """The matrix cannot be interpreted as a head rotation."""


@dataclass(frozen=True, slots=True)
class FacePose:
    """Head orientation in degrees, in the app convention."""

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
    """Nearest rotation matrix via polar decomposition (§8.2.1)."""
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
    """Extract intrinsic Y-X-Z angles from a facial transformation matrix."""
    rotation = _orthonormalize(_rotation_part(matrix))

    # R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    sin_pitch = float(min(1.0, max(-1.0, -rotation[1, 2])))
    pitch = asin(sin_pitch)
    cos_pitch = float(np.sqrt(max(0.0, 1.0 - sin_pitch * sin_pitch)))

    if cos_pitch > GIMBAL_EPSILON:
        yaw = atan2(float(rotation[0, 2]), float(rotation[2, 2]))
        roll = atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    else:
        # Gimbal lock: yaw and roll are not separable, so pin roll to zero.
        yaw = atan2(-float(rotation[2, 0]), float(rotation[0, 0]))
        roll = 0.0

    return FacePose(
        yaw_deg=degrees(yaw),
        pitch_deg=degrees(pitch),
        roll_deg=degrees(roll),
    )
