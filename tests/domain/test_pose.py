"""Pose contract tests (§8.2.1).

The app convention is intrinsic Y-X-Z on a right-handed camera frame with
+X right, +Y up, +Z toward the camera: `yaw > 0` turns the nose toward the
right of the non-mirrored image, `pitch > 0` lifts it, `roll > 0` is clockwise.
"""

import math

import numpy as np
import pytest

from face_symmetry_lab.domain.pose import PoseError, pose_from_matrix


def _rx(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.array(
        [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]],
        dtype=np.float64,
    )


def _ry(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.array(
        [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]],
        dtype=np.float64,
    )


def _rz(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]],
        dtype=np.float64,
    )


def _homogeneous(rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [12.0, -3.0, 40.0]
    return matrix


def test_identity_is_a_frontal_pose() -> None:
    pose = pose_from_matrix(np.eye(4))

    assert pose.yaw_deg == pytest.approx(0.0, abs=1e-9)
    assert pose.pitch_deg == pytest.approx(0.0, abs=1e-9)
    assert pose.roll_deg == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("degrees", [-10.0, -3.0, 3.0, 10.0])
def test_single_axis_yaw_round_trips(degrees: float) -> None:
    pose = pose_from_matrix(_homogeneous(_ry(degrees)))

    assert pose.yaw_deg == pytest.approx(degrees, abs=1e-6)
    assert pose.pitch_deg == pytest.approx(0.0, abs=1e-6)
    assert pose.roll_deg == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("degrees", [-10.0, -3.0, 3.0, 10.0])
def test_single_axis_pitch_round_trips(degrees: float) -> None:
    pose = pose_from_matrix(_homogeneous(_rx(degrees)))

    assert pose.pitch_deg == pytest.approx(degrees, abs=1e-6)
    assert pose.yaw_deg == pytest.approx(0.0, abs=1e-6)
    assert pose.roll_deg == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("degrees", [-10.0, -3.0, 3.0, 10.0])
def test_single_axis_roll_round_trips(degrees: float) -> None:
    pose = pose_from_matrix(_homogeneous(_rz(degrees)))

    assert pose.roll_deg == pytest.approx(degrees, abs=1e-6)
    assert pose.yaw_deg == pytest.approx(0.0, abs=1e-6)
    assert pose.pitch_deg == pytest.approx(0.0, abs=1e-6)


def test_composed_rotation_uses_intrinsic_y_x_z_order() -> None:
    rotation = _ry(9.0) @ _rx(-6.0) @ _rz(4.0)

    pose = pose_from_matrix(_homogeneous(rotation))

    assert pose.yaw_deg == pytest.approx(9.0, abs=1e-6)
    assert pose.pitch_deg == pytest.approx(-6.0, abs=1e-6)
    assert pose.roll_deg == pytest.approx(4.0, abs=1e-6)


def test_uniform_scale_is_removed_by_the_polar_decomposition() -> None:
    scaled = _homogeneous(_ry(8.0) @ np.diag([2.5, 2.5, 2.5]))

    pose = pose_from_matrix(scaled)

    assert pose.yaw_deg == pytest.approx(8.0, abs=1e-6)


def test_anisotropic_scale_and_shear_are_removed() -> None:
    shear = np.array([[1.0, 0.08, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    matrix = _homogeneous(_ry(10.0) @ np.diag([1.4, 0.9, 1.1]) @ shear)

    pose = pose_from_matrix(matrix)

    assert abs(pose.yaw_deg - 10.0) < 4.0
    assert math.isfinite(pose.pitch_deg)
    assert math.isfinite(pose.roll_deg)


def test_a_bare_three_by_three_rotation_is_accepted() -> None:
    pose = pose_from_matrix(_ry(7.0))

    assert pose.yaw_deg == pytest.approx(7.0, abs=1e-6)


def test_a_row_major_flat_sequence_is_accepted() -> None:
    pose = pose_from_matrix(_homogeneous(_rz(5.0)).reshape(-1).tolist())

    assert pose.roll_deg == pytest.approx(5.0, abs=1e-6)


def test_gimbal_lock_still_returns_finite_angles() -> None:
    pose = pose_from_matrix(_homogeneous(_rx(90.0)))

    assert pose.pitch_deg == pytest.approx(90.0, abs=1e-6)
    assert math.isfinite(pose.yaw_deg)
    assert math.isfinite(pose.roll_deg)


def test_a_mirrored_matrix_is_refused_rather_than_silently_flipped() -> None:
    reflection = np.diag([-1.0, 1.0, 1.0])

    with pytest.raises(PoseError, match="回転"):
        pose_from_matrix(_homogeneous(reflection))


def test_a_degenerate_matrix_is_refused() -> None:
    with pytest.raises(PoseError):
        pose_from_matrix(np.zeros((4, 4)))


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_matrices_are_refused(bad: float) -> None:
    matrix = _homogeneous(_ry(5.0))
    matrix[0, 0] = bad

    with pytest.raises(PoseError, match="有限"):
        pose_from_matrix(matrix)


def test_wrong_shapes_are_refused() -> None:
    with pytest.raises(PoseError, match="形状"):
        pose_from_matrix(np.eye(2))
