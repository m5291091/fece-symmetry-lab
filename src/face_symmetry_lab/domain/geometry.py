"""Pure 2D geometry for the symmetry metric (§8.3, §8.4).

Normalization removes roll, scale, and translation so that only the shape
difference between the two halves remains. The midline is fitted with total
least squares (PCA), not an ``x = ay + b`` regression, so a near-horizontal
midline cannot blow the fit up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Below this normalized eye distance the input is treated as unusable (§8.3).
MIN_EYE_DISTANCE: float = 1e-9


class GeometryError(ValueError):
    """Invalid geometry input."""


@dataclass(frozen=True, slots=True)
class Midline:
    """A line through ``origin`` along the unit vector ``direction``."""

    origin: NDArray[np.float64]
    direction: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NormalizationTransform:
    """The similarity transform §8.3 applies, and its exact inverse.

    Forward is ``((p - center) @ rotation.T) / scale``. Because ``rotation``
    is orthonormal the inverse is the same three steps undone in reverse,
    which is what §8.6.2 needs to put a symmetrized point back onto the
    original image's pixel grid rather than approximating its way there.
    """

    #: Midpoint of the two eye anchors, in the source coordinate space.
    center: NDArray[np.float64]
    #: 2x2 roll correction, applied on the right as ``@ rotation.T``.
    rotation: NDArray[np.float64]
    #: Interocular distance, scaled to 1 by the forward transform.
    scale: float

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Source coordinates to the normalized frame."""
        array = _as_points(points)
        return ((array - self.center) @ self.rotation.T) / self.scale

    def invert(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalized coordinates back to the source frame."""
        array = _as_points(points)
        return (array * self.scale) @ self.rotation + self.center


def _as_points(points: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise GeometryError("座標は (N, 2) の配列である必要があります。")
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
    """Derive §8.3's transform: level the eye line, scale it to 1, centre it.

    Returned rather than applied so a caller that has to map back — §8.6.2's
    symmetrized points, which are computed in the normalized frame but drawn
    on the original image — uses the very same transform in both directions.
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
    """Level the eye line, scale the eye distance to 1, and centre on the eyes."""
    transform = normalization_transform(
        points,
        left_eye_corners=left_eye_corners,
        right_eye_corners=right_eye_corners,
    )
    return transform.apply(points)


def fit_midline(points: NDArray[np.float64], indices: tuple[int, ...]) -> Midline:
    """Fit the midline through the given points by orthogonal distance (TLS)."""
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

    # The first principal component is the total-least-squares line direction.
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
    """Mirror every point across the midline. The operation is an involution."""
    array = _as_points(points)
    direction = np.asarray(midline.direction, dtype=np.float64)
    relative = array - midline.origin
    projection = np.outer(relative @ direction, direction)
    return midline.origin + 2.0 * projection - relative
