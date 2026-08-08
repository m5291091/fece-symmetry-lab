import numpy as np
import pytest

from face_symmetry_lab.adapters.imaging.piecewise_warp import OpenCvPiecewiseWarper
from face_symmetry_lab.application.symmetrization import symmetrize_frame
from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.symmetrization import SymmetrizationError

WIDTH = 200
HEIGHT = 200

#: 正規化フレーム座標での対称な顔（tests/domain を参照）。
_POINTS = (
    (0.35, 0.40),  # 0 左目 外側
    (0.45, 0.40),  # 1 左目 内側
    (0.55, 0.40),  # 2 右目 内側
    (0.65, 0.40),  # 3 右目 外側
    (0.50, 0.30),  # 4 正中
    (0.50, 0.50),  # 5 正中
    (0.50, 0.62),  # 6 正中
    (0.30, 0.55),  # 7 輪郭
    (0.70, 0.55),  # 8
    (0.31, 0.45),  # 9
    (0.69, 0.45),  # 10
    (0.33, 0.35),  # 11
    (0.67, 0.35),  # 12
    (0.47, 0.50),  # 13 鼻
    (0.53, 0.50),  # 14
    (0.44, 0.58),  # 15 口
    (0.56, 0.58),  # 16
)


def _config():
    return build_metric_config(
        metric_version="symmetry-2d/test",
        landmark_map_version="test/1.0.0",
        landmark_count=len(_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=2),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=2),
            "nose": PartMap(pairs=((13, 14),), min_pairs=1),
            "mouth": PartMap(pairs=((15, 16),), min_pairs=1),
        },
    )


def _frame(points=_POINTS, *, face_count: int = 1, width: int = WIDTH, height: int = HEIGHT):
    return build_landmark_frame(
        frame_id=1,
        face_count=face_count,
        points=points if face_count == 1 else (),
        width=width,
        height=height,
    )


def _image():
    return np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)


def _skewed():
    points = list(_POINTS)
    points[15] = (0.36, 0.58)
    return tuple(points)


def test_a_symmetric_face_produces_a_view_with_no_deviation() -> None:
    view = symmetrize_frame(_image(), _frame(), config=_config())

    assert view.max_deviation == pytest.approx(0.0, abs=1e-9)
    assert view.symmetrized_point_count == len(view.result.symmetrized_indices)
    assert view.overlay.shape == (HEIGHT, WIDTH, 3)


def test_a_skewed_face_reports_a_deviation_and_paints_the_overlay() -> None:
    image = _image()

    view = symmetrize_frame(image, _frame(_skewed()), config=_config())

    assert view.max_deviation is not None and view.max_deviation > 0.0
    assert view.mean_deviation is not None and view.mean_deviation > 0.0
    assert not np.array_equal(view.overlay, image)


def test_the_source_image_is_never_modified() -> None:
    """§14.3g は、取り込んだフレームに対する読み取り専用の表示である。"""
    image = _image()
    original = image.copy()

    symmetrize_frame(image, _frame(_skewed()), config=_config())

    assert np.array_equal(image, original)


def test_the_call_is_stateless_and_repeatable() -> None:
    """EMA も履歴も無いので、同じ入力からは常に同じ答えが出る。"""
    frame = _frame(_skewed())
    config = _config()

    first = symmetrize_frame(_image(), frame, config=config)
    second = symmetrize_frame(_image(), frame, config=config)

    assert first.max_deviation == pytest.approx(second.max_deviation)
    assert np.array_equal(first.overlay, second.overlay)


def test_the_symmetrized_positions_are_in_pixel_coordinates() -> None:
    view = symmetrize_frame(_image(), _frame(_skewed()), config=_config())

    measured = view.result.measured
    assert measured[:, 0].max() <= WIDTH
    assert measured[:, 1].max() <= HEIGHT
    # ほぼ対称な顔は、同じフレーム内の位置へ対称化される。
    symmetrized = view.result.symmetrized[list(view.result.symmetrized_indices)]
    assert symmetrized[:, 0].min() >= 0.0
    assert symmetrized[:, 1].min() >= 0.0


def test_a_multi_face_frame_is_refused() -> None:
    """FR-008: 1 本の正中線が 2 つの顔を代表することはできない。"""
    with pytest.raises(SymmetrizationError, match="単一顔"):
        symmetrize_frame(_image(), _frame(face_count=2), config=_config())


def test_a_frame_without_image_dimensions_is_refused() -> None:
    with pytest.raises(SymmetrizationError):
        symmetrize_frame(_image(), _frame(width=0, height=0), config=_config())


def test_a_degenerate_face_is_refused_rather_than_drawn() -> None:
    collapsed = list(_POINTS)
    collapsed[2] = collapsed[0]
    collapsed[3] = collapsed[1]

    with pytest.raises(SymmetrizationError):
        symmetrize_frame(_image(), _frame(tuple(collapsed)), config=_config())


# --- §8.6.4 画素の変形（FR-035）---------------------------------------------


class RecordingWarper:
    """渡されたものをそのまま報告する `PiecewiseWarper`。"""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.meshes: list[object] = []
        self.error = error

    def warp(self, rgb_image, mesh):
        self.meshes.append(mesh)
        if self.error is not None:
            raise self.error
        marked = rgb_image.copy()
        marked[0, 0] = (1, 2, 3)
        return marked


def test_no_warper_means_no_warped_image_and_no_complaint() -> None:
    """§8.6.4 は任意指定である。第 1 段階はその費用も言及も負ってはならない。"""
    view = symmetrize_frame(_image(), _frame(_skewed()), config=_config())

    assert view.warped is None
    assert view.warp_unavailable_reason is None


def test_a_warper_receives_the_symmetrized_points_as_its_target() -> None:
    """変形は幾何を追加しない。§8.6.2 がすでに計算したものを再利用する。"""
    warper = RecordingWarper()

    view = symmetrize_frame(_image(), _frame(_skewed()), config=_config(), warper=warper)

    assert view.warped is not None
    assert len(warper.meshes) == 1
    mesh = warper.meshes[0]
    assert np.allclose(mesh.source[: mesh.landmark_count], view.result.measured)
    assert np.allclose(mesh.target[: mesh.landmark_count], view.result.symmetrized)


def test_the_overlay_survives_a_warp_failure() -> None:
    """AC-SYMM-02: 画像化の失敗が、点の表示まで道連れにしてはならない。"""
    warper = RecordingWarper(error=RuntimeError("boom"))

    view = symmetrize_frame(_image(), _frame(_skewed()), config=_config(), warper=warper)

    assert view.warped is None
    assert view.warp_unavailable_reason is not None
    assert "boom" in view.warp_unavailable_reason
    assert view.overlay is not None
    assert view.max_deviation is not None


def test_an_over_displaced_face_refuses_the_image_but_keeps_the_points() -> None:
    """AC-SYMM-02: 変位ガードは理由を添えて fail-soft する。"""
    wrecked = list(_POINTS)
    wrecked[15] = (0.02, 0.95)  # 口の点をフレームの反対側へ飛ばす
    warper = RecordingWarper()

    view = symmetrize_frame(_image(), _frame(tuple(wrecked)), config=_config(), warper=warper)

    assert view.warped is None
    assert view.warp_unavailable_reason is not None
    assert "変形量" in view.warp_unavailable_reason
    assert warper.meshes == []  # 画素処理に入る前に拒否される
    assert view.max_deviation is not None


def test_a_real_warp_still_produces_an_image_when_triangles_are_skipped() -> None:
    """AC-SYMM-02: 三角形の読み飛ばしは fail-soft を壊さない。

    実 adapter を通す唯一の test である。向きの判定は読み飛ばす条件を増やす
    ので、それが「画像が出ない」へ退化していないことを、fake ではなく本物の
    区分アフィンで確かめる。
    """
    view = symmetrize_frame(
        _image(),
        _frame(_skewed()),
        config=_config(),
        warper=OpenCvPiecewiseWarper(),
    )

    assert view.warped is not None
    assert view.warp_unavailable_reason is None
    assert view.warped.shape == (HEIGHT, WIDTH, 3)
    assert view.max_deviation is not None


def test_the_warp_is_stateless_and_repeatable() -> None:
    frame = _frame(_skewed())
    config = _config()

    first = symmetrize_frame(_image(), frame, config=config, warper=RecordingWarper())
    second = symmetrize_frame(_image(), frame, config=config, warper=RecordingWarper())

    assert first.warped is not None and second.warped is not None
    assert np.array_equal(first.warped, second.warped)
