"""§8.5.1 / §8.6.5 の 3D object symmetry 解析。

このファイルの test は 1 つの罠を意識して書かれている。**「常に 0 を返す実装」は
剛体不変性の test を全部通る。** したがって不変性は必ず**左右非対称な**合成顔で
確かめ、同じ test の中で値が 0 でないことも要求する。
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.io.metric_config import (
    SYMMETRY_3D_METRIC_CONFIG_PATH,
    load_metric_config,
)
from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    build_landmark_frame,
    to_pixel_array,
    to_pixel_array_3d,
)
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig, PartMap, build_metric_config
from face_symmetry_lab.domain.symmetry import compute_symmetry
from face_symmetry_lab.domain.symmetry_3d import Symmetry3DError, compute_symmetry_3d

#: 計測される対応点の総数（12 ペア = 24 点）。既知量の期待値に効く。
MEASURED_POINTS = 24

#: 両眼 anchor 間距離。左 anchor は (-20, 0, z)、右 anchor は (20, 0, z) で、`z` は
#: 左右で厳密に等しいため、正面では 3D 距離も 2D 投影距離もこの値になる。
EYE_DISTANCE = 40.0

# x = 0 について厳密に左右対称な合成顔の面内座標。
# 19 と 21 は x を等しく取ってある。§8.5.1 の重ね合わせが吸収できる 3 自由度
# （x 平行移動・roll・yaw）と直交する変位を注入するために必要である。
_PLANE = np.array(
    [
        [-30.0, 0.0],  # 0 左目 外側（眼 anchor）
        [-10.0, 0.0],  # 1 左目 内側（眼 anchor）
        [10.0, 0.0],  # 2 右目 内側（眼 anchor）
        [30.0, 0.0],  # 3 右目 外側（眼 anchor）
        [0.0, -20.0],  # 4 正中 額
        [0.0, 20.0],  # 5 正中 鼻先
        [0.0, 50.0],  # 6 正中 顎
        [-40.0, 30.0],  # 7 輪郭
        [40.0, 30.0],  # 8
        [-45.0, 10.0],  # 9
        [45.0, 10.0],  # 10
        [-35.0, -10.0],  # 11
        [35.0, -10.0],  # 12
        [-8.0, 25.0],  # 13 鼻
        [8.0, 25.0],  # 14
        [-12.0, 22.0],  # 15
        [12.0, 22.0],  # 16
        [-6.0, 28.0],  # 17
        [6.0, 28.0],  # 18
        [-15.0, 40.0],  # 19 口（注入点。21 と x が等しい）
        [15.0, 40.0],  # 20
        [-15.0, 46.0],  # 21 口（注入点）
        [15.0, 46.0],  # 22
        [-5.0, 43.0],  # 23 口
        [5.0, 43.0],  # 24
        [-20.0, -5.0],  # 25 眉
        [20.0, -5.0],  # 26
    ],
    dtype=np.float64,
)


def _depth(x, y):
    """顔の奥行き。`x` の偶関数なので、鏡映ペアの `z` は厳密に一致する。

    平面ではなく本物の立体にしておくことが重要である。奥行きが一定だと
    Procrustes の回転に縮退が生まれ、剛体不変性の test が理由もなく通る。
    """
    head = -0.015 * x**2 - 0.008 * (y - 10.0) ** 2
    nose = 8.0 * np.exp(-((x / 8.0) ** 2 + ((y - 25.0) / 12.0) ** 2))
    return head + nose


SYNTHETIC_FACE_3D = np.column_stack([_PLANE, _depth(_PLANE[:, 0], _PLANE[:, 1])])

_PARTS = {
    "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=2),
    "eyes": PartMap(pairs=((0, 3), (1, 2), (25, 26)), min_pairs=2),
    "nose": PartMap(pairs=((13, 14), (15, 16), (17, 18)), min_pairs=2),
    "mouth": PartMap(pairs=((19, 20), (21, 22), (23, 24)), min_pairs=2),
}


def _config(**overrides):
    settings = {
        "metric_version": "symmetry-3d/test",
        "landmark_map_version": "test/1.0.0",
        "landmark_count": len(SYNTHETIC_FACE_3D),
        "left_eye_corners": (0, 1),
        "right_eye_corners": (2, 3),
        "midline_indices": (4, 5, 6),
        "parts": dict(_PARTS),
    }
    settings.update(overrides)
    return build_metric_config(**settings)


def _asymmetric():
    """左右非対称な合成顔。不変性の test はすべてこれを使う。"""
    points = SYNTHETIC_FACE_3D.copy()
    points[19] += (-4.0, 1.5, 0.0)  # 口を片側だけ広げる
    points[13, 2] += 3.0  # 鼻を片側だけ手前へ出す（z にだけ効く）
    points[9, 1] -= 2.5  # 輪郭を片側だけ下げる
    return points


def _rotation(axis: str, degrees: float):
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "roll":  # z 軸まわり
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if axis == "yaw":  # y 軸まわり
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
    return np.array(  # pitch: x 軸まわり
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _normalized_points(width: int = 100, height: int = 100):
    """`LandmarkFrame` へ渡せる 0〜1 正規化座標と、対応する奥行き。"""
    points = tuple((float(x) / width, float(y) / height) for x, y in _PLANE)
    depths = tuple(float(z) / width for z in SYNTHETIC_FACE_3D[:, 2])
    return points, depths


# --------------------------------------------------------------------------
# 完全対称な入力（AC-SCORE-06）
# --------------------------------------------------------------------------
def test_a_perfectly_symmetric_face_has_zero_distance_everywhere() -> None:
    result = compute_symmetry_3d(SYNTHETIC_FACE_3D, _config())

    assert result.overall is not None
    assert result.overall.distance == pytest.approx(0.0, abs=1e-12)
    assert result.overall.plane == pytest.approx(0.0, abs=1e-12)
    assert result.overall.depth == pytest.approx(0.0, abs=1e-12)
    for name in PART_NAMES:
        split = result.split(name)
        assert split is not None
        assert split.distance == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 剛体不変性（AC-SCORE-05）— すべて左右非対称な顔で確かめる
# --------------------------------------------------------------------------
def test_the_asymmetric_reference_face_is_not_symmetric() -> None:
    """以下の不変性 test が「常に 0 を返す実装」を通さないための前提。"""
    result = compute_symmetry_3d(_asymmetric(), _config())

    assert result.overall is not None
    assert result.overall.distance > 0.01


@pytest.mark.parametrize("degrees", [0.0, 17.0, 45.0, 90.0, 143.0, 180.0])
def test_the_distance_is_invariant_to_roll(degrees: float) -> None:
    points = _asymmetric()
    baseline = compute_symmetry_3d(points, _config()).overall

    rolled = compute_symmetry_3d(points @ _rotation("roll", degrees).T, _config()).overall

    assert baseline is not None
    assert rolled is not None
    assert baseline.distance > 0.0
    assert rolled.distance == pytest.approx(baseline.distance, abs=1e-9)


@pytest.mark.parametrize("axis", ["yaw", "pitch"])
@pytest.mark.parametrize("degrees", [3.0, 8.0, 25.0])
def test_the_distance_is_invariant_to_yaw_and_pitch(axis: str, degrees: float) -> None:
    """§8.5.1: 折る平面を先に決めないので、頭部の回転は距離へ入らない。"""
    points = _asymmetric()
    baseline = compute_symmetry_3d(points, _config()).overall

    turned = compute_symmetry_3d(points @ _rotation(axis, degrees).T, _config()).overall

    assert baseline is not None
    assert turned is not None
    assert baseline.distance > 0.0
    assert turned.distance == pytest.approx(baseline.distance, abs=1e-9)


def test_the_distance_is_invariant_to_translation_and_uniform_scale() -> None:
    points = _asymmetric()
    baseline = compute_symmetry_3d(points, _config()).overall

    moved = compute_symmetry_3d(
        points * 7.5 + np.array([300.0, -80.0, 45.0]),
        _config(),
    ).overall

    assert baseline is not None
    assert moved is not None
    assert baseline.distance > 0.0
    assert moved.distance == pytest.approx(baseline.distance, abs=1e-9)


def test_yaw_moves_the_two_dimensional_distance_but_not_the_three_dimensional_one() -> None:
    """姿勢依存が 2D にはあり 3D には無いことを、1 つの test で対にして示す。

    §8.5.1 が正規化スケールへ 3D の両眼間距離を要求する理由でもある。2D 投影
    距離で割ると yaw で `1/cos θ` 倍に膨らみ、この test が落ちる。
    """
    points = _asymmetric()
    turned = points @ _rotation("yaw", 12.0).T
    config = _config()

    flat_before = compute_symmetry(points[:, :2], config).overall
    flat_after = compute_symmetry(turned[:, :2], config).overall
    solid_before = compute_symmetry_3d(points, config).overall
    solid_after = compute_symmetry_3d(turned, config).overall

    assert flat_before is not None
    assert flat_after is not None
    assert abs(flat_after - flat_before) > 1e-3
    assert solid_before is not None
    assert solid_after is not None
    assert solid_after.distance == pytest.approx(solid_before.distance, abs=1e-9)


def test_moving_the_midline_points_does_not_move_the_three_dimensional_distance() -> None:
    """§8.4: 3D 指標は正中線をあてはめないので、正中点を読まない。

    2D 側は同じ変位で値が動く。これが §8.6.6 の「測っていない点が、測っている
    全部を動かす」経路である。
    """
    points = _asymmetric()
    dragged = points.copy()
    dragged[[4, 5, 6], 0] += 6.0
    config = _config()

    solid = compute_symmetry_3d(points, config).overall
    dragged_solid = compute_symmetry_3d(dragged, config).overall
    flat = compute_symmetry(points[:, :2], config).overall
    dragged_flat = compute_symmetry(dragged[:, :2], config).overall

    assert solid is not None
    assert dragged_solid is not None
    assert dragged_solid.distance == pytest.approx(solid.distance, abs=1e-12)
    assert flat is not None
    assert dragged_flat is not None
    assert abs(dragged_flat - flat) > 1e-3


# --------------------------------------------------------------------------
# 既知量の注入（AC-SCORE-07）
# --------------------------------------------------------------------------
def test_a_known_injection_produces_exactly_the_expected_distance() -> None:
    """符号と大きさを閉じた式で確かめる。

    19 と 21 は x が等しい左側の点である。19 を `+b`、21 を `-b` だけ `y` へ動かす
    と、鏡映との差ベクトル場が、重ね合わせの吸収できる 3 自由度（x 平行移動・
    roll・yaw）のすべてと直交する。したがって残差は差ベクトルそのものになり、
    期待値が閉じた形で書ける。

        差が立つのは 19 / 20 / 21 / 22 の 4 点、各点の大きさは b
        d_3d = sqrt(4b² / M) / 両眼間距離 = 2b / (sqrt(M) × 40)
    """
    displacement = 4.0
    points = SYNTHETIC_FACE_3D.copy()
    points[19, 1] += displacement
    points[21, 1] -= displacement

    result = compute_symmetry_3d(points, _config())

    expected = 2.0 * displacement / (math.sqrt(MEASURED_POINTS) * EYE_DISTANCE)
    assert result.overall is not None
    assert result.overall.distance == pytest.approx(expected, rel=1e-9)
    # 変位は y だけなので、残差は面内へ全部乗る。
    assert result.overall.plane == pytest.approx(expected, rel=1e-9)
    assert result.overall.depth == pytest.approx(0.0, abs=1e-12)


def test_a_known_injection_lands_on_the_injected_part() -> None:
    displacement = 4.0
    points = SYNTHETIC_FACE_3D.copy()
    points[19, 1] += displacement
    points[21, 1] -= displacement

    result = compute_symmetry_3d(points, _config())

    mouth = result.split("mouth")
    assert mouth is not None
    # 口の 6 点だけが残差を受け持つ: 2b / (sqrt(6) × 40)。
    assert mouth.distance == pytest.approx(
        2.0 * displacement / (math.sqrt(6.0) * EYE_DISTANCE), rel=1e-9
    )
    for name in ("contour", "eyes", "nose"):
        split = result.split(name)
        assert split is not None
        assert split.distance == pytest.approx(0.0, abs=1e-12)


def test_injecting_more_asymmetry_raises_the_distance_monotonically() -> None:
    distances = []
    for offset in (0.0, 1.0, 3.0, 8.0):
        points = SYNTHETIC_FACE_3D.copy()
        points[19, 0] -= offset
        overall = compute_symmetry_3d(points, _config()).overall
        assert overall is not None
        distances.append(overall.distance)

    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0.0, abs=1e-12)
    assert distances[-1] > distances[1] > 0.0


def test_depth_only_asymmetry_is_detected() -> None:
    """`z` にしか現れない非対称。2D 指標が原理的に見られない量である。"""
    points = SYNTHETIC_FACE_3D.copy()
    points[13, 2] += 5.0
    config = _config()

    result = compute_symmetry_3d(points, config)
    flat = compute_symmetry(points[:, :2], config)

    assert result.overall is not None
    assert result.overall.distance > 0.0
    assert result.overall.depth > 0.0
    assert flat.overall == pytest.approx(0.0, abs=1e-12)


def test_perturbing_an_eye_anchor_is_visible_in_the_distance() -> None:
    """眼 anchor も 3D 指標では通常の対応点である（§8.6.6 の目のゲイン 0 と対比）。"""
    points = SYNTHETIC_FACE_3D.copy()
    points[0, 1] += 3.0

    overall = compute_symmetry_3d(points, _config()).overall

    assert overall is not None
    assert overall.distance > 0.0


# --------------------------------------------------------------------------
# xy / z 分解の恒等式（AC-SCORE-08）
# --------------------------------------------------------------------------
def test_the_axis_split_is_an_exact_identity() -> None:
    result = compute_symmetry_3d(_asymmetric(), _config())

    assert result.overall is not None
    for split in [result.overall, *(result.split(name) for name in PART_NAMES)]:
        assert split is not None
        assert split.distance**2 == pytest.approx(split.plane**2 + split.depth**2, abs=1e-12)


def test_the_overall_distance_is_the_point_weighted_pool_of_the_parts() -> None:
    """§8.6.5: 部位別は、同じ 1 回の重ね合わせの残差を行で分けたものである。"""
    config = _config()
    result = compute_symmetry_3d(_asymmetric(), config)

    weighted = 0.0
    points = 0
    for name in PART_NAMES:
        split = result.split(name)
        assert split is not None
        count = 2 * len(config.parts[name].pairs)
        weighted += count * split.distance**2
        points += count

    assert points == MEASURED_POINTS
    assert result.overall is not None
    assert result.overall.distance**2 == pytest.approx(weighted / points, abs=1e-12)


def test_a_part_value_reflects_the_single_global_superimposition() -> None:
    """§8.6.5: 部位別は部位ごとに Procrustes を走らせ直したものではない。

    全体で 1 回だけ重ね合わせているなら、**他部位だけ**を動かしても重ね方が
    変わり、鼻の値が動く。部位ごとに解く実装では鼻の値は他部位に影響されない
    ので、この test が両者を分ける。
    """
    config = _config()
    base = SYNTHETIC_FACE_3D.copy()
    base[13, 2] += 3.0  # 鼻だけを非対称にする
    moved = base.copy()
    moved[9, 1] -= 6.0  # 輪郭だけをさらに非対称にする

    before = compute_symmetry_3d(base, config).split("nose")
    after = compute_symmetry_3d(moved, config).split("nose")

    assert before is not None
    assert after is not None
    assert before.distance > 0.0
    assert after.distance != pytest.approx(before.distance, abs=1e-9)


# --------------------------------------------------------------------------
# 欠測（AC-SCORE-09）
# --------------------------------------------------------------------------
def test_a_frame_without_depth_yields_no_three_dimensional_input() -> None:
    """`z` が無いフレームからは 3D 座標を作れない。0 で埋めない（§8.1.1）。"""
    points, _ = _normalized_points()
    frame = build_landmark_frame(frame_id=1, face_count=1, points=points, width=100, height=100)

    assert frame.has_depth is False
    with pytest.raises(LandmarkError):
        to_pixel_array_3d(frame)


def test_missing_depth_leaves_the_two_dimensional_metric_untouched() -> None:
    """AC-SCORE-09: 3D 側が欠測でも、2D 側は同じ値を出し続ける。"""
    points, depths = _normalized_points()
    config = _config()
    without_depth = build_landmark_frame(
        frame_id=1, face_count=1, points=points, width=100, height=100
    )
    with_depth = build_landmark_frame(
        frame_id=1, face_count=1, points=points, width=100, height=100, depths=depths
    )

    flat_missing = compute_symmetry(to_pixel_array(without_depth), config)
    flat_present = compute_symmetry(to_pixel_array(with_depth), config)

    assert flat_missing.overall == flat_present.overall
    for name in PART_NAMES:
        assert flat_missing.distance(name) == flat_present.distance(name)
    # 奥行きがある側だけが 3D 指標を持てる。
    assert compute_symmetry_3d(to_pixel_array_3d(with_depth), config).overall is not None


def test_too_few_usable_pairs_makes_everything_missing_not_zero() -> None:
    """§8.5.1: 重ね合わせが成立しなければ、部位別も定義されない。"""
    result = compute_symmetry_3d(SYNTHETIC_FACE_3D[:8], _config())

    assert result.overall is None
    assert result.is_complete is False
    for name in PART_NAMES:
        assert result.split(name) is None


def test_a_missing_part_does_not_make_the_overall_distance_missing() -> None:
    """部位の最小ペア数を割っても、まとめた重ね合わせは成立し続ける。"""
    result = compute_symmetry_3d(SYNTHETIC_FACE_3D[:21], _config())

    assert result.split("mouth") is None
    assert result.overall is not None
    assert result.is_complete is False


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_landmarks_produce_no_distance(bad: float) -> None:
    points = SYNTHETIC_FACE_3D.copy()
    points[13, 2] = bad

    with pytest.raises(Symmetry3DError):
        compute_symmetry_3d(points, _config())


def test_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(Symmetry3DError):
        compute_symmetry_3d(SYNTHETIC_FACE_3D[:, :2], _config())


def test_zero_eye_distance_produces_no_distance() -> None:
    points = SYNTHETIC_FACE_3D.copy()
    points[[2, 3]] = points[[0, 1]]

    with pytest.raises(Symmetry3DError):
        compute_symmetry_3d(points, _config())


def test_the_result_carries_its_own_version_pair() -> None:
    result = compute_symmetry_3d(SYNTHETIC_FACE_3D, _config())

    assert result.metric_version == "symmetry-3d/test"
    assert result.landmark_map_version == "test/1.0.0"


def test_the_input_array_is_not_modified() -> None:
    points = _asymmetric()
    before = points.copy()

    compute_symmetry_3d(points, _config())

    assert np.array_equal(points, before)


# --------------------------------------------------------------------------
# 部位等重み距離 s_3d（§8.6.5, AC-SCORE-10）
# --------------------------------------------------------------------------


def _mirror_symmetric_full_face(config: MetricConfig) -> NDArray[np.float64]:
    """`config` の全対応ペアを、左右対称な baseline で埋めた合成 3D 顔。

    実際の bundle（部位ごとにペア数が違う）を使う test 専用の fixture。ペアに
    現れない landmark（正中点・虹彩）は 3D 指標が読まないので原点のままでよい。
    左右のペア (left, right) を x について厳密に鏡映させて配置するので、摂動を
    加えるまでは全体が完全対称になる。
    """
    points = np.zeros((config.landmark_count, 3), dtype=np.float64)
    step = 0
    for name in PART_NAMES:
        for left, right in config.parts[name].pairs:
            step += 1
            x = 10.0 + step
            y = 4.0 * math.sin(step)
            z = 3.0 * math.cos(step)
            points[left] = (-x, y, z)
            points[right] = (x, y, z)
    return points


def test_the_equal_weight_distance_matches_the_point_weighted_one_when_parts_are_balanced() -> None:
    """AC-SCORE-10: 各部位の使用点数が等しいとき `s_3d` は `d_3d` と一致する。

    `_config()` は 4 部位いずれも 3 ペア（6 点）で、点数加重の重みが部位ごとに
    等しくなる退化ケースである。次の test はペア数が違う非退化ケースを扱う。
    """
    result = compute_symmetry_3d(_asymmetric(), _config())

    assert result.overall is not None
    assert result.equal_weight_distance is not None
    assert result.equal_weight_distance > 0.0
    assert result.equal_weight_distance == pytest.approx(result.overall.distance, abs=1e-12)


def test_the_equal_weight_distance_differs_from_the_point_weighted_one_on_the_real_bundle() -> None:
    """AC-SCORE-10: 実際の bundle は部位ごとの点数が違う（17/26/7/13 ペア）ため、
    等重みの `s_3d` と点数加重の `d_3d` は一致しない。等重みが実際に効いている
    ことの証拠であり、両方とも 0 でないことも確かめる。
    """
    config = load_metric_config(SYMMETRY_3D_METRIC_CONFIG_PATH)
    assert {name: len(config.parts[name].pairs) for name in PART_NAMES} == {
        "contour": 17,
        "eyes": 26,
        "nose": 7,
        "mouth": 13,
    }
    points = _mirror_symmetric_full_face(config)
    nose_left, _ = config.parts["nose"].pairs[0]
    points[nose_left] += (0.0, 3.0, 0.0)  # 最小部位（鼻）だけへ非対称を注入する。

    result = compute_symmetry_3d(points, config)

    assert result.overall is not None
    assert result.equal_weight_distance is not None
    assert result.overall.distance > 0.0
    assert result.equal_weight_distance > 0.0
    assert result.equal_weight_distance != pytest.approx(result.overall.distance, rel=1e-6)


def test_a_missing_part_makes_the_equal_weight_distance_missing_but_not_the_overall() -> None:
    """AC-SCORE-10: `s_3d` は部位集合が固定であることに定義が依存するので、1 部位
    でも欠測なら欠測にする。使える部位だけで埋め合わせない。点数加重の `d_3d` は
    §8.5.1 のとおり定義され続ける（両者は欠測条件が異なる独立した出力である）。
    """
    result = compute_symmetry_3d(SYNTHETIC_FACE_3D[:21], _config())

    assert result.split("mouth") is None
    assert result.overall is not None
    assert result.equal_weight_distance is None


def test_the_equal_weight_distance_is_invariant_to_rigid_transforms() -> None:
    """AC-SCORE-10: `s_3d` も `d_3d` と同じく並進・回転・一様 scale に対して不変。"""
    points = _asymmetric()
    baseline = compute_symmetry_3d(points, _config())

    transformed_points = (points @ _rotation("yaw", 12.0).T) * 3.0 + np.array([120.0, -40.0, 15.0])
    transformed = compute_symmetry_3d(transformed_points, _config())

    assert baseline.equal_weight_distance is not None
    assert transformed.equal_weight_distance is not None
    assert baseline.equal_weight_distance > 0.0
    assert transformed.equal_weight_distance == pytest.approx(
        baseline.equal_weight_distance, abs=1e-9
    )


def test_a_perfectly_symmetric_face_has_zero_equal_weight_distance() -> None:
    result = compute_symmetry_3d(SYNTHETIC_FACE_3D, _config())

    assert result.equal_weight_distance is not None
    assert result.equal_weight_distance == pytest.approx(0.0, abs=1e-12)
