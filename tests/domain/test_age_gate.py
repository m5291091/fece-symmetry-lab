"""FR-038: 年齢ゲートの特徴量組み立てと判定（純粋計算）。

特徴量の定義の正本は `research/utkface-symmetry/utkface_age_gate_lab.ipynb` で、
そこで学習したモデルと同じ並び・同じ式でなければ意味を持たない。**並びの一致は
`tests/model/test_age_gate_contract.py` が実物のモデルで見る。** ここで見るのは、
式そのものと、退化した入力の扱いである。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from face_symmetry_lab.domain.age_gate import (
    FEATURE_NAMES,
    GEOMETRY_FEATURE_NAMES,
    POSE_FEATURE_NAMES,
    SYMMETRY_FEATURE_NAMES,
    AgeGateDecision,
    build_features,
    decide,
)
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.pose import FacePose

#: notebook の `IDX`。ここが違うと特徴量が別物になる。
IDX = {
    "forehead": 10,
    "chin": 152,
    "cheek_l": 234,
    "cheek_r": 454,
    "jaw_l": 58,
    "jaw_r": 288,
    "nasion": 168,
    "nose_tip": 1,
    "subnasale": 2,
    "nose_wing_l": 98,
    "nose_wing_r": 327,
    "mouth_l": 61,
    "mouth_r": 291,
    "lip_top": 0,
    "lip_bottom": 17,
    "eye_l_out": 33,
    "eye_l_in": 133,
    "eye_l_top": 159,
    "eye_l_bottom": 145,
    "eye_r_out": 263,
    "eye_r_in": 362,
    "eye_r_top": 386,
    "eye_r_bottom": 374,
    "brow_l": 105,
    "brow_r": 334,
}


class _StubSymmetry3D:
    """`Symmetry3DResult` のうち、ゲートが読む 2 つだけを持つ stub。"""

    def __init__(self, parts: dict[str, float | None], overall: float | None, equal: float | None):
        self._parts = parts
        self._overall = overall
        self.equal_weight_distance = equal

    def distance(self, name: str) -> float | None:
        return self._overall if name == "overall" else self._parts.get(name)


def _symmetry(value: float = 0.03) -> _StubSymmetry3D:
    return _StubSymmetry3D(parts=dict.fromkeys(PART_NAMES, value), overall=value, equal=value)


def _landmarks() -> np.ndarray:
    """手計算できる形の正規化済み座標（478 点、両眼間距離 = 1、両眼中点 = 原点）。

    face_height = 2.0（額 y=-1.0、顎 y=+1.0）、face_width = 1.5（頬 x=±0.75）に
    なるよう置く。z はすべて 0 にして、立体感の特徴が 0 になることを利用する。
    """
    points = np.zeros((478, 3), dtype=np.float64)
    place = {
        "forehead": (0.0, -1.0, 0.0),
        "chin": (0.0, 1.0, 0.0),
        "cheek_l": (-0.75, 0.25, 0.0),
        "cheek_r": (0.75, 0.25, 0.0),
        "jaw_l": (-0.6, 0.6, 0.0),
        "jaw_r": (0.6, 0.6, 0.0),
        "nasion": (0.0, -0.1, 0.0),
        "nose_tip": (0.0, 0.3, 0.0),
        "subnasale": (0.0, 0.5, 0.0),
        "nose_wing_l": (-0.15, 0.45, 0.0),
        "nose_wing_r": (0.15, 0.45, 0.0),
        "mouth_l": (-0.25, 0.7, 0.0),
        "mouth_r": (0.25, 0.7, 0.0),
        "lip_top": (0.0, 0.65, 0.0),
        "lip_bottom": (0.0, 0.75, 0.0),
        "eye_l_out": (-0.5, 0.0, 0.0),
        "eye_l_in": (-0.2, 0.0, 0.0),
        "eye_l_top": (-0.35, -0.05, 0.0),
        "eye_l_bottom": (-0.35, 0.05, 0.0),
        "eye_r_out": (0.5, 0.0, 0.0),
        "eye_r_in": (0.2, 0.0, 0.0),
        "eye_r_top": (0.35, -0.05, 0.0),
        "eye_r_bottom": (0.35, 0.05, 0.0),
        "brow_l": (-0.35, -0.2, 0.0),
        "brow_r": (0.35, -0.2, 0.0),
    }
    for key, value in place.items():
        points[IDX[key]] = value
    return points


def _named(vector: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, vector, strict=True))


def test_feature_name_groups_concatenate_in_the_published_order() -> None:
    """41 = 幾何 32 + 姿勢 3 + 対称度 6。この分割と順序が仕様である。"""
    assert (*GEOMETRY_FEATURE_NAMES, *POSE_FEATURE_NAMES, *SYMMETRY_FEATURE_NAMES) == FEATURE_NAMES
    assert (len(GEOMETRY_FEATURE_NAMES), len(POSE_FEATURE_NAMES), len(SYMMETRY_FEATURE_NAMES)) == (
        32,
        3,
        6,
    )
    assert len(FEATURE_NAMES) == 41
    assert len(set(FEATURE_NAMES)) == 41, "同じ名前が 2 度現れてはならない"


def test_build_features_computes_the_documented_ratios() -> None:
    """テスト 4: 既知の正規化座標から、手計算できる比率が出る。"""
    features = build_features(
        _landmarks(),
        pose=FacePose(yaw_deg=1.5, pitch_deg=-2.5, roll_deg=3.5),
        symmetry_3d=_symmetry(0.03),
    )

    assert features is not None
    values = _named(features.as_vector())

    assert values["face_height"] == pytest.approx(2.0)
    assert values["face_width"] == pytest.approx(1.5)
    assert values["face_aspect"] == pytest.approx(2.0 / 1.5)
    # 額は y<0 側。(eye_y - forehead_y) / face_height = (0 - (-1.0)) / 2.0
    assert values["forehead_ratio"] == pytest.approx(0.5)
    # (subnasale_y - nasion_y) / face_height = (0.5 - (-0.1)) / 2.0
    assert values["midface_ratio"] == pytest.approx(0.3)
    # (chin_y - subnasale_y) / face_height = (1.0 - 0.5) / 2.0
    assert values["lower_face_ratio"] == pytest.approx(0.25)
    # (chin_y - lip_bottom_y) / face_height = (1.0 - 0.75) / 2.0
    assert values["chin_ratio"] == pytest.approx(0.125)
    assert values["nose_width_ratio"] == pytest.approx(0.3 / 1.5)
    assert values["mouth_width_ratio"] == pytest.approx(0.5 / 1.5)
    assert values["jaw_width_ratio"] == pytest.approx(1.2 / 1.5)
    assert values["jaw_taper"] == pytest.approx(1.2 / 1.5)
    assert values["eye_width_left"] == pytest.approx(0.3)
    assert values["eye_width_right"] == pytest.approx(0.3)
    assert values["eye_open_left"] == pytest.approx(0.1)
    assert values["eye_aspect_left"] == pytest.approx(0.1 / 0.3)
    assert values["brow_gap_left"] == pytest.approx(0.15)
    # 口の中心 = (lip_top + lip_bottom) / 2 = 0.7
    assert values["mouth_position_ratio"] == pytest.approx(0.35)
    assert values["eye_to_mouth_ratio"] == pytest.approx(0.35)
    assert values["eye_to_nose_ratio"] == pytest.approx(0.15)
    assert values["lip_height_ratio"] == pytest.approx(0.05)


def test_pose_and_symmetry_features_are_passed_through_unchanged() -> None:
    """姿勢は度のまま、対称度は生値のまま入る（スケールし直さない）。"""
    features = build_features(
        _landmarks(),
        pose=FacePose(yaw_deg=1.5, pitch_deg=-2.5, roll_deg=3.5),
        symmetry_3d=_StubSymmetry3D(
            parts={"contour": 0.041, "eyes": 0.022, "nose": 0.033, "mouth": 0.044},
            overall=0.055,
            equal=0.066,
        ),
    )

    assert features is not None
    values = _named(features.as_vector())

    assert values["yaw_deg"] == pytest.approx(1.5)
    assert values["pitch_deg"] == pytest.approx(-2.5)
    assert values["roll_deg"] == pytest.approx(3.5)
    assert values["s3d_distance"] == pytest.approx(0.066)
    assert values["distance_contour"] == pytest.approx(0.041)
    assert values["distance_eyes"] == pytest.approx(0.022)
    assert values["distance_nose"] == pytest.approx(0.033)
    assert values["distance_mouth"] == pytest.approx(0.044)
    assert values["distance_overall"] == pytest.approx(0.055)


def test_depth_features_are_zero_for_a_flat_face() -> None:
    """z を全て 0 にした入力では、立体感の 6 特徴がちょうど 0 になる。"""
    features = build_features(_landmarks(), pose=FacePose(0.0, 0.0, 0.0), symmetry_3d=_symmetry())

    assert features is not None
    values = _named(features.as_vector())

    for name in (
        "nose_depth",
        "depth_forehead",
        "depth_chin",
        "depth_cheek_mean",
        "depth_spread",
        "contour_depth_range",
        "face_convexity",
    ):
        assert values[name] == pytest.approx(0.0), name


def test_depth_spread_reads_the_versioned_pair_pool_not_all_478_points() -> None:
    """`depth_spread` は対応ペアで使う 126 点の z の標準偏差である。

    罠: 478 点すべての std を取る実装は、対応表に無い点まで数えてしまい、
    学習時と違う値を返す。ここでは**対応表に無い点だけ**へ大きな z を入れ、
    それが値を動かさないことを見る。
    """
    baseline = build_features(_landmarks(), pose=FacePose(0.0, 0.0, 0.0), symmetry_3d=_symmetry())
    points = _landmarks()
    # 468〜477 は虹彩で、§8.5.1 の対応ペア表には含まれない。
    points[468:478, 2] = 50.0
    polluted = build_features(points, pose=FacePose(0.0, 0.0, 0.0), symmetry_3d=_symmetry())

    assert baseline is not None and polluted is not None
    assert _named(polluted.as_vector())["depth_spread"] == pytest.approx(
        _named(baseline.as_vector())["depth_spread"]
    )


@pytest.mark.parametrize(
    ("key", "position"),
    [
        ("chin", (0.0, -1.0, 0.0)),  # 顎を額と同じ高さへ → face_height = 0
        ("cheek_r", (-0.75, 0.25, 0.0)),  # 両頬を重ねる → face_width = 0
    ],
)
def test_build_features_returns_missing_for_a_degenerate_face(
    key: str, position: tuple[float, float, float]
) -> None:
    """テスト 5: 退化した顔は**例外ではなく欠測**を返す。

    呼び出し側が「判定できない」へ倒せるようにするためで、例外にすると
    失敗分離（不変条件 12）を application 側で書き直すことになる。
    """
    points = _landmarks()
    points[IDX[key]] = position

    assert build_features(points, pose=FacePose(0.0, 0.0, 0.0), symmetry_3d=_symmetry()) is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_build_features_returns_missing_for_non_finite_landmarks(bad: float) -> None:
    """テスト 6: NaN / Inf は欠測（不変条件 11）。0 で埋めない。"""
    points = _landmarks()
    points[IDX["nose_tip"], 2] = bad

    assert build_features(points, pose=FacePose(0.0, 0.0, 0.0), symmetry_3d=_symmetry()) is None


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_build_features_returns_missing_for_non_finite_pose_or_symmetry(bad: float) -> None:
    assert (
        build_features(_landmarks(), pose=FacePose(bad, 0.0, 0.0), symmetry_3d=_symmetry()) is None
    )
    assert (
        build_features(_landmarks(), pose=FacePose(0.0, 0.0, 0.0), symmetry_3d=_symmetry(bad))
        is None
    )


def test_build_features_returns_missing_when_a_symmetry_part_is_absent() -> None:
    """部位が 1 つでも欠測なら特徴ベクトルを作らない（0 で埋めない）。"""
    partial = _StubSymmetry3D(
        parts={"contour": 0.03, "eyes": None, "nose": 0.03, "mouth": 0.03},
        overall=0.03,
        equal=0.03,
    )

    assert build_features(_landmarks(), pose=FacePose(0.0, 0.0, 0.0), symmetry_3d=partial) is None


def test_build_features_rejects_a_wrong_landmark_count() -> None:
    assert (
        build_features(
            np.zeros((68, 3), dtype=np.float64),
            pose=FacePose(0.0, 0.0, 0.0),
            symmetry_3d=_symmetry(),
        )
        is None
    )


def test_build_features_does_not_mutate_its_input() -> None:
    points = _landmarks()
    before = points.copy()

    build_features(points, pose=FacePose(0.0, 0.0, 0.0), symmetry_3d=_symmetry())

    assert np.array_equal(points, before)


def test_decide_uses_strictly_greater_than_the_threshold() -> None:
    """テスト 7: `probability == threshold` は**範囲外**（`>` であり `>=` ではない）。

    spec の文言が `probabilities[:, 1] > threshold` なので、境界は範囲外側へ倒れる。
    安全側の既定と向きが一致している。
    """
    assert decide(0.2001, threshold=0.20, spec_version="age-gate/1.0.0").in_range is True
    assert decide(0.20, threshold=0.20, spec_version="age-gate/1.0.0").in_range is False
    assert decide(0.1999, threshold=0.20, spec_version="age-gate/1.0.0").in_range is False


def test_decide_carries_the_operating_point_it_used() -> None:
    """どの版のどの閾値で判定したかを、判定と一緒に持ち回る（§9.2）。"""
    decision = decide(0.42, threshold=0.20, spec_version="age-gate/1.0.0")

    assert isinstance(decision, AgeGateDecision)
    assert decision.probability_in_range == pytest.approx(0.42)
    assert decision.threshold == pytest.approx(0.20)
    assert decision.spec_version == "age-gate/1.0.0"
    assert decision.in_range is True


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 1.5])
def test_decide_rejects_an_impossible_probability(bad: float) -> None:
    """確率でない値を判定へ通さない（不変条件 11）。"""
    with pytest.raises(ValueError):
        decide(bad, threshold=0.20, spec_version="age-gate/1.0.0")


def test_the_frozen_pair_table_matches_the_bundle_it_was_copied_from() -> None:
    """`domain.age_gate` の固定コピーが `symmetry-3d/1.0.0` の対応表と一致する。

    `age-gate-features/1.0.0` は学習時の対応表に凍結されており、将来
    `symmetry-3d` 側の bundle が改訂されても変わってはならない。そのため
    domain は JSON を実行時に読まず固定コピーを持つ——**しかしその写しが
    正しいことは誰も見ていなかった。** 63 ペアのうち 1 つを写し間違えても、
    contract test は固定の特徴ベクトルを流すだけなので気づけず、`depth_spread`
    と `contour_depth_range` だけが学習時と違う値になる。

    比較先はファイル名で版が固定された `symmetry-3d-v1.json` である。指標が
    新しい版になれば別ファイル（`symmetry-3d-v2.json`）になるので、この test は
    改訂に引きずられない。
    """
    import json
    from pathlib import Path

    from face_symmetry_lab.domain.age_gate import _PART_PAIRS

    bundle = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs" / "metrics" / "symmetry-3d-v1.json"
        ).read_text(encoding="utf-8")
    )

    assert bundle["metric_version"] == "symmetry-3d/1.0.0"
    assert set(_PART_PAIRS) == set(PART_NAMES)
    for part in PART_NAMES:
        expected = tuple(tuple(pair) for pair in bundle["parts"][part]["pairs"])
        assert _PART_PAIRS[part] == expected, part

    flattened = [index for part in PART_NAMES for pair in _PART_PAIRS[part] for index in pair]
    assert len(flattened) == 126
    assert len(set(flattened)) == 126, "同じ landmark が 2 度使われてはならない"
