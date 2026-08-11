import dataclasses

import numpy as np
import pytest

from face_symmetry_lab.application.analyzer import FrameAnalyzer
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.models import ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose

# 640x480 のフレーム内にある、対称な顔の正規化ランドマーク。
_BASE_POINTS = (
    (0.40, 0.40),  # 0 左目 外側
    (0.45, 0.40),  # 1 左目 内側
    (0.55, 0.40),  # 2 右目 内側
    (0.60, 0.40),  # 3 右目 外側
    (0.50, 0.30),  # 4 正中
    (0.50, 0.50),  # 5 正中
    (0.50, 0.62),  # 6 正中
    (0.35, 0.55),  # 7 輪郭
    (0.65, 0.55),  # 8
    (0.36, 0.45),  # 9
    (0.64, 0.45),  # 10
    (0.38, 0.35),  # 11
    (0.62, 0.35),  # 12
    (0.47, 0.50),  # 13 鼻
    (0.53, 0.50),  # 14
    (0.46, 0.48),  # 15
    (0.54, 0.48),  # 16
    (0.48, 0.52),  # 17
    (0.52, 0.52),  # 18
    (0.44, 0.58),  # 19 口
    (0.56, 0.58),  # 20
    (0.47, 0.57),  # 21
    (0.53, 0.57),  # 22
    (0.48, 0.60),  # 23
    (0.52, 0.60),  # 24
    (0.42, 0.36),  # 25 眉
    (0.58, 0.36),  # 26
)


def _scaled(points, factor: float = 1.3):
    """フレーム中心を基準に顔を拡大し、顔面積のゲートを満たすようにする。"""
    return tuple((0.5 + (x - 0.5) * factor, 0.5 + (y - 0.5) * factor) for x, y in points)


SYMMETRIC_POINTS = _scaled(_BASE_POINTS)


def _config():
    return build_metric_config(
        metric_version="symmetry-2d/test",
        landmark_map_version="test/1.0.0",
        landmark_count=len(SYMMETRIC_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=3),
            "eyes": PartMap(pairs=((0, 3), (1, 2), (25, 26)), min_pairs=3),
            "nose": PartMap(pairs=((13, 14), (15, 16), (17, 18)), min_pairs=3),
            "mouth": PartMap(pairs=((19, 20), (21, 22), (23, 24)), min_pairs=3),
        },
    )


def _config_3d():
    """§8.5.1 の並走 config。2D 側と同じ landmark index を使い、`metric_version`
    だけを 3D のものにする（本物の bundle が 2 つの JSON を共有するのと同じ形）。
    """
    return build_metric_config(
        metric_version="symmetry-3d/test",
        landmark_map_version="test/1.0.0",
        landmark_count=len(SYMMETRIC_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=3),
            "eyes": PartMap(pairs=((0, 3), (1, 2), (25, 26)), min_pairs=3),
            "nose": PartMap(pairs=((13, 14), (15, 16), (17, 18)), min_pairs=3),
            "mouth": PartMap(pairs=((19, 20), (21, 22), (23, 24)), min_pairs=3),
        },
    )


def _frame(points=SYMMETRIC_POINTS, *, frame_id=1, yaw=0.0, face_count=1):
    return build_landmark_frame(
        frame_id=frame_id,
        face_count=face_count,
        points=points if face_count == 1 else (),
        pose=FacePose(yaw_deg=yaw, pitch_deg=0.0, roll_deg=0.0) if face_count == 1 else None,
        width=640,
        height=480,
    )


def _analyzer(**kwargs) -> FrameAnalyzer:
    return FrameAnalyzer(
        config=_config(),
        config_3d=_config_3d(),
        skin_pairs=((7, 8), (9, 10), (11, 12)),
        **kwargs,
    )


def _drive_to_ok(
    analyzer: FrameAnalyzer,
    *,
    start: float = 0.0,
    rgb_image: np.ndarray | None = None,
):
    outcome = None
    for index in range(1, 6):
        outcome = analyzer.analyze(
            _frame(frame_id=index),
            at=start + index * 0.1,
            rgb_image=rgb_image,
        )
    assert outcome is not None
    return outcome


def test_no_distance_is_produced_before_the_gate_opens() -> None:
    analyzer = _analyzer()

    outcome = analyzer.analyze(_frame(), at=0.0)

    assert outcome.state.status is Status.HOLD
    assert outcome.symmetry is None
    assert outcome.overall_distance_raw is None
    assert outcome.overall_distance_smooth is None


def test_a_symmetric_face_measures_zero_distance_once_the_gate_opens() -> None:
    outcome = _drive_to_ok(_analyzer())

    assert outcome.state.status is Status.OK
    assert outcome.overall_distance_raw == pytest.approx(0.0, abs=1e-9)
    assert outcome.symmetry is not None
    assert outcome.symmetry.distance("mouth") == pytest.approx(0.0, abs=1e-9)


def test_the_smoothed_value_starts_at_the_first_raw_distance() -> None:
    outcome = _drive_to_ok(_analyzer())

    assert outcome.overall_distance_smooth == pytest.approx(outcome.overall_distance_raw, abs=1e-6)


def test_the_smoothed_value_lags_a_change_in_the_raw_distance() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    skewed = list(SYMMETRIC_POINTS)
    skewed[19] = (0.20, SYMMETRIC_POINTS[19][1])
    outcome = analyzer.analyze(_frame(tuple(skewed), frame_id=6), at=0.65)

    # 生の距離は跳ね上がったが、EMA はまだそれに追随しきれていない。ただし
    # 出発点である対称な顔の 0 よりは上にある。
    assert outcome.overall_distance_raw is not None
    assert outcome.overall_distance_smooth is not None
    assert outcome.overall_distance_raw > outcome.overall_distance_smooth > 0.0


def test_an_off_axis_frame_stops_producing_distances() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    outcome = analyzer.analyze(_frame(frame_id=6, yaw=30.0), at=0.7)

    assert outcome.state.status is Status.HOLD
    assert ReasonCode.OFF_AXIS_YAW in outcome.state.reason_codes
    assert outcome.overall_distance_raw is None


def test_multi_face_frames_never_measure() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    outcome = analyzer.analyze(_frame(frame_id=6, face_count=2), at=0.7)

    assert outcome.state.status is Status.MULTI_FACE
    assert outcome.symmetry is None


def test_a_prolonged_face_loss_resets_the_smoothing() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    analyzer.analyze(_frame(frame_id=6, face_count=0), at=1.0)
    analyzer.analyze(_frame(frame_id=7, face_count=0), at=2.0)
    recovered = None
    for index in range(8, 13):
        recovered = analyzer.analyze(_frame(frame_id=index), at=2.0 + index * 0.1)

    assert recovered is not None
    assert recovered.overall_distance_smooth == pytest.approx(
        recovered.overall_distance_raw, abs=1e-6
    )


def test_degenerate_landmarks_become_an_error_rather_than_a_distance() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    collapsed = list(SYMMETRIC_POINTS)
    collapsed[2] = collapsed[0]
    collapsed[3] = collapsed[1]
    outcome = analyzer.analyze(_frame(tuple(collapsed), frame_id=6), at=0.7)

    assert outcome.state.status is Status.ERROR
    assert outcome.symmetry is None
    assert outcome.overall_distance_smooth is None


def test_reset_clears_the_gate_and_the_smoothing() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    analyzer.reset()
    outcome = analyzer.analyze(_frame(frame_id=6), at=1.0)

    assert outcome.state.status is Status.HOLD
    assert outcome.overall_distance_smooth is None


def test_the_outcome_carries_the_versions_needed_to_reproduce_it() -> None:
    outcome = _drive_to_ok(_analyzer())

    assert outcome.metric_version == "symmetry-2d/test"
    assert outcome.landmark_map_version == "test/1.0.0"


def test_the_outcome_reports_pose_and_face_area() -> None:
    outcome = _drive_to_ok(_analyzer())

    assert outcome.pose is not None
    assert outcome.pose.yaw_deg == pytest.approx(0.0)
    assert outcome.face_area_ratio is not None
    assert 0.0 < outcome.face_area_ratio < 1.0


def test_tau_seconds_reads_back_the_currently_active_value() -> None:
    analyzer = _analyzer()

    analyzer.update(thresholds=QualityThresholds(), tau_seconds=0.9)

    assert analyzer.tau_seconds == pytest.approx(0.9)


def test_update_swaps_thresholds_and_resets_the_gate_and_smoothing() -> None:
    """§15.2: 新しい Advanced 設定の適用が、古い EMA 履歴を残してはならない。"""
    analyzer = _analyzer()
    _drive_to_ok(analyzer)  # ゲートは OK で、EMA は値を持つ

    analyzer.update(thresholds=QualityThresholds(valid_frames_to_ok=1), tau_seconds=0.2)
    outcome = analyzer.analyze(_frame(frame_id=99), at=100.0)

    assert analyzer.thresholds.valid_frames_to_ok == 1
    assert outcome.state.status is Status.OK  # 新しい閾値が即座に適用される
    assert (
        outcome.overall_distance_smooth == outcome.overall_distance_raw
    )  # 平滑化の履歴がリセットされた


def test_update_cannot_swap_the_pinned_metric_bundle() -> None:
    """§8.6/§15.2: bundle に調整対象が残っていないので、固定されたままである。"""
    analyzer = _analyzer()
    original = analyzer.config

    analyzer.update(thresholds=QualityThresholds(), tau_seconds=0.2)

    assert analyzer.config is original


def test_custom_thresholds_change_how_quickly_the_gate_opens() -> None:
    analyzer = _analyzer(thresholds=QualityThresholds(valid_frames_to_ok=1))

    outcome = analyzer.analyze(_frame(), at=0.0)

    assert outcome.state.status is Status.OK


def test_skin_surface_reference_is_separate_from_the_geometry_overall() -> None:
    mirrored = np.full((480, 640, 3), (150, 120, 105), dtype=np.uint8)
    analyzer = _analyzer()

    baseline = _drive_to_ok(analyzer, rgb_image=mirrored)
    changed_image = mirrored.copy()
    rng = np.random.default_rng(42)
    changed_image[250:290, 180:220] = rng.integers(
        0,
        256,
        size=(40, 40, 3),
        dtype=np.uint8,
    )
    changed = analyzer.analyze(_frame(frame_id=6), at=0.7, rgb_image=changed_image)

    assert baseline.skin_quality is not None
    assert baseline.skin_quality.score == pytest.approx(100.0, abs=1e-6)
    assert changed.skin_quality is not None
    assert changed.skin_quality.score is not None
    assert changed.skin_quality.score < baseline.skin_quality.score
    assert changed.overall_distance_raw == pytest.approx(baseline.overall_distance_raw, abs=1e-6)


# --------------------------------------------------------------------------
# 推定奥行き z の中立性（§8.1.1）
# --------------------------------------------------------------------------


def _depths(points) -> tuple[float, ...]:
    """指標が z を読んでいれば必ず値が動く、非対称で非自明な奥行き。"""
    return tuple(0.05 * (x - 0.5) - 0.02 * y for x, y in points)


def test_carrying_a_depth_does_not_move_any_measured_value() -> None:
    """§8.1.1: §8.3〜§8.6 は xy だけで定義される。z の有無で値が変わらない。

    `symmetry_3d`・`overall_distance_3d_smooth` はこの比較から除く。3D 指標
    （§8.5.1、FR-036）は z を読むことこそが目的なので、この 2 つだけは z の
    有無で異なることが正しい。除いた残り全部が一致することが「2D は z を
    読まない」の証拠になる。
    """
    rgb_image = np.full((480, 640, 3), (150, 120, 105), dtype=np.uint8)
    depths = _depths(SYMMETRIC_POINTS)

    without_depth = _drive_to_ok(_analyzer(), rgb_image=rgb_image)

    with_depth_analyzer = _analyzer()
    outcome = None
    for index in range(1, 6):
        frame = build_landmark_frame(
            frame_id=index,
            face_count=1,
            points=SYMMETRIC_POINTS,
            depths=depths,
            pose=FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
            width=640,
            height=480,
        )
        outcome = with_depth_analyzer.analyze(frame, at=index * 0.1, rgb_image=rgb_image)

    assert outcome is not None
    assert outcome.symmetry_3d is not None  # z があるので 3D 側は算出される
    assert (
        dataclasses.replace(outcome, symmetry_3d=None, overall_distance_3d_smooth=None)
        == without_depth
    )


def test_an_asymmetric_face_measures_the_same_distance_with_and_without_depth() -> None:
    """対称な顔だけで比べると 0 同士の一致になるため、ズレのある顔でも確かめる。

    `symmetry_3d`・`overall_distance_3d_smooth` は z の有無で異なることが正しい
    ため比較から除く（`test_carrying_a_depth_does_not_move_any_measured_value`
    と同じ理由）。
    """
    skewed = tuple(
        (x + 0.01, y) if index in {8, 10, 12} else (x, y)
        for index, (x, y) in enumerate(SYMMETRIC_POINTS)
    )

    def measure(depths: tuple[float, ...] | None):
        analyzer = _analyzer()
        outcome = None
        for index in range(1, 6):
            outcome = analyzer.analyze(
                build_landmark_frame(
                    frame_id=index,
                    face_count=1,
                    points=skewed,
                    depths=depths,
                    pose=FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
                    width=640,
                    height=480,
                ),
                at=index * 0.1,
            )
        assert outcome is not None
        return outcome

    without_depth = measure(None)
    with_depth = measure(_depths(skewed))

    assert without_depth.overall_distance_raw is not None
    assert without_depth.overall_distance_raw > 0.0
    assert with_depth.symmetry_3d is not None  # z があるので 3D 側は算出される
    assert (
        dataclasses.replace(with_depth, symmetry_3d=None, overall_distance_3d_smooth=None)
        == without_depth
    )


# --------------------------------------------------------------------------
# 3D object symmetry の並走（§8.5.1, §8.6.5, FR-036）
# --------------------------------------------------------------------------


def _depth_frame(
    points,
    *,
    frame_id: int,
    depths: tuple[float, ...],
) -> LandmarkFrame:
    return build_landmark_frame(
        frame_id=frame_id,
        face_count=1,
        points=points,
        depths=depths,
        pose=FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
        width=640,
        height=480,
    )


def test_the_outcome_carries_the_three_dimensional_result_and_its_smoothed_distance() -> None:
    """§8.5.1/§8.6.5/FR-036: 3D は並走するだけだが、`AnalysisOutcome` までは届く。"""
    depths = _depths(SYMMETRIC_POINTS)
    analyzer = _analyzer()
    outcome = None
    for index in range(1, 6):
        outcome = analyzer.analyze(
            _depth_frame(SYMMETRIC_POINTS, frame_id=index, depths=depths),
            at=index * 0.1,
        )

    assert outcome is not None
    assert outcome.state.status is Status.OK
    assert outcome.symmetry_3d is not None
    assert outcome.symmetry_3d.overall is not None
    assert outcome.symmetry_3d.overall.distance > 0.0  # z が非対称なので 0 ではない
    assert outcome.overall_distance_3d_smooth is not None
    assert outcome.overall_distance_3d_smooth == pytest.approx(
        outcome.symmetry_3d.overall.distance, abs=1e-6
    )


def test_missing_depth_leaves_only_the_three_dimensional_side_missing() -> None:
    """回帰（AC-SCORE-09）: z 欠測のフレームは 3D 側だけを欠測にし、2D の距離・
    EMA・state・肌表面参考値は z のあるフレームと完全に一致する。
    """
    rgb_image = np.full((480, 640, 3), (150, 120, 105), dtype=np.uint8)
    depths = _depths(SYMMETRIC_POINTS)

    with_depth_analyzer = _analyzer()
    with_depth = None
    for index in range(1, 6):
        with_depth = with_depth_analyzer.analyze(
            _depth_frame(SYMMETRIC_POINTS, frame_id=index, depths=depths),
            at=index * 0.1,
            rgb_image=rgb_image,
        )

    without_depth = _drive_to_ok(_analyzer(), rgb_image=rgb_image)

    assert with_depth is not None
    assert with_depth.symmetry_3d is not None
    assert with_depth.overall_distance_3d_smooth is not None
    assert without_depth.symmetry_3d is None
    assert without_depth.overall_distance_3d_smooth is None

    assert with_depth.state == without_depth.state
    assert with_depth.symmetry == without_depth.symmetry
    assert with_depth.overall_distance_smooth == without_depth.overall_distance_smooth
    assert with_depth.skin_quality == without_depth.skin_quality
    assert with_depth.face_area_ratio == without_depth.face_area_ratio


def test_a_three_dimensional_failure_leaves_the_frame_and_two_dimensional_ema_untouched() -> None:
    """3D の失敗（ここでは z 欠測）は Status・品質ゲートの連続性・2D の EMA の
    いずれも変えない（不変条件 12）。既存の 2D 失敗時の処理（`Status.ERROR` に
    してゲートと EMA をリセットする）を 3D へコピーしていないことの回帰。
    """
    depths = _depths(SYMMETRIC_POINTS)
    analyzer = _analyzer()
    settled = None
    for index in range(1, 6):
        settled = analyzer.analyze(
            _depth_frame(SYMMETRIC_POINTS, frame_id=index, depths=depths),
            at=index * 0.1,
        )
    assert settled is not None
    assert settled.state.status is Status.OK
    assert settled.symmetry_3d is not None

    # z を欠いた非対称フレームを 1 枚だけ挟む。3D は欠測になるが、Status は
    # ERROR にならず、2D の EMA は（リセットされていれば raw と一致するはずの
    # ところ）依然として直前の値へ引きずられている。
    skewed = list(SYMMETRIC_POINTS)
    skewed[19] = (0.20, SYMMETRIC_POINTS[19][1])
    failed = analyzer.analyze(_frame(tuple(skewed), frame_id=6), at=0.6)

    assert failed.state.status is Status.OK
    assert failed.symmetry_3d is None
    assert failed.symmetry is not None
    assert failed.overall_distance_raw is not None
    assert failed.overall_distance_smooth is not None
    assert failed.overall_distance_raw > failed.overall_distance_smooth > 0.0

    # ゲートがリセットされていれば、既定 valid_frames_to_ok=5 では 1 枚では
    # Status.OK に戻れない。1 枚だけで即座に OK であることが連続性の証拠になる。
    recovered = analyzer.analyze(
        _depth_frame(SYMMETRIC_POINTS, frame_id=7, depths=depths),
        at=0.7,
    )
    assert recovered.state.status is Status.OK


def test_reset_also_restarts_the_three_dimensional_smoothing() -> None:
    """`reset()` の後、3D の EMA も立ち上がり直す（前のセッションの値が漏れない）。"""
    depths = _depths(SYMMETRIC_POINTS)
    skewed = list(SYMMETRIC_POINTS)
    skewed[19] = (0.20, SYMMETRIC_POINTS[19][1])

    analyzer = _analyzer()
    # まず対称な顔でゲートを開く（既定 valid_frames_to_ok=5）。
    for index in range(1, 6):
        analyzer.analyze(
            _depth_frame(SYMMETRIC_POINTS, frame_id=index, depths=depths), at=index * 0.1
        )

    # ゲートが開いた状態のまま、非対称な顔を 1 枚だけ与える。3D の生値が跳ぶが、
    # EMA はまだ追いつききっていない（ラグが残っていることが継続の証拠）。
    before_reset = analyzer.analyze(_depth_frame(tuple(skewed), frame_id=6, depths=depths), at=0.6)

    assert before_reset.symmetry_3d is not None
    assert before_reset.symmetry_3d.overall is not None
    assert before_reset.overall_distance_3d_smooth is not None
    assert before_reset.overall_distance_3d_smooth != pytest.approx(
        before_reset.symmetry_3d.overall.distance, abs=1e-9
    )

    analyzer.reset()
    # reset 後は既定 valid_frames_to_ok=5 のゲートをもう一度満たす必要がある。
    after_reset = None
    for index in range(7, 12):
        after_reset = analyzer.analyze(
            _depth_frame(tuple(skewed), frame_id=index, depths=depths), at=1.0 + index * 0.1
        )

    assert after_reset is not None
    assert after_reset.state.status is Status.OK
    assert after_reset.symmetry_3d is not None
    assert after_reset.symmetry_3d.overall is not None
    assert after_reset.overall_distance_3d_smooth is not None
    # reset 後にゲートが開いた瞬間（= 3D EMA への最初の更新）は必ず生値と一致
    # する。前セッションの値が漏れていない証拠である。
    assert after_reset.overall_distance_3d_smooth == pytest.approx(
        after_reset.symmetry_3d.overall.distance, abs=1e-6
    )


def test_a_prolonged_face_loss_resets_the_three_dimensional_smoothing_too() -> None:
    """顔を `FACE_LOST_RESET_SECONDS` 以上見失うと、2D と同じく 3D の EMA も
    リセットされる。"""
    depths = _depths(SYMMETRIC_POINTS)
    skewed = list(SYMMETRIC_POINTS)
    skewed[19] = (0.20, SYMMETRIC_POINTS[19][1])

    analyzer = _analyzer()
    for index in range(1, 6):
        analyzer.analyze(_depth_frame(tuple(skewed), frame_id=index, depths=depths), at=index * 0.1)

    analyzer.analyze(_frame(frame_id=6, face_count=0), at=1.0)
    analyzer.analyze(_frame(frame_id=7, face_count=0), at=2.0)
    recovered = None
    for index in range(8, 13):
        recovered = analyzer.analyze(
            _depth_frame(tuple(skewed), frame_id=index, depths=depths), at=2.0 + index * 0.1
        )

    assert recovered is not None
    assert recovered.symmetry_3d is not None
    assert recovered.symmetry_3d.overall is not None
    assert recovered.overall_distance_3d_smooth is not None
    assert recovered.overall_distance_3d_smooth == pytest.approx(
        recovered.symmetry_3d.overall.distance, abs=1e-6
    )
