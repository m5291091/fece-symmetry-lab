import numpy as np
import pytest

from face_symmetry_lab.application.auto_tune_controller import (
    CADENCE_RETUNE_INTERVAL_SECONDS,
    DISPLAY_RETUNE_INTERVAL_SECONDS,
    MIN_CADENCE_SAMPLES,
    AutoTuneController,
)
from face_symmetry_lab.domain.landmarks import build_landmark_frame

#: `face_area_ratio` が退化しない有効な比になるよう広げてある。
SPREAD_POINTS = ((0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6))


def _controller(*, enabled: bool = True) -> AutoTuneController:
    return AutoTuneController(enabled=enabled, started_at=0.0)


def _frame(points: tuple[tuple[float, float], ...] = SPREAD_POINTS):
    return build_landmark_frame(
        frame_id=1,
        face_count=1 if points else 0,
        points=points,
        width=100,
        height=100,
    )


def _flat_rgb(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_a_disabled_controller_recommends_nothing_at_all() -> None:
    """AC-AUTOTUNE-01/02: 自動調整が無効なら、何一つ推奨されない。"""
    controller = _controller(enabled=False)

    display = controller.display_tuning(_flat_rgb(64), _frame(), at=0.0)

    assert display.brightness_gain is None
    assert display.zoom_policy is None
    assert controller.enabled is False


def test_display_tuning_recommends_a_gain_from_the_raw_frame_luminance() -> None:
    tuning = _controller().display_tuning(_flat_rgb(64), _frame(), at=0.0)

    assert tuning.brightness_gain == pytest.approx(2.0)  # 目標 128 / 実測 64
    assert tuning.zoom_policy is not None


def test_brightness_and_zoom_share_one_throttle() -> None:
    """どちらも同じ瞬間の同じフレームに由来する。したがって間隔内の 2 回目の
    呼び出しが、片方だけを再推奨してはならない。"""
    controller = _controller()

    controller.display_tuning(_flat_rgb(64), _frame(), at=0.0)
    within = controller.display_tuning(
        _flat_rgb(64), _frame(), at=DISPLAY_RETUNE_INTERVAL_SECONDS / 2
    )
    after = controller.display_tuning(_flat_rgb(64), _frame(), at=DISPLAY_RETUNE_INTERVAL_SECONDS)

    assert within.brightness_gain is None
    assert within.zoom_policy is None
    assert after.brightness_gain is not None
    assert after.zoom_policy is not None


def test_an_unusable_frame_leaves_the_gain_alone_without_raising() -> None:
    """AC-AUTOTUNE-01: 計測の失敗は「変更なし」として握り潰す。"""
    tuning = _controller().display_tuning(_flat_rgb(0), _frame(), at=0.0)

    assert tuning.brightness_gain is None  # 完全な黒フレームには有効な gain が無い


def test_zoom_is_not_recommended_without_a_confirmed_single_face() -> None:
    tuning = _controller().display_tuning(_flat_rgb(64), _frame(points=()), at=0.0)

    assert tuning.zoom_policy is None
    assert tuning.brightness_gain is not None  # 明るさ側は引き続き機能する


def test_cadence_needs_a_minimum_number_of_samples_before_recommending() -> None:
    controller = _controller()

    for index in range(MIN_CADENCE_SAMPLES - 1):
        recommended = controller.cadence_tuning(
            300.0, at=index * CADENCE_RETUNE_INTERVAL_SECONDS, current_interval_ms=125
        )
        assert recommended is None


def test_a_sustained_heavy_load_recommends_a_slower_interval() -> None:
    controller = _controller()
    at = 0.0
    recommended = None

    for _ in range(MIN_CADENCE_SAMPLES + 1):
        recommended = controller.cadence_tuning(300.0, at=at, current_interval_ms=125)
        at += CADENCE_RETUNE_INTERVAL_SECONDS

    assert recommended == 333


def test_cadence_returns_none_when_the_recommendation_already_holds() -> None:
    """AC-AUTOTUNE-02: 変化のない答えが recorder を無駄に作り直してはならない。"""
    controller = _controller()
    at = 0.0
    recommended = None

    for _ in range(MIN_CADENCE_SAMPLES + 1):
        recommended = controller.cadence_tuning(300.0, at=at, current_interval_ms=333)
        at += CADENCE_RETUNE_INTERVAL_SECONDS

    assert recommended is None


def test_the_controller_exposes_no_scoring_affecting_recommendation() -> None:
    """§8.6/AC-AUTOTUNE-02: ここから測定値へ到達できてはならない。

    指標 bundle は感度係数を失った。したがって controller が、利用者の与り
    知らぬところで `MetricConfig` を差し替え config_revision/区間/EMA を
    リセットする経路を、再び生やしてはならない。
    """
    controller = _controller()

    assert not hasattr(controller, "sensitivity_tuning")


def test_a_disabled_controller_never_recommends_cadence() -> None:
    controller = _controller(enabled=False)
    at = 0.0

    for _ in range(MIN_CADENCE_SAMPLES + 2):
        assert controller.cadence_tuning(300.0, at=at, current_interval_ms=125) is None
        at += CADENCE_RETUNE_INTERVAL_SECONDS
