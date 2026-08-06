import pytest

from face_symmetry_lab.domain.auto_tuning import (
    ANALYSIS_INTERVAL_CHOICES_MS,
    MAX_AGE_INTERVAL_MS,
    MAX_HISTORY_CAPACITY,
    MIN_AGE_INTERVAL_MS,
    MIN_HISTORY_CAPACITY,
    TARGET_MEAN_LUMINANCE,
    AutoTuningError,
    recommend_age_interval_ms,
    recommend_analysis_interval_ms,
    recommend_brightness_gain,
    recommend_history_capacity,
    recommend_zoom_policy,
)
from face_symmetry_lab.domain.frame_adjustments import MAX_BRIGHTNESS_GAIN, MIN_BRIGHTNESS_GAIN
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy


def test_a_frame_already_at_target_luminance_recommends_a_gain_of_one() -> None:
    """§14.3b: 計測対象は常に生フレームであり（補正後ではない）、したがって
    推奨値は相対的な更新ではなく、絶対的な目標 gain である。"""
    gain = recommend_brightness_gain(TARGET_MEAN_LUMINANCE)

    assert gain == pytest.approx(1.0)


def test_a_dark_frame_recommends_a_higher_gain() -> None:
    gain = recommend_brightness_gain(TARGET_MEAN_LUMINANCE / 2.0)

    assert gain == pytest.approx(2.0)


def test_a_bright_frame_recommends_a_lower_gain() -> None:
    gain = recommend_brightness_gain(TARGET_MEAN_LUMINANCE * 2.0)

    assert gain == pytest.approx(0.5)


def test_the_recommended_gain_never_exceeds_frs_028_bounds() -> None:
    assert recommend_brightness_gain(1.0) == pytest.approx(MAX_BRIGHTNESS_GAIN)
    assert recommend_brightness_gain(2000.0) == pytest.approx(MIN_BRIGHTNESS_GAIN)


@pytest.mark.parametrize("measured", [0.0, -1.0, float("nan"), float("inf")])
def test_an_invalid_measured_luminance_is_refused(measured: float) -> None:
    with pytest.raises(AutoTuningError):
        recommend_brightness_gain(measured)


def test_a_face_at_the_reference_area_ratio_returns_the_default_policy() -> None:
    default = TrackingZoomPolicy()

    policy = recommend_zoom_policy(0.12)

    assert policy.target_px == default.target_px
    assert policy.margin_ratio == pytest.approx(default.margin_ratio)


def test_a_small_face_recommends_a_larger_target_and_margin() -> None:
    default = TrackingZoomPolicy()

    policy = recommend_zoom_policy(0.03)

    assert policy.target_px > default.target_px
    assert policy.margin_ratio > default.margin_ratio


def test_a_large_face_recommends_a_smaller_target_and_margin() -> None:
    default = TrackingZoomPolicy()

    policy = recommend_zoom_policy(0.5)

    assert policy.target_px < default.target_px
    assert policy.margin_ratio < default.margin_ratio


def test_the_recommended_policy_always_satisfies_frs_026_bounds() -> None:
    for ratio in (0.001, 0.01, 0.12, 0.55, 1.0):
        # TrackingZoomPolicy.__post_init__ が範囲外の値に対してすでに例外を
        # 送出するので、構築が成功することそのものが契約である。
        recommend_zoom_policy(ratio)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_an_invalid_face_area_ratio_is_refused(ratio: float) -> None:
    with pytest.raises(AutoTuningError):
        recommend_zoom_policy(ratio)


def test_light_processing_load_recommends_the_fastest_interval() -> None:
    assert recommend_analysis_interval_ms(1.0) == min(ANALYSIS_INTERVAL_CHOICES_MS)


def test_heavy_processing_load_recommends_the_slowest_interval() -> None:
    assert recommend_analysis_interval_ms(10_000.0) == max(ANALYSIS_INTERVAL_CHOICES_MS)


def test_analysis_interval_is_always_one_of_the_documented_choices() -> None:
    for load in (0.0, 50.0, 100.0, 150.0, 250.0, 500.0, 1000.0):
        assert recommend_analysis_interval_ms(load) in ANALYSIS_INTERVAL_CHOICES_MS


@pytest.mark.parametrize("load", [-1.0, float("nan"), float("inf")])
def test_an_invalid_processing_load_is_refused(load: float) -> None:
    with pytest.raises(AutoTuningError):
        recommend_analysis_interval_ms(load)


def test_age_interval_scales_with_the_analysis_interval() -> None:
    fast = recommend_age_interval_ms(125)
    slow = recommend_age_interval_ms(333)

    assert MIN_AGE_INTERVAL_MS <= fast < slow <= MAX_AGE_INTERVAL_MS


def test_age_interval_is_clamped_to_its_bounds() -> None:
    assert recommend_age_interval_ms(1) == MIN_AGE_INTERVAL_MS
    assert recommend_age_interval_ms(1_000_000) == MAX_AGE_INTERVAL_MS


@pytest.mark.parametrize("interval_ms", [0, -1])
def test_an_invalid_analysis_interval_for_age_is_refused(interval_ms: int) -> None:
    with pytest.raises(AutoTuningError):
        recommend_age_interval_ms(interval_ms)


def test_history_capacity_grows_as_the_interval_shrinks() -> None:
    fast = recommend_history_capacity(125)
    slow = recommend_history_capacity(333)

    assert MIN_HISTORY_CAPACITY <= slow < fast <= MAX_HISTORY_CAPACITY


def test_history_capacity_is_clamped_to_its_bounds() -> None:
    assert recommend_history_capacity(1) == MAX_HISTORY_CAPACITY
    assert recommend_history_capacity(1_000_000) == MIN_HISTORY_CAPACITY


@pytest.mark.parametrize("interval_ms", [0, -1])
def test_an_invalid_analysis_interval_for_history_is_refused(interval_ms: int) -> None:
    with pytest.raises(AutoTuningError):
        recommend_history_capacity(interval_ms)
