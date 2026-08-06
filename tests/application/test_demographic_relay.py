"""年齢・性別中継役単体のテスト（§14.3, FR-022〜024）。

左右差の解析から独立していること、表示 OFF で worker へ触れないこと、
再表示に `is_cached` の印が付くことを、pipeline を通さずに確認する。
"""

from datetime import UTC, datetime

import numpy as np

from face_symmetry_lab.application.demographic_relay import DemographicRelay
from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.landmarks import build_landmark_frame

#: `extract_face_crop` が退化しない bbox を導けるよう広げた点。
SPREAD_POINTS = ((0.5, 0.2), (0.5, 0.5), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5))


def _estimate(age: float = 34.0):
    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(0.8, 0.2),
        model_version="age-gender/test",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )


class FakeWorker:
    def __init__(self, due: bool = True) -> None:
        self.due = due
        self.submitted: list[np.ndarray] = []
        self.result: tuple[object, int] | None = None
        self.intervals: list[int] = []
        self.resets = 0
        self.closed = False

    def is_due(self) -> bool:
        return self.due

    def submit(self, rgb_crop: np.ndarray) -> object:
        self.submitted.append(rgb_crop)
        return None

    def latest(self):
        return self.result

    def set_interval_ms(self, interval_ms: int) -> None:
        self.intervals.append(interval_ms)

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


def _frame(face_count: int = 1):
    points = SPREAD_POINTS if face_count == 1 else ()
    return build_landmark_frame(
        frame_id=1,
        face_count=face_count,
        points=points,
        width=64,
        height=64,
    )


def _image() -> np.ndarray:
    return np.full((64, 64, 3), 40, dtype=np.uint8)


def test_a_relay_without_a_worker_reports_the_feature_as_unavailable() -> None:
    """モデル未配置でも呼び出し側に分岐が増えないようにする。"""
    relay = DemographicRelay(None)

    assert relay.available is False
    assert relay.collect(timestamp_ms=10) is None
    relay.submit(_image(), _frame())  # 例外を投げない
    relay.set_interval_ms(1000)
    relay.reset()
    relay.close()


def test_a_due_worker_receives_a_single_face_crop() -> None:
    worker = FakeWorker(due=True)
    relay = DemographicRelay(worker)  # type: ignore[arg-type]

    relay.submit(_image(), _frame())

    assert len(worker.submitted) == 1


def test_a_worker_that_is_not_due_is_never_paid_for_a_crop() -> None:
    """worker の周期は解析周期よりずっと遅いので、先に問い合わせる。"""
    worker = FakeWorker(due=False)
    relay = DemographicRelay(worker)  # type: ignore[arg-type]

    relay.submit(_image(), _frame())

    assert worker.submitted == []


def test_a_multi_face_frame_is_not_submitted() -> None:
    worker = FakeWorker(due=True)
    relay = DemographicRelay(worker)  # type: ignore[arg-type]

    relay.submit(_image(), _frame(face_count=2))

    assert worker.submitted == []


def test_disabling_the_display_stops_submission_and_collection() -> None:
    """§15.1: 利用者が切っている間は worker に触れない。"""
    worker = FakeWorker(due=True)
    worker.result = (_estimate(), 1)
    relay = DemographicRelay(worker)  # type: ignore[arg-type]

    relay.set_enabled(False)
    relay.submit(_image(), _frame())

    assert worker.submitted == []
    assert relay.collect(timestamp_ms=10) is None


def test_a_fresh_version_is_reported_as_a_new_observation() -> None:
    worker = FakeWorker()
    worker.result = (_estimate(), 1)
    relay = DemographicRelay(worker)  # type: ignore[arg-type]

    collected = relay.collect(timestamp_ms=120)

    assert collected is not None
    estimate, observed_elapsed_ms = collected
    assert estimate.is_cached is False
    assert observed_elapsed_ms == 120


def test_redisplaying_the_same_version_is_marked_as_cached() -> None:
    """FR-023: 新規推論と、同じ区間の直前結果の再表示を区別する。"""
    worker = FakeWorker()
    worker.result = (_estimate(), 1)
    relay = DemographicRelay(worker)  # type: ignore[arg-type]
    relay.collect(timestamp_ms=120)

    collected = relay.collect(timestamp_ms=250)

    assert collected is not None
    estimate, observed_elapsed_ms = collected
    assert estimate.is_cached is True
    # 再表示は最初に観測した時刻を保つ。
    assert observed_elapsed_ms == 120


def test_a_new_version_clears_the_cached_marking() -> None:
    worker = FakeWorker()
    worker.result = (_estimate(), 1)
    relay = DemographicRelay(worker)  # type: ignore[arg-type]
    relay.collect(timestamp_ms=120)

    worker.result = (_estimate(age=40.0), 2)
    collected = relay.collect(timestamp_ms=380)

    assert collected is not None
    estimate, observed_elapsed_ms = collected
    assert estimate.is_cached is False
    assert observed_elapsed_ms == 380


def test_nothing_is_collected_before_the_first_result() -> None:
    worker = FakeWorker()
    relay = DemographicRelay(worker)  # type: ignore[arg-type]

    assert relay.collect(timestamp_ms=10) is None


def test_resetting_starts_a_new_comparison_segment() -> None:
    """§8.8, FR-023: 区間が変わったらキャッシュ済みの推定を破棄する。"""
    worker = FakeWorker()
    worker.result = (_estimate(), 1)
    relay = DemographicRelay(worker)  # type: ignore[arg-type]
    relay.collect(timestamp_ms=120)

    relay.reset()
    collected = relay.collect(timestamp_ms=500)

    assert worker.resets == 1
    assert collected is not None
    assert collected[0].is_cached is False


def test_the_interval_and_close_reach_the_worker() -> None:
    worker = FakeWorker()
    relay = DemographicRelay(worker)  # type: ignore[arg-type]

    relay.set_interval_ms(1500)
    relay.close()

    assert worker.intervals == [1500]
    assert worker.closed is True
