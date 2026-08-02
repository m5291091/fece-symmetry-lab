import threading
from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.application.demographic_worker import DemographicWorker
from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.models import DemographicEstimate


def _estimate(age: float = 30.0) -> DemographicEstimate:
    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(0.4, 0.6),
        model_version="demographic/test",
        model_sha256="a" * 64,
        observed_at=datetime.now(UTC),
    )


def _crop() -> NDArray[np.uint8]:
    return np.zeros((62, 62, 3), dtype=np.uint8)


class ScriptedEstimator:
    """Returns scripted results/errors in call order; records received crops."""

    def __init__(self, results: list[DemographicEstimate | Exception]) -> None:
        self._results = list(results)
        self.calls: list[NDArray[np.uint8]] = []

    def estimate(self, rgb_image: NDArray[np.uint8]) -> DemographicEstimate:
        self.calls.append(rgb_image)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class BlockingEstimator:
    """Blocks inside estimate() until released, to test capacity-1 gating."""

    def __init__(self, result: DemographicEstimate) -> None:
        self._result = result
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def estimate(self, rgb_image: NDArray[np.uint8]) -> DemographicEstimate:
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return self._result


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_submit_runs_the_estimator_in_the_background_and_publishes_the_result() -> None:
    estimate = _estimate()
    estimator = ScriptedEstimator([estimate])
    worker = DemographicWorker(estimator=estimator, clock=ManualClock())

    thread = worker.submit(_crop())
    assert thread is not None
    thread.join(timeout=2)

    result = worker.latest()
    assert result is not None
    published, _version = result
    assert published == estimate
    assert len(estimator.calls) == 1


def test_a_second_submission_while_busy_is_dropped() -> None:
    estimator = BlockingEstimator(_estimate())
    worker = DemographicWorker(estimator=estimator, clock=ManualClock())

    first_thread = worker.submit(_crop())
    assert first_thread is not None
    estimator.started.wait(timeout=2)

    second_thread = worker.submit(_crop())
    assert second_thread is None

    estimator.release.set()
    first_thread.join(timeout=2)
    assert estimator.calls == 1


def test_submission_before_the_interval_elapses_is_dropped() -> None:
    clock = ManualClock()
    estimator = ScriptedEstimator([_estimate(), _estimate(age=40.0)])
    worker = DemographicWorker(estimator=estimator, interval_ms=1000, clock=clock)

    first = worker.submit(_crop())
    assert first is not None
    first.join(timeout=2)

    clock.advance(0.5)
    too_soon = worker.submit(_crop())
    assert too_soon is None

    clock.advance(0.6)
    second = worker.submit(_crop())
    assert second is not None
    second.join(timeout=2)

    assert len(estimator.calls) == 2


def test_a_result_computed_after_reset_is_discarded() -> None:
    estimator = BlockingEstimator(_estimate())
    worker = DemographicWorker(estimator=estimator, clock=ManualClock())

    thread = worker.submit(_crop())
    assert thread is not None
    estimator.started.wait(timeout=2)

    worker.reset()
    estimator.release.set()
    thread.join(timeout=2)

    assert worker.latest() is None


def test_estimator_failures_keep_the_previous_result() -> None:
    good = _estimate()
    estimator = ScriptedEstimator([good, RuntimeError("boom")])
    clock = ManualClock()
    worker = DemographicWorker(estimator=estimator, interval_ms=0, clock=clock)

    first = worker.submit(_crop())
    assert first is not None
    first.join(timeout=2)
    first_result = worker.latest()
    assert first_result is not None

    second = worker.submit(_crop())
    assert second is not None
    second.join(timeout=2)

    assert worker.latest() == first_result


def test_submissions_are_refused_after_close() -> None:
    estimator = ScriptedEstimator([_estimate()])
    worker = DemographicWorker(estimator=estimator, clock=ManualClock())

    worker.close()
    thread = worker.submit(_crop())

    assert thread is None
    assert estimator.calls == []


def test_a_result_completing_after_close_is_discarded() -> None:
    estimator = BlockingEstimator(_estimate())
    worker = DemographicWorker(estimator=estimator, clock=ManualClock())

    thread = worker.submit(_crop())
    assert thread is not None
    estimator.started.wait(timeout=2)

    worker.close()
    estimator.release.set()
    thread.join(timeout=2)

    assert worker.latest() is None


def test_negative_interval_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        DemographicWorker(estimator=ScriptedEstimator([]), interval_ms=-1)


def test_is_due_tracks_the_cadence_without_consuming_it() -> None:
    clock = ManualClock()
    estimator = ScriptedEstimator([_estimate(), _estimate()])
    worker = DemographicWorker(estimator=estimator, interval_ms=1000, clock=clock)

    assert worker.is_due() is True
    assert worker.is_due() is True  # advisory only: asking never claims the slot

    thread = worker.submit(_crop())
    assert thread is not None
    thread.join(timeout=2)

    assert worker.is_due() is False
    clock.advance(1.1)
    assert worker.is_due() is True


def test_is_due_is_false_while_an_inference_is_running() -> None:
    estimator = BlockingEstimator(_estimate())
    worker = DemographicWorker(estimator=estimator, interval_ms=0, clock=ManualClock())

    thread = worker.submit(_crop())
    assert thread is not None
    estimator.started.wait(timeout=2)

    assert worker.is_due() is False

    estimator.release.set()
    thread.join(timeout=2)


def test_is_due_is_false_after_close() -> None:
    worker = DemographicWorker(estimator=ScriptedEstimator([]), clock=ManualClock())

    worker.close()

    assert worker.is_due() is False


def test_set_interval_ms_changes_the_cadence_for_the_next_submission() -> None:
    """§15.2 `age_interval_ms`: editable independently of the analysis cadence."""
    clock = ManualClock()
    estimator = ScriptedEstimator([_estimate()])
    worker = DemographicWorker(estimator=estimator, interval_ms=1000, clock=clock)
    thread = worker.submit(_crop())
    assert thread is not None
    thread.join(timeout=2)
    assert worker.is_due() is False  # next slot is 1000ms out

    worker.set_interval_ms(0)

    assert worker.is_due() is True


def test_set_interval_ms_rejects_a_negative_value() -> None:
    worker = DemographicWorker(estimator=ScriptedEstimator([]), clock=ManualClock())

    with pytest.raises(ValueError):
        worker.set_interval_ms(-1)
