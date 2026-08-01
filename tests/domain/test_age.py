from datetime import UTC, datetime
from math import nan

import pytest
from pydantic import ValidationError

from face_symmetry_lab.domain.age import build_age_estimate
from face_symmetry_lab.domain.models import AGE_BANDS, AgeBand, AgeEstimate


def test_build_age_estimate_selects_highest_probability_band() -> None:
    estimate = build_age_estimate(
        logits=(0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        model_version="age-mobilenetv2-utkface/1.0.0",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 7, 30, tzinfo=UTC),
    )

    assert estimate.band is AgeBand.AGE_20_29
    assert estimate.confidence == pytest.approx(max(estimate.probabilities))
    assert sum(estimate.probabilities) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "logits",
    [
        (0.0,) * 7,
        (0.0,) * 7 + (nan,),
    ],
)
def test_build_age_estimate_rejects_invalid_logits(logits: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        build_age_estimate(
            logits=logits,
            model_version="age-mobilenetv2-utkface/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 7, 30, tzinfo=UTC),
        )


def test_age_estimate_requires_complete_probability_distribution() -> None:
    with pytest.raises(ValidationError):
        AgeEstimate(
            band=AgeBand.AGE_20_29,
            confidence=0.8,
            probabilities=(0.1,) * (len(AGE_BANDS) - 1),
            model_version="age-mobilenetv2-utkface/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 7, 30, tzinfo=UTC),
        )


def test_age_estimate_requires_timezone_aware_observation_time() -> None:
    with pytest.raises(ValidationError):
        AgeEstimate(
            band=AgeBand.AGE_20_29,
            confidence=0.8,
            probabilities=(0.025, 0.025, 0.8, 0.025, 0.025, 0.025, 0.025, 0.05),
            model_version="age-mobilenetv2-utkface/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 7, 30),
        )


def test_age_estimate_requires_band_to_match_highest_probability() -> None:
    with pytest.raises(ValidationError):
        AgeEstimate(
            band=AgeBand.AGE_30_39,
            confidence=0.8,
            probabilities=(0.025, 0.025, 0.8, 0.025, 0.025, 0.025, 0.025, 0.05),
            model_version="age-mobilenetv2-utkface/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 7, 30, tzinfo=UTC),
        )
