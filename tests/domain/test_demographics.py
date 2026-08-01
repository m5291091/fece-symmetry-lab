from datetime import UTC, datetime
from math import nan

import pytest
from pydantic import ValidationError

from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel


def test_build_demographic_estimate_keeps_continuous_age_and_gender_probabilities() -> None:
    estimate = build_demographic_estimate(
        estimated_age_years=36.5,
        gender_probabilities=(0.2, 0.8),
        model_version="openvino-age-gender/1.0.0",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert estimate.estimated_age_years == pytest.approx(36.5)
    assert estimate.gender is GenderLabel.MALE
    assert estimate.gender_confidence == pytest.approx(0.8)
    assert estimate.gender_probabilities == pytest.approx((0.2, 0.8))


@pytest.mark.parametrize("age", [17.99, 75.01, nan])
def test_demographic_estimate_rejects_age_outside_model_scope(age: float) -> None:
    with pytest.raises((ValueError, ValidationError)):
        build_demographic_estimate(
            estimated_age_years=age,
            gender_probabilities=(0.5, 0.5),
            model_version="openvino-age-gender/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        )


def test_demographic_estimate_requires_two_finite_probabilities() -> None:
    with pytest.raises((ValueError, ValidationError)):
        build_demographic_estimate(
            estimated_age_years=40,
            gender_probabilities=(1.0,),
            model_version="openvino-age-gender/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        )


def test_demographic_estimate_requires_timezone_aware_observation() -> None:
    with pytest.raises(ValidationError):
        DemographicEstimate(
            estimated_age_years=40,
            gender=GenderLabel.FEMALE,
            gender_confidence=0.7,
            gender_probabilities=(0.7, 0.3),
            model_version="openvino-age-gender/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 8, 1),
        )
