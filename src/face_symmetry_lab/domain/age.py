from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from math import exp, isfinite

from face_symmetry_lab.domain.models import AGE_BANDS, AgeEstimate


def build_age_estimate(
    *,
    logits: Sequence[float],
    model_version: str,
    model_sha256: str,
    observed_at: datetime,
) -> AgeEstimate:
    """Convert eight model logits into a validated, versioned age-band result."""
    numeric_logits = tuple(float(value) for value in logits)
    if len(numeric_logits) != len(AGE_BANDS):
        raise ValueError(f"expected {len(AGE_BANDS)} age logits")
    if not all(isfinite(value) for value in numeric_logits):
        raise ValueError("age logits must be finite")

    maximum = max(numeric_logits)
    exponentials = tuple(exp(value - maximum) for value in numeric_logits)
    total = sum(exponentials)
    probabilities = tuple(value / total for value in exponentials)
    winning_index = max(range(len(probabilities)), key=probabilities.__getitem__)

    return AgeEstimate(
        band=AGE_BANDS[winning_index],
        confidence=probabilities[winning_index],
        probabilities=probabilities,
        model_version=model_version,
        model_sha256=model_sha256,
        observed_at=observed_at,
    )
