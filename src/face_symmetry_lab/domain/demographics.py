"""18〜75 歳の年齢・性別モデルに対する、検証付きの domain 変換。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from math import isfinite

from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel


def build_demographic_estimate(
    *,
    estimated_age_years: float,
    gender_probabilities: tuple[float, ...],
    model_version: str,
    model_sha256: str,
    observed_at: datetime,
) -> DemographicEstimate:
    """モデルの生出力を検証する。対象外の年齢を切り詰めることはしない。"""
    age = float(estimated_age_years)
    probabilities = tuple(float(value) for value in gender_probabilities)
    if not isfinite(age) or not 18 <= age <= 75:
        raise ValueError("estimated age is outside the supported 18-75 range")
    if len(probabilities) != 2 or any(not isfinite(value) for value in probabilities):
        raise ValueError("expected two finite gender probabilities")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("gender probabilities must have a positive sum")
    normalized = (probabilities[0] / total, probabilities[1] / total)
    winning_index = 0 if normalized[0] >= normalized[1] else 1
    labels = (GenderLabel.FEMALE, GenderLabel.MALE)
    return DemographicEstimate(
        estimated_age_years=age,
        gender=labels[winning_index],
        gender_confidence=normalized[winning_index],
        gender_probabilities=normalized,
        model_version=model_version,
        model_sha256=model_sha256,
        observed_at=observed_at,
    )


def mean_gender_probabilities(
    probabilities: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """観測された (女性, 男性) 確率の要素ごと平均。空なら None。

    各 estimate の `gender_probabilities` は正規化済み（和 1）なので、平均も和 1 に近い。
    """
    if not probabilities:
        return None
    n = len(probabilities)
    female = sum(p[0] for p in probabilities) / n
    male = sum(p[1] for p in probabilities) / n
    return (female, male)
