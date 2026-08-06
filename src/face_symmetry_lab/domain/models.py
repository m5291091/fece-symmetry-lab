from datetime import datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class Status(StrEnum):
    OK = "OK"
    HOLD = "HOLD"
    NO_FACE = "NO_FACE"
    MULTI_FACE = "MULTI_FACE"
    ERROR = "ERROR"


class ReasonCode(StrEnum):
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    POSTURE = "POSTURE"
    DISTANCE = "DISTANCE"
    LOW_QUALITY = "LOW_QUALITY"
    # §6.3 のコード。上の 5 つはこの表より前からあり、粗い分類だけを必要と
    # する呼び出し側のために残してある。
    OFF_AXIS_YAW = "OFF_AXIS_YAW"
    OFF_AXIS_PITCH = "OFF_AXIS_PITCH"
    OFF_AXIS_ROLL = "OFF_AXIS_ROLL"
    TOO_CLOSE = "TOO_CLOSE"
    TOO_FAR = "TOO_FAR"
    UNSTABLE = "UNSTABLE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_LIGHT = "LOW_LIGHT"
    OVEREXPOSED = "OVEREXPOSED"
    BLUR = "BLUR"
    STALE_RESULT = "STALE_RESULT"


class GenderLabel(StrEnum):
    """採用した外見モデルが出力する 2 値ラベル。"""

    FEMALE = "female"
    MALE = "male"


class AnalysisState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Status
    reason: ReasonCode | None = None
    #: §6.2 は、阻害理由と阻害しない警告を分けて保つ。
    reason_codes: tuple[ReasonCode, ...] = ()
    warning_codes: tuple[ReasonCode, ...] = ()


class SymmetryDistance(BaseModel):
    """§8.6 の正規化 RMS 距離。単位は両眼間隔である。

    下限は 0 で、上限は無い。これらは 0〜100 のスコアではなく測定値であり、
    `overall` は 4 部位の重み付き混合ではなく、まとめて折りたたんだ値である。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    contour: float = Field(ge=0)
    eyes: float = Field(ge=0)
    nose: float = Field(ge=0)
    mouth: float = Field(ge=0)
    overall: float = Field(ge=0)


class DemographicEstimate(BaseModel):
    """18〜75 歳を対象とする年齢・外見モデルの参考出力。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    estimated_age_years: float = Field(ge=18, le=75)
    gender: GenderLabel
    gender_confidence: float = Field(ge=0, le=1)
    gender_probabilities: tuple[float, float]
    model_version: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: datetime
    is_cached: bool = False

    @field_validator("gender_probabilities")
    @classmethod
    def validate_probabilities(
        cls,
        probabilities: tuple[float, float],
    ) -> tuple[float, float]:
        if any(not 0 <= probability <= 1 for probability in probabilities):
            raise ValueError("probabilities must be between 0 and 1")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to 1")
        return probabilities

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, observed_at: datetime) -> datetime:
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return observed_at

    @model_validator(mode="after")
    def confidence_matches_distribution(self) -> "DemographicEstimate":
        winning_index = max(
            range(len(self.gender_probabilities)),
            key=self.gender_probabilities.__getitem__,
        )
        if abs(self.gender_confidence - self.gender_probabilities[winning_index]) > 1e-6:
            raise ValueError("confidence must equal the largest probability")
        labels = (GenderLabel.FEMALE, GenderLabel.MALE)
        if self.gender is not labels[winning_index]:
            raise ValueError("gender must match the largest probability")
        return self


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: AnalysisState
    distance: SymmetryDistance | None = None
    demographic_estimate: DemographicEstimate | None = None
