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
    # §6.3 codes. The five above predate this table and are kept for callers
    # that only need the coarse category.
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


class AgeBand(StrEnum):
    """Age-group labels used by the seminar MobileNetV2 model."""

    AGE_0_9 = "0-9"
    AGE_10_19 = "10-19"
    AGE_20_29 = "20-29"
    AGE_30_39 = "30-39"
    AGE_40_49 = "40-49"
    AGE_50_59 = "50-59"
    AGE_60_69 = "60-69"
    AGE_70_PLUS = "70+"


AGE_BANDS: tuple[AgeBand, ...] = tuple(AgeBand)


class AnalysisState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Status
    reason: ReasonCode | None = None
    #: §6.2 keeps blocking reasons and non-blocking warnings apart.
    reason_codes: tuple[ReasonCode, ...] = ()
    warning_codes: tuple[ReasonCode, ...] = ()


class SymmetryScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contour: float = Field(ge=0, le=100)
    eyes: float = Field(ge=0, le=100)
    nose: float = Field(ge=0, le=100)
    mouth: float = Field(ge=0, le=100)
    overall: float = Field(ge=0, le=100)


class AgeEstimate(BaseModel):
    """A model output, expressed as a reference range rather than a claimed age."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band: AgeBand
    confidence: float = Field(ge=0, le=1)
    probabilities: tuple[float, ...] = Field(
        min_length=len(AGE_BANDS),
        max_length=len(AGE_BANDS),
    )
    model_version: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: datetime
    is_cached: bool = False

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(
        cls,
        probabilities: tuple[float, ...],
    ) -> tuple[float, ...]:
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
    def confidence_matches_distribution(self) -> "AgeEstimate":
        winning_index = max(
            range(len(self.probabilities)),
            key=self.probabilities.__getitem__,
        )
        if abs(self.confidence - self.probabilities[winning_index]) > 1e-6:
            raise ValueError("confidence must equal the largest probability")
        if self.band is not AGE_BANDS[winning_index]:
            raise ValueError("band must match the largest probability")
        return self


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: AnalysisState
    score: SymmetryScore | None = None
    age_estimate: AgeEstimate | None = None
