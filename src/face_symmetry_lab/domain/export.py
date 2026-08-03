"""Immutable export DTOs for the versioned session schema (§9.2)."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from face_symmetry_lab.domain.models import GenderLabel, ReasonCode, Status

#: §9.2's record schema version, carried by both the records and the
#: manifest. `Final` narrows it to its literal type, so it can serve as
#: the default for the `Literal` fields below and the version has one
#: definition instead of one per model that must agree on it.
SCHEMA_VERSION: Final = "3.0.0"


class InputMode(StrEnum):
    CAMERA = "camera"
    IMAGE = "image"


class ExportRecord(BaseModel):
    """One row shared by the CSV and JSON representations."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["3.0.0"] = SCHEMA_VERSION
    session_id: str
    segment_id: str
    frame_id: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)
    captured_at: datetime | None = None
    input_mode: InputMode
    analysis_status: Status
    reason_codes: tuple[ReasonCode, ...] = ()
    warning_codes: tuple[ReasonCode, ...] = ()
    face_count: int | None = Field(default=None, ge=0)
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    face_area_ratio: float | None = Field(default=None, ge=0, le=1)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    # §8.6: normalized RMS distances, so 0 is the floor and there is no upper
    # bound to validate against. `distance_overall_raw` is the pooled fold, not
    # a blend of the four parts above.
    distance_contour_raw: float | None = Field(default=None, ge=0)
    distance_eyes_raw: float | None = Field(default=None, ge=0)
    distance_nose_raw: float | None = Field(default=None, ge=0)
    distance_mouth_raw: float | None = Field(default=None, ge=0)
    distance_overall_raw: float | None = Field(default=None, ge=0)
    distance_overall_smooth: float | None = Field(default=None, ge=0)
    skin_quality_score_raw: float | None = Field(default=None, ge=0, le=100)
    skin_tone_uniformity_raw: float | None = Field(default=None, ge=0, le=100)
    skin_texture_uniformity_raw: float | None = Field(default=None, ge=0, le=100)
    skin_correction_gain_r: float | None = Field(default=None, ge=0.5, le=2)
    skin_correction_gain_g: float | None = Field(default=None, ge=0.5, le=2)
    skin_correction_gain_b: float | None = Field(default=None, ge=0.5, le=2)
    skin_metric_version: str | None = None
    skin_quality_frozen: bool | None = None
    estimated_age_years: float | None = Field(default=None, ge=18, le=75)
    gender_label: GenderLabel | None = None
    gender_confidence: float | None = Field(default=None, ge=0, le=1)
    gender_prob_female: float | None = Field(default=None, ge=0, le=1)
    gender_prob_male: float | None = Field(default=None, ge=0, le=1)
    demographic_observed_elapsed_ms: int | None = Field(default=None, ge=0)
    demographic_is_cached: bool | None = None
    demographic_model_version: str | None = None
    demographic_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    landmark_model_version: str = Field(min_length=1)
    landmark_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_version: str = Field(min_length=1)
    app_version: str = Field(min_length=1)
    lock_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_revision: int = Field(ge=1)
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_ms: float = Field(ge=0)
    pipeline_latency_ms: float = Field(ge=0)

    @field_validator("session_id", "segment_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        UUID(value)
        return value

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() != timedelta(0):
            raise ValueError("captured_at must be UTC")
        return value

    @model_validator(mode="after")
    def measurements_belong_only_to_ok_records(self) -> ExportRecord:
        measured_values = (
            self.distance_contour_raw,
            self.distance_eyes_raw,
            self.distance_nose_raw,
            self.distance_mouth_raw,
            self.distance_overall_raw,
            self.distance_overall_smooth,
            self.skin_quality_score_raw,
            self.skin_tone_uniformity_raw,
            self.skin_texture_uniformity_raw,
        )
        if self.analysis_status is not Status.OK and any(
            value is not None for value in measured_values
        ):
            raise ValueError("左右差の距離と肌表面参考値は OK レコードだけに保持できます。")
        return self

    @model_validator(mode="after")
    def demographic_fields_are_complete(self) -> ExportRecord:
        values = (
            self.estimated_age_years,
            self.gender_label,
            self.gender_confidence,
            self.gender_prob_female,
            self.gender_prob_male,
            self.demographic_observed_elapsed_ms,
            self.demographic_is_cached,
            self.demographic_model_version,
            self.demographic_model_sha256,
        )
        if all(value is None for value in values):
            return self
        if any(value is None for value in values):
            raise ValueError("年齢・性別推定fieldは一式で保持する必要があります。")
        assert self.gender_prob_female is not None
        assert self.gender_prob_male is not None
        if abs(self.gender_prob_female + self.gender_prob_male - 1.0) > 1e-6:
            raise ValueError("gender probabilities must sum to 1")
        return self


class ModelIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ConfigRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(ge=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    settings: dict[str, JsonValue]


class ExportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["3.0.0"] = SCHEMA_VERSION
    app_version: str = Field(min_length=1)
    lock_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_id: str
    started_at: datetime | None = None
    models: tuple[ModelIdentity, ...]
    metric_version: str = Field(min_length=1)
    config_revisions: tuple[ConfigRevision, ...]

    @field_validator("session_id")
    @classmethod
    def validate_session_uuid(cls, value: str) -> str:
        UUID(value)
        return value


class SessionExport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: ExportManifest
    records: tuple[ExportRecord, ...]

    @model_validator(mode="after")
    def records_match_manifest(self) -> SessionExport:
        revisions = {(item.revision, item.fingerprint) for item in self.manifest.config_revisions}
        for record in self.records:
            if record.session_id != self.manifest.session_id:
                raise ValueError("record session_id does not match manifest")
            if record.metric_version != self.manifest.metric_version:
                raise ValueError("record metric_version does not match manifest")
            if (record.config_revision, record.config_fingerprint) not in revisions:
                raise ValueError("record config does not exist in manifest")
        return self
