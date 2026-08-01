"""Immutable export DTOs for the versioned session schema (§9.2)."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from face_symmetry_lab.domain.models import ReasonCode, Status

SCHEMA_VERSION = "1.1.0"


class InputMode(StrEnum):
    CAMERA = "camera"
    IMAGE = "image"


class ExportRecord(BaseModel):
    """One row shared by the CSV and JSON representations."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.1.0"] = "1.1.0"
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
    score_contour_raw: float | None = Field(default=None, ge=0, le=100)
    score_eyes_raw: float | None = Field(default=None, ge=0, le=100)
    score_nose_raw: float | None = Field(default=None, ge=0, le=100)
    score_mouth_raw: float | None = Field(default=None, ge=0, le=100)
    score_overall_raw: float | None = Field(default=None, ge=0, le=100)
    score_overall_smooth: float | None = Field(default=None, ge=0, le=100)
    distance_contour_raw: float | None = Field(default=None, ge=0)
    distance_eyes_raw: float | None = Field(default=None, ge=0)
    distance_nose_raw: float | None = Field(default=None, ge=0)
    distance_mouth_raw: float | None = Field(default=None, ge=0)
    age_band: str | None = None
    age_confidence: float | None = Field(default=None, ge=0, le=1)
    age_prob_0_9: float | None = Field(default=None, ge=0, le=1)
    age_prob_10_19: float | None = Field(default=None, ge=0, le=1)
    age_prob_20_29: float | None = Field(default=None, ge=0, le=1)
    age_prob_30_39: float | None = Field(default=None, ge=0, le=1)
    age_prob_40_49: float | None = Field(default=None, ge=0, le=1)
    age_prob_50_59: float | None = Field(default=None, ge=0, le=1)
    age_prob_60_69: float | None = Field(default=None, ge=0, le=1)
    age_prob_70_plus: float | None = Field(default=None, ge=0, le=1)
    age_observed_elapsed_ms: int | None = Field(default=None, ge=0)
    age_is_cached: bool | None = None
    age_model_version: str | None = None
    age_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
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
    def scores_belong_only_to_ok_records(self) -> ExportRecord:
        score_values = (
            self.score_contour_raw,
            self.score_eyes_raw,
            self.score_nose_raw,
            self.score_mouth_raw,
            self.score_overall_raw,
            self.score_overall_smooth,
            self.distance_contour_raw,
            self.distance_eyes_raw,
            self.distance_nose_raw,
            self.distance_mouth_raw,
        )
        if self.analysis_status is not Status.OK and any(
            value is not None for value in score_values
        ):
            raise ValueError("左右差スコアと距離は OK レコードだけに保持できます。")
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

    schema_version: Literal["1.1.0"] = "1.1.0"
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
