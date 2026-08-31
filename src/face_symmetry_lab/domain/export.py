"""版付きセッションスキーマのための、不変なエクスポート DTO（§9.2）。"""

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

#: §9.2 の record schema の版。records と manifest の両方が持つ。`Final` が
#: リテラル型へ絞り込むので、下の `Literal` フィールドの既定値として使え、版の
#: 定義が 1 箇所に収まる。モデルごとに定義して整合を取る必要が無くなる。
#: v4.0.0 は主指標を `symmetry-3d/1.0.0` へ切り替えた破壊的変更（§8.6.3, §8.6.5,
#: §9.2）。`distance3d_*` 列を追加し、2D の EMA 列 `distance_overall_smooth` を
#: 削除した。
#: v5.0.0 は肌表面参考値列（`skin_*` 8 列）を削除した（FR-025 / §8.6.1 撤回、
#: 2026-08-31）。
SCHEMA_VERSION: Final = "5.0.0"


class AgeGateDecisionLabel(StrEnum):
    """§9.2 の `age_gate_decision` 列。

    `application.age_gate.AgeGateStatus` の 4 値のうち、記録へ残すのは「呼んだ
    結果」の 3 つだけである。`NOT_EVALUATED`（顔なし・姿勢ゲート閉で呼ばなかった）
    は列としては `None` になる——呼んでいない判定に名前を与えると、行を読む側が
    「ゲートが何かを決めた」と受け取る。
    """

    IN_RANGE = "in_range"
    OUT_OF_RANGE = "out_of_range"
    UNDETERMINED = "undetermined"


class InputMode(StrEnum):
    CAMERA = "camera"
    IMAGE = "image"


class ExportRecord(BaseModel):
    """CSV 表現と JSON 表現が共有する 1 行。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["5.0.0"] = SCHEMA_VERSION
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
    # §8.6: 正規化 RMS 距離なので下限は 0 であり、検証すべき上限は存在しない。
    # `distance_overall_raw` はまとめて折りたたんだ値であって、上の 4 部位を
    # 混合したものではない。`symmetry-2d/2.0.0` は画面から外れており、これらは
    # 版の比較可能性のために残す診断列である（§8.6.6, §9.2）。
    distance_contour_raw: float | None = Field(default=None, ge=0)
    distance_eyes_raw: float | None = Field(default=None, ge=0)
    distance_nose_raw: float | None = Field(default=None, ge=0)
    distance_mouth_raw: float | None = Field(default=None, ge=0)
    distance_overall_raw: float | None = Field(default=None, ge=0)
    # §8.6.5/§9.2: `symmetry-3d/1.0.0` の出力一式（主指標）。部位別・全体は並列の
    # 診断関係にあり、`distance3d_overall_raw` は部位別からの合成値ではない。
    # `distance3d_parts_equal_raw` は `s_3d`（部位を 1 票ずつ数える）であり、
    # `distance3d_overall_raw`（`d_3d`、対応点を 1 票ずつ数える）とは重みだけが
    # 異なる別の総合量である。
    distance3d_contour_raw: float | None = Field(default=None, ge=0)
    distance3d_eyes_raw: float | None = Field(default=None, ge=0)
    distance3d_nose_raw: float | None = Field(default=None, ge=0)
    distance3d_mouth_raw: float | None = Field(default=None, ge=0)
    distance3d_overall_raw: float | None = Field(default=None, ge=0)
    distance3d_overall_smooth: float | None = Field(default=None, ge=0)
    distance3d_parts_equal_raw: float | None = Field(default=None, ge=0)
    estimated_age_years: float | None = Field(default=None, ge=18, le=75)
    gender_label: GenderLabel | None = None
    gender_confidence: float | None = Field(default=None, ge=0, le=1)
    gender_prob_female: float | None = Field(default=None, ge=0, le=1)
    gender_prob_male: float | None = Field(default=None, ge=0, le=1)
    demographic_observed_elapsed_ms: int | None = Field(default=None, ge=0)
    demographic_is_cached: bool | None = None
    #: §9.2/FR-038: ゲートの判定と、その動作点の版。**生の確率は持たない**
    #: （記録すると凍結閾値以外での再判定・順位付け・「どれだけ若く見えるか」と
    #: いう読み替えを可能にしてしまう）。判定・版・凍結閾値の 3 つから当時の
    #: 挙動は一意に辿れる。
    #:
    #: **この列が無いと、`estimated_age_years` が `null` のとき「ゲートで弾かれた」
    #: のか「モデル未配置」なのかを行から区別できない。** ゲートを呼ばなかった
    #: フレーム（顔なし・姿勢ゲート閉）は `None` である。
    age_gate_decision: AgeGateDecisionLabel | None = None
    age_gate_version: str | None = None
    age_gate_spec_revision: int | None = Field(default=None, ge=1)
    age_gate_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    demographic_model_version: str | None = None
    demographic_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    landmark_model_version: str = Field(min_length=1)
    landmark_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: 主指標（`symmetry-3d/1.0.0`）の版（§8.6.5）。
    metric_version: str = Field(min_length=1)
    #: `distance_*_raw`（診断列）を生成した `symmetry-2d/2.0.0` 系 bundle の版。
    #: 行ごとに持つ理由は §9.2「2D の版の追跡」を参照。
    metric_version_2d: str = Field(min_length=1)
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
            self.distance3d_contour_raw,
            self.distance3d_eyes_raw,
            self.distance3d_nose_raw,
            self.distance3d_mouth_raw,
            self.distance3d_overall_raw,
            self.distance3d_overall_smooth,
            self.distance3d_parts_equal_raw,
        )
        if self.analysis_status is not Status.OK and any(
            value is not None for value in measured_values
        ):
            raise ValueError("左右差の距離は OK レコードだけに保持できます。")
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

    schema_version: Literal["5.0.0"] = SCHEMA_VERSION
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
