"""セッション単位の数値履歴と、再現性のためのメタデータ（§9, §14.2）。"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import cast
from uuid import uuid4

from pydantic import JsonValue

from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateService, AgeGateStatus
from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.domain.export import (
    AgeGateDecisionLabel,
    ConfigRevision,
    ExportManifest,
    ExportRecord,
    InputMode,
    ModelIdentity,
    SessionExport,
)
from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.models import DemographicEstimate, ReasonCode, Status

DEFAULT_HISTORY_CAPACITY = 3_000


def build_config_settings(
    config: MetricConfig,
    thresholds: QualityThresholds,
    *,
    tau_seconds: float,
    analysis_interval_ms: int | None = None,
) -> dict[str, JsonValue]:
    """比較可能性を定義する digest の元になる、正規化済み設定を返す。

    ``config`` は**主指標**の bundle（`symmetry-3d/1.0.0`）を渡す（§8.6.5, §9.2）。
    2D と 3D は同じ `landmark_map_version` と対応ペア表を共有するため
    `part_pair_counts` / `overall_min_pairs` は bundle を差し替えても値は変わらないが、
    `metric_version` が変わるぶん fingerprint は変わる。これは版の切り替えとして
    正しい振る舞いである。2D 側の版（`metric_version_2d`）は比較可能性の定義には
    使わず、`SessionRecorder` が行ごとの列として別に追跡する。
    """
    settings: dict[str, JsonValue] = {
        "metric_version": config.metric_version,
        "landmark_map_version": config.landmark_map_version,
        # §8.6 は bundle に係数を残さなかった。そのため、まとめた総合距離の
        # 組み立て方を決めるペア数こそが、2 つの区間を比較可能にする要素である。
        "part_pair_counts": cast(
            dict[str, JsonValue],
            {name: len(part.pairs) for name, part in config.parts.items()},
        ),
        "overall_min_pairs": config.overall_min_pairs,
        "yaw_inner_deg": thresholds.yaw_inner_deg,
        "yaw_outer_deg": thresholds.yaw_outer_deg,
        "pitch_inner_deg": thresholds.pitch_inner_deg,
        "pitch_outer_deg": thresholds.pitch_outer_deg,
        "roll_inner_deg": thresholds.roll_inner_deg,
        "roll_outer_deg": thresholds.roll_outer_deg,
        "face_area_min": thresholds.face_area_min,
        "face_area_max": thresholds.face_area_max,
        "valid_frames_to_ok": thresholds.valid_frames_to_ok,
        "ema_tau_seconds": tau_seconds,
    }
    if analysis_interval_ms is not None:
        settings["analysis_interval_ms"] = analysis_interval_ms
    return settings


def config_fingerprint(settings: Mapping[str, object]) -> str:
    """正準形 JSON をハッシュ化し、キー順や書式が同一性を変えないようにする。"""
    canonical = json.dumps(
        dict(settings),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class AgeGateProvenance:
    """§9.2/FR-038: 1 フレーム分のゲート判定と、その動作点の版。

    **生の確率は持たない。** 記録すると凍結閾値以外での再判定・順位付け・
    「どれだけ若く見えるか」という読み替えを可能にしてしまう（§9.2）。判定・版・
    凍結閾値の 3 つから当時の挙動は一意に辿れる。

    ゲートを呼ばなかったフレーム（顔なし・姿勢ゲート閉）ではこの DTO 自体を作らず、
    記録側は 4 列とも `None` になる——呼んでいない判定に名前を与えると、行を読む側が
    「ゲートが何かを決めた」と受け取る。
    """

    decision: AgeGateDecisionLabel
    version: str
    spec_revision: int | None
    model_sha256: str | None


#: §9.2: `AgeGateStatus` から記録用のラベルへの対応。**`NOT_EVALUATED` を持たない**
#: ——呼ばなかったフレームは列ごと `None` になる。
_AGE_GATE_LABELS: dict[AgeGateStatus, AgeGateDecisionLabel] = {
    AgeGateStatus.IN_RANGE: AgeGateDecisionLabel.IN_RANGE,
    AgeGateStatus.OUT_OF_RANGE: AgeGateDecisionLabel.OUT_OF_RANGE,
    AgeGateStatus.UNDETERMINED: AgeGateDecisionLabel.UNDETERMINED,
}


def age_gate_provenance(
    outcome: AgeGateOutcome | None,
    *,
    service: AgeGateService | None,
) -> AgeGateProvenance | None:
    """§9.2/FR-038: ゲートを**呼んだ**フレームだけ、判定と版を記録用に作る。

    `NOT_EVALUATED`（顔なし・姿勢ゲート閉で呼ばなかった）とゲート未接続は `None`
    を返す——呼んでいない判定に名前を与えると、行を読む側が「ゲートが何かを決めた」
    と受け取る。生の確率は持ち出さない（§9.2）。
    """
    if outcome is None or service is None:
        return None
    label = _AGE_GATE_LABELS.get(outcome.status)
    if label is None:
        return None
    return AgeGateProvenance(
        decision=label,
        version=service.spec_version,
        spec_revision=service.spec_revision,
        model_sha256=service.model_sha256,
    )


class SessionRecorder:
    """セッションのコールバックと UI が共有する、スレッドセーフな有界履歴。

    ``metric_config`` は `symmetry-2d/2.0.0`（診断列 `distance_*_raw` の由来）、
    ``metric_config_3d`` は `symmetry-3d/1.0.0`（主指標。`distance3d_*` 列と
    `config_fingerprint` の由来、§9.2）を渡す。`config_3d` を省略可能にすると
    渡し忘れが静かに主指標なしへ落ちてしまうため、`FrameAnalyzer` と同じく
    必須引数にしてある。
    """

    def __init__(
        self,
        *,
        metric_config: MetricConfig,
        metric_config_3d: MetricConfig,
        thresholds: QualityThresholds,
        tau_seconds: float,
        landmark_model_version: str,
        landmark_model_sha256: str,
        app_version: str,
        lock_digest: str,
        analysis_interval_ms: int | None = None,
        history_capacity: int = DEFAULT_HISTORY_CAPACITY,
    ) -> None:
        if history_capacity < 1:
            raise ValueError("history_capacity は 1 以上である必要があります。")
        self._metric_config = metric_config
        self._metric_config_3d = metric_config_3d
        self._thresholds = thresholds
        self._tau_seconds = tau_seconds
        self._analysis_interval_ms = analysis_interval_ms
        self._settings = build_config_settings(
            metric_config_3d,
            thresholds,
            tau_seconds=tau_seconds,
            analysis_interval_ms=analysis_interval_ms,
        )
        self._fingerprint = config_fingerprint(self._settings)
        self._landmark_model_version = landmark_model_version
        self._landmark_model_sha256 = landmark_model_sha256
        self._app_version = app_version
        self._lock_digest = lock_digest
        self._session_id = str(uuid4())
        self._segment_id = str(uuid4())
        self._config_revision = 1
        self._records: deque[ExportRecord] = deque(maxlen=history_capacity)
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def segment_id(self) -> str:
        with self._lock:
            return self._segment_id

    @property
    def config_revision(self) -> int:
        with self._lock:
            return self._config_revision

    def append_analysis(
        self,
        frame: LandmarkFrame,
        outcome: AnalysisOutcome,
        *,
        elapsed_ms: int,
        processing_ms: float,
        pipeline_latency_ms: float | None = None,
        input_mode: InputMode = InputMode.CAMERA,
        captured_at: datetime | None = None,
        age_gate: AgeGateProvenance | None = None,
    ) -> ExportRecord:
        symmetry = outcome.symmetry
        symmetry_3d = outcome.symmetry_3d
        pose = outcome.pose
        with self._lock:
            record = ExportRecord(
                session_id=self._session_id,
                segment_id=self._segment_id,
                frame_id=frame.frame_id,
                elapsed_ms=elapsed_ms,
                captured_at=captured_at,
                input_mode=input_mode,
                analysis_status=outcome.state.status,
                reason_codes=outcome.state.reason_codes,
                warning_codes=outcome.state.warning_codes,
                face_count=frame.face_count,
                yaw_deg=pose.yaw_deg if pose is not None else None,
                pitch_deg=pose.pitch_deg if pose is not None else None,
                roll_deg=pose.roll_deg if pose is not None else None,
                face_area_ratio=outcome.face_area_ratio,
                # §9.2/FR-038: 判定と版だけを残す（生の確率は残さない）。ゲートを
                # 呼ばなかったフレームは 4 列とも `None` になる。
                age_gate_decision=age_gate.decision if age_gate is not None else None,
                age_gate_version=age_gate.version if age_gate is not None else None,
                age_gate_spec_revision=age_gate.spec_revision if age_gate is not None else None,
                age_gate_model_sha256=age_gate.model_sha256 if age_gate is not None else None,
                # symmetry-2d/2.0.0: 画面から外れた診断列（§8.6.6, §9.2）。
                distance_contour_raw=symmetry.distance("contour") if symmetry is not None else None,
                distance_eyes_raw=symmetry.distance("eyes") if symmetry is not None else None,
                distance_nose_raw=symmetry.distance("nose") if symmetry is not None else None,
                distance_mouth_raw=symmetry.distance("mouth") if symmetry is not None else None,
                distance_overall_raw=symmetry.overall if symmetry is not None else None,
                # symmetry-3d/1.0.0: 主指標（§8.6.5, §9.2）。
                distance3d_contour_raw=(
                    symmetry_3d.distance("contour") if symmetry_3d is not None else None
                ),
                distance3d_eyes_raw=(
                    symmetry_3d.distance("eyes") if symmetry_3d is not None else None
                ),
                distance3d_nose_raw=(
                    symmetry_3d.distance("nose") if symmetry_3d is not None else None
                ),
                distance3d_mouth_raw=(
                    symmetry_3d.distance("mouth") if symmetry_3d is not None else None
                ),
                distance3d_overall_raw=(
                    symmetry_3d.overall.distance
                    if symmetry_3d is not None and symmetry_3d.overall is not None
                    else None
                ),
                distance3d_overall_smooth=outcome.overall_distance_3d_smooth,
                distance3d_parts_equal_raw=(
                    symmetry_3d.equal_weight_distance if symmetry_3d is not None else None
                ),
                landmark_model_version=self._landmark_model_version,
                landmark_model_sha256=self._landmark_model_sha256,
                metric_version=self._metric_config_3d.metric_version,
                metric_version_2d=self._metric_config.metric_version,
                app_version=self._app_version,
                lock_digest=self._lock_digest,
                config_revision=self._config_revision,
                config_fingerprint=self._fingerprint,
                processing_ms=processing_ms,
                pipeline_latency_ms=(
                    processing_ms if pipeline_latency_ms is None else pipeline_latency_ms
                ),
            )
            self._records.append(record)
        return record

    def append_error(
        self,
        *,
        frame_id: int,
        elapsed_ms: int,
        processing_ms: float,
        pipeline_latency_ms: float | None = None,
        input_mode: InputMode = InputMode.CAMERA,
    ) -> ExportRecord:
        with self._lock:
            record = ExportRecord(
                session_id=self._session_id,
                segment_id=self._segment_id,
                frame_id=frame_id,
                elapsed_ms=elapsed_ms,
                input_mode=input_mode,
                analysis_status=Status.ERROR,
                reason_codes=(ReasonCode.LOW_QUALITY,),
                landmark_model_version=self._landmark_model_version,
                landmark_model_sha256=self._landmark_model_sha256,
                metric_version=self._metric_config_3d.metric_version,
                metric_version_2d=self._metric_config.metric_version,
                app_version=self._app_version,
                lock_digest=self._lock_digest,
                config_revision=self._config_revision,
                config_fingerprint=self._fingerprint,
                processing_ms=processing_ms,
                pipeline_latency_ms=(
                    processing_ms if pipeline_latency_ms is None else pipeline_latency_ms
                ),
            )
            self._records.append(record)
        return record

    def records(self) -> tuple[ExportRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def attach_demographic_estimate(
        self,
        estimate: DemographicEstimate,
        *,
        observed_elapsed_ms: int,
    ) -> ExportRecord | None:
        """独立に生成された年齢・性別の結果を、最新の記録へ添付する。"""
        with self._lock:
            if not self._records:
                return None
            current = self._records[-1]
            probabilities = estimate.gender_probabilities
            payload = current.model_dump(mode="python")
            payload.update(
                {
                    "estimated_age_years": estimate.estimated_age_years,
                    "gender_label": estimate.gender,
                    "gender_confidence": estimate.gender_confidence,
                    "gender_prob_female": probabilities[0],
                    "gender_prob_male": probabilities[1],
                    "demographic_observed_elapsed_ms": observed_elapsed_ms,
                    "demographic_is_cached": estimate.is_cached,
                    "demographic_model_version": estimate.model_version,
                    "demographic_model_sha256": estimate.model_sha256,
                }
            )
            enriched = ExportRecord.model_validate(payload)
            self._records[-1] = enriched
            return enriched

    def export_bundle(self) -> SessionExport:
        with self._lock:
            records = tuple(self._records)
            revision = self._config_revision
            segment_settings = dict(self._settings)
            demographic_models = tuple(
                dict.fromkeys(
                    (record.demographic_model_version, record.demographic_model_sha256)
                    for record in records
                    if record.demographic_model_version is not None
                    and record.demographic_model_sha256 is not None
                )
            )
        models = [
            ModelIdentity(
                name="face_landmarker",
                version=self._landmark_model_version,
                sha256=self._landmark_model_sha256,
            )
        ]
        models.extend(
            ModelIdentity(name="age_gender", version=version, sha256=model_hash)
            for version, model_hash in demographic_models
        )
        manifest = ExportManifest(
            app_version=self._app_version,
            lock_digest=self._lock_digest,
            session_id=self._session_id,
            models=tuple(models),
            metric_version=self._metric_config_3d.metric_version,
            config_revisions=(
                ConfigRevision(
                    revision=revision,
                    fingerprint=self._fingerprint,
                    settings=segment_settings,
                ),
            ),
        )
        return SessionExport(manifest=manifest, records=records)

    def reset(
        self,
        *,
        thresholds: QualityThresholds | None = None,
        tau_seconds: float | None = None,
        analysis_interval_ms: int | None = None,
        history_capacity: int | None = None,
    ) -> None:
        """履歴を消去し、新しい比較区間を開始する。

        閾値・tau・間隔を変更して渡すと config fingerprint も再計算されるため、
        新しい区間のエクスポート記録は直前の区間と正しく区別される（§15.2）。
        指標 bundle はこれらに含まれない。§8.6 が調整可能な係数を残さなかった
        ので、セッション中は固定である。``history_capacity`` は有界バッファを
        新しい大きさで作り直す。いずれにせよ区間は消去されるため、場所を空ける
        ために記録が捨てられることはない。
        """
        with self._lock:
            if thresholds is not None:
                self._thresholds = thresholds
            if tau_seconds is not None:
                self._tau_seconds = tau_seconds
            if analysis_interval_ms is not None:
                self._analysis_interval_ms = analysis_interval_ms
            if (
                thresholds is not None
                or tau_seconds is not None
                or analysis_interval_ms is not None
            ):
                self._settings = build_config_settings(
                    self._metric_config_3d,
                    self._thresholds,
                    tau_seconds=self._tau_seconds,
                    analysis_interval_ms=self._analysis_interval_ms,
                )
                self._fingerprint = config_fingerprint(self._settings)
            if history_capacity is not None:
                if history_capacity < 1:
                    raise ValueError("history_capacity は 1 以上である必要があります。")
                self._records = deque(maxlen=history_capacity)
            else:
                self._records.clear()
            self._segment_id = str(uuid4())
            self._config_revision += 1

    def resize_history_capacity(self, capacity: int) -> None:
        """新しい区間を開始せずに、有界履歴バッファを拡大・縮小する。

        ``reset(history_capacity=...)`` と違い、既存の記録は（新しい容量まで、
        古いものから捨てつつ）保持され、区間と config の同一性はそのままである。
        `history_capacity` は採点に影響しない（§15.2）ため、cadence の自動調整
        （§14.3d, FR-030）が動作中にサイズを変更できる。
        """
        if capacity < 1:
            raise ValueError("history_capacity は 1 以上である必要があります。")
        with self._lock:
            self._records = deque(self._records, maxlen=capacity)
