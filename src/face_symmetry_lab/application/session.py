"""Session-scoped numeric history and reproducibility metadata (§9, §14.2)."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from typing import cast
from uuid import uuid4

from pydantic import JsonValue

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.domain.export import (
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
from face_symmetry_lab.domain.skin_quality import SKIN_METRIC_VERSION

DEFAULT_HISTORY_CAPACITY = 3_000


def build_config_settings(
    config: MetricConfig,
    thresholds: QualityThresholds,
    *,
    tau_seconds: float,
    analysis_interval_ms: int | None = None,
) -> dict[str, JsonValue]:
    """Return the normalized settings whose digest defines comparability."""
    settings: dict[str, JsonValue] = {
        "metric_version": config.metric_version,
        "landmark_map_version": config.landmark_map_version,
        # §8.6 left the bundle with no coefficient, so the pair counts that
        # decide how the pooled overall distance is composed are what makes
        # two segments comparable.
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
        "skin_metric_version": SKIN_METRIC_VERSION,
        "skin_quality_frozen": False,
    }
    if analysis_interval_ms is not None:
        settings["analysis_interval_ms"] = analysis_interval_ms
    return settings


def config_fingerprint(settings: Mapping[str, object]) -> str:
    """Hash canonical JSON so key order and formatting cannot change identity."""
    canonical = json.dumps(
        dict(settings),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


class SessionRecorder:
    """Thread-safe bounded history shared by a session's callback and UI."""

    def __init__(
        self,
        *,
        metric_config: MetricConfig,
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
        self._thresholds = thresholds
        self._tau_seconds = tau_seconds
        self._analysis_interval_ms = analysis_interval_ms
        self._settings = build_config_settings(
            metric_config,
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
    ) -> ExportRecord:
        symmetry = outcome.symmetry
        pose = outcome.pose
        skin = outcome.skin_quality
        # §8.6.1: a skin result may exist without a usable white-balance
        # correction, so the two are absent independently.
        correction = skin.correction if skin is not None else None
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
                distance_contour_raw=symmetry.distance("contour") if symmetry is not None else None,
                distance_eyes_raw=symmetry.distance("eyes") if symmetry is not None else None,
                distance_nose_raw=symmetry.distance("nose") if symmetry is not None else None,
                distance_mouth_raw=symmetry.distance("mouth") if symmetry is not None else None,
                distance_overall_raw=symmetry.overall if symmetry is not None else None,
                distance_overall_smooth=outcome.overall_distance_smooth,
                skin_quality_score_raw=skin.score if skin is not None else None,
                skin_tone_uniformity_raw=skin.tone_uniformity if skin is not None else None,
                skin_texture_uniformity_raw=skin.texture_uniformity if skin is not None else None,
                skin_correction_gain_r=correction.gain_r if correction is not None else None,
                skin_correction_gain_g=correction.gain_g if correction is not None else None,
                skin_correction_gain_b=correction.gain_b if correction is not None else None,
                skin_metric_version=skin.metric_version if skin is not None else None,
                skin_quality_frozen=skin.frozen if skin is not None else None,
                landmark_model_version=self._landmark_model_version,
                landmark_model_sha256=self._landmark_model_sha256,
                metric_version=outcome.metric_version,
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
                metric_version=self._metric_config.metric_version,
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
        """Attach an independently produced age/gender result to the latest record."""
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
            metric_version=self._metric_config.metric_version,
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
        """Clear history and begin a new comparison segment.

        Passing a changed threshold/tau/interval recomputes the config
        fingerprint too, so the new segment's exported records are correctly
        distinguished from the one before it (§15.2). The metric bundle is
        not among them: §8.6 left it with no tunable coefficient, so it is
        pinned for the session. ``history_capacity`` rebuilds the bounded
        buffer at the new size; the segment is being cleared either way, so
        no record is dropped to make room.
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
                    self._metric_config,
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
        """Grow/shrink the bounded history buffer without starting a new segment.

        Unlike ``reset(history_capacity=...)``, existing records (up to the
        new capacity, oldest dropped first) are preserved and segment/config
        identity is untouched: `history_capacity` does not affect scoring
        (§15.2), so cadence auto-tuning (§14.3d, FR-030) can resize it live.
        """
        if capacity < 1:
            raise ValueError("history_capacity は 1 以上である必要があります。")
        with self._lock:
            self._records = deque(self._records, maxlen=capacity)
