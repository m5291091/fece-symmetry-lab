import csv
import io
import json

from face_symmetry_lab.adapters.export.session_exporter import (
    csv_bytes,
    export_filename,
    json_bytes,
)
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config


def _bundle():
    config = build_metric_config(
        metric_version="symmetry/test",
        landmark_map_version="map/test",
        landmark_count=16,
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8),), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
            "nose": PartMap(pairs=((9, 10),), min_pairs=1),
            "mouth": PartMap(pairs=((11, 12),), min_pairs=1),
        },
    )
    recorder = SessionRecorder(
        metric_config=config,
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
    )
    recorder.append_error(frame_id=4, elapsed_ms=25, processing_ms=3.4567)
    return recorder.export_bundle()


def test_csv_and_json_records_have_the_same_meaning() -> None:
    bundle = _bundle()
    payload = json.loads(json_bytes(bundle))
    csv_record = next(csv.DictReader(io.StringIO(csv_bytes(bundle).decode("utf-8"))))
    json_record = payload["records"][0]

    assert csv_record["schema_version"] == json_record["schema_version"] == "3.0.0"
    assert csv_record["metric_version"] == json_record["metric_version"]
    assert csv_record["config_fingerprint"] == json_record["config_fingerprint"]
    assert csv_record["segment_id"] == json_record["segment_id"]
    assert csv_record["analysis_status"] == json_record["analysis_status"] == "ERROR"
    assert "skin_quality_score_raw" in csv_record
    assert "skin_quality_score_raw" in json_record
    assert "NaN" not in csv_bytes(bundle).decode("utf-8")
    assert "Infinity" not in json_bytes(bundle).decode("utf-8")


def test_json_contains_the_reproducibility_manifest() -> None:
    manifest = json.loads(json_bytes(_bundle()))["manifest"]

    assert manifest["schema_version"] == "3.0.0"
    assert manifest["metric_version"] == "symmetry/test"
    assert manifest["models"][0]["version"] == "landmark/1"
    assert manifest["config_revisions"][0]["fingerprint"]
    assert manifest["config_revisions"][0]["settings"]["part_pair_counts"]


def test_default_filename_contains_only_the_short_random_session_id() -> None:
    bundle = _bundle()

    assert export_filename(bundle, "csv") == (
        f"face-symmetry-session_{str(bundle.manifest.session_id)[:8]}.csv"
    )
