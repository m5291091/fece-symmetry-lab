# Repository Structure

This is the compact, AI-facing repository index for `face-symmetry-lab`. It is an exploration aid, not a specification. `requirement.md` remains the product source of truth and `AGENTS.md` remains the execution policy.

Read this file once at the start of a task. After locating the relevant module, inspect only that module, its corresponding tests, model card, and requirement sections.

## Entry points

| Concern | Path |
|---|---|
| Product requirements | `requirement.md` |
| AI and development policy | `AGENTS.md` |
| Test traceability | `testbox.md` |
| Streamlit application | `src/face_symmetry_lab/ui/app.py` |
| Realtime camera UI | `src/face_symmetry_lab/ui/camera.py` |
| Domain models | `src/face_symmetry_lab/domain/` |
| Upload validation | `src/face_symmetry_lab/application/image_input.py` |
| Realtime use case | `src/face_symmetry_lab/application/realtime.py` |
| Realtime age/gender worker | `src/face_symmetry_lab/application/demographic_worker.py` |
| Session history / export records | `src/face_symmetry_lab/application/session.py` |
| CSV / JSON export adapter | `src/face_symmetry_lab/adapters/export/session_exporter.py` |
| Still-image landmarks | `src/face_symmetry_lab/application/still_image.py` |
| Runtime ports | `src/face_symmetry_lab/application/ports.py` |
| Pinned model manifest | `configs/model-manifest.json` |
| Frozen metric bundle | `configs/metrics/symmetry-2d-v1.json` |
| Landmark map derivation | `scripts/derive_landmark_map.py` |
| Symmetry metric | `src/face_symmetry_lab/domain/symmetry.py` |
| Camera correction / skin-surface metric | `src/face_symmetry_lab/domain/skin_quality.py` |
| Quality gate / hysteresis | `src/face_symmetry_lab/application/quality.py` |
| Frame analysis orchestration | `src/face_symmetry_lab/application/analyzer.py` |
| Landmark model card | `docs/model-cards/mediapipe-face-landmarker-v1.md` |
| Model store (hash + atomic install) | `src/face_symmetry_lab/adapters/io/model_store.py` |
| Model preparation CLI | `scripts/bootstrap_models.py` |
| OpenVINO reference parity (OD-07, dev-only) | `scripts/verify_openvino_parity.py` |
| Cross-ONNX-runtime agreement (OD-07 due diligence, dev-only) | `scripts/verify_onnx_runtime_agreement.py` |
| Frame admission gate | `src/face_symmetry_lab/runtime/frame_scheduler.py` |
| MediaPipe landmark adapter | `src/face_symmetry_lab/adapters/inference/mediapipe_landmarker.py` |
| Tracking-zoom landmark wrapper | `src/face_symmetry_lab/adapters/inference/tracking_zoom_landmarker.py` |
| Tracking-zoom geometry | `src/face_symmetry_lab/domain/tracking_zoom.py` |
| MediaPipe segmentation adapter | `src/face_symmetry_lab/adapters/inference/mediapipe_segmenter.py` |
| Brightness / background compositing | `src/face_symmetry_lab/domain/frame_adjustments.py` |
| 明るさ・追跡ズームの自動調整heuristic | `src/face_symmetry_lab/domain/auto_tuning.py` |
| Segmentation model card | `docs/model-cards/mediapipe-selfie-segmenter.md` |
| Age / gender adapter | `src/face_symmetry_lab/adapters/inference/opencv_demographic_estimator.py` |
| Age / gender model card | `docs/model-cards/openvino-age-gender-retail-0013.md` |
| Architecture decisions | `docs/adr/` |
| Repository verification | `scripts/verify.py` |

## Requirement routing

| Capability | Production | Tests | Requirements |
|---|---|---|---|
| Analysis DTO | `domain/models.py` | `tests/domain/test_models.py` | FR-008, FR-011 |
| Landmark DTO / pixel mapping | `domain/landmarks.py` | `tests/domain/test_landmarks.py` | FR-008, FR-009 |
| Landmark overlay painting | `domain/overlay.py` | `tests/domain/test_overlay.py` | FR-013 |
| Export record DTO | `domain/export.py` | `tests/domain/test_export.py` | FR-017, NFR-020 |
| Metric bundle validation | `domain/metrics.py` | `tests/domain/test_metrics.py` | FR-011, FR-018 |
| Normalization / TLS midline | `domain/geometry.py` | `tests/domain/test_geometry.py` | FR-011, AC-SCORE-02 |
| Symmetry scoring | `domain/symmetry.py` | `tests/domain/test_symmetry.py` | FR-011, AC-SCORE-01 |
| Skin-surface reference scoring | `domain/skin_quality.py` | `tests/domain/test_skin_quality.py` | FR-025, AC-APPEARANCE-01 |
| Head pose | `domain/pose.py` | `tests/domain/test_pose.py` | FR-009, §8.2.1 |
| Display smoothing | `domain/smoothing.py` | `tests/domain/test_smoothing.py` | FR-012, §8.7 |
| Demographic mapping | `domain/demographics.py` | `tests/domain/test_demographics.py` | FR-022, FR-023 |
| Safe image input | `application/image_input.py` | `tests/application/test_image_input.py` | FR-006, FR-007 |
| Realtime frame pipeline | `application/realtime.py` | `tests/application/test_realtime.py` | FR-013, NFR-017 |
| 年齢・性別の独立worker | `application/demographic_worker.py` | `tests/application/test_demographic_worker.py` | FR-022〜024, §14.3 |
| Session recording / fingerprint | `application/session.py` | `tests/application/test_session.py` | FR-016, FR-017, FR-020 |
| Quality gate | `application/quality.py` | `tests/application/test_quality.py` | FR-010, AC-STATE-01 |
| Analysis orchestration | `application/analyzer.py` | `tests/application/test_analyzer.py` | FR-011, FR-012 |
| Latest-frame-wins gate | `runtime/frame_scheduler.py` | `tests/runtime/test_frame_scheduler.py` | NFR-006, NFR-007 |
| Landmark detection | `adapters/inference/mediapipe_landmarker.py` | `tests/adapters/test_mediapipe_landmarker.py` | FR-008, FR-009 |
| Tracking-zoom geometry | `domain/tracking_zoom.py` | `tests/domain/test_tracking_zoom.py` | FR-026, §14.3a, AC-ZOOM-01 |
| Tracking-zoom landmark wrapper | `adapters/inference/tracking_zoom_landmarker.py` | `tests/adapters/test_tracking_zoom_landmarker.py` | FR-026, FR-008, AC-ZOOM-01 |
| Brightness / background compositing | `domain/frame_adjustments.py` | `tests/domain/test_frame_adjustments.py` | FR-027, FR-028, AC-DISPLAY-01 |
| 明るさ・追跡ズームの自動調整heuristic | `domain/auto_tuning.py` | `tests/domain/test_auto_tuning.py` | FR-026, FR-028, FR-029, AC-AUTOTUNE-01 |
| MediaPipe segmentation adapter | `adapters/inference/mediapipe_segmenter.py` | `tests/adapters/test_mediapipe_segmenter.py` | FR-027, AC-DISPLAY-01 |
| OpenCV age / gender inference | `adapters/inference/opencv_demographic_estimator.py` | `tests/adapters/test_opencv_demographic_estimator.py` | FR-022〜024 |
| Metric bundle loading | `adapters/io/metric_config.py` | `tests/adapters/test_metric_config.py` | FR-011, FR-020 |
| CSV / JSON session export | `adapters/export/session_exporter.py` | `tests/adapters/test_session_exporter.py` | FR-017, AC-EXPORT-01 |
| Score presentation | `ui/scores.py` | UI / E2E tests は未整備 | FR-014, §5.3 |
| Still-image landmarks | `application/still_image.py` | `tests/application/test_still_image.py` | FR-006, FR-013 |
| Model manifest / install | `adapters/io/model_store.py` | `tests/adapters/test_model_store.py` | FR-002, AC-MODEL-01 |
| Real-model smoke | `scripts/bootstrap_models.py` | `tests/model/test_real_landmarker.py` | FR-002, §21.1 |
| OpenVINO reference parity | `scripts/verify_openvino_parity.py` | `tests/model/test_openvino_parity.py` | OD-07, FR-022〜024, §21.1 |
| Cross-ONNX-runtime agreement | `scripts/verify_onnx_runtime_agreement.py` | `tests/model/test_onnx_runtime_agreement.py` | OD-07, FR-022〜024, §21.1 |
| Camera composition | `ui/camera.py` | `tests/ui/test_camera.py` | FR-004, FR-005, NFR-009 |
| Camera numeric overlay | `ui/camera_overlay.py` | `tests/ui/test_camera_overlay.py` | FR-013, FR-014 |
| Model readiness UI | `ui/model_setup.py` | `tests/ui/test_model_setup.py` | FR-001, FR-002, FR-020 |
| Streamlit flow | `ui/app.py` | future UI / E2E tests | AC-FLOW-01, AC-AGE-02 |

## Maintenance

- Curated tables above change when responsibilities change.
- The tree below is generated from tracked and non-ignored paths.
- Do not edit between the generated markers.
- Regenerate: `uv run python scripts/generate_structure.py --include-untracked`
- Verify: `uv run python scripts/generate_structure.py --include-untracked --check`

<!-- BEGIN GENERATED TREE -->
_File-list SHA-256: `48e8a4a3396b80e4245b9e2837bca79749da1d5552ec34bfad215b3e8e56d6a9`_

face-symmetry-lab/
- `.antigravity/`
  - `instructions.md`
- `.github/`
  - `workflows/`
    - `ci.yml`
- `.streamlit/`
  - `config.toml`
- `configs/`
  - `metrics/`
    - `symmetry-2d-v1.json`
  - `model-manifest.json`
- `docs/`
  - `adr/`
    - `0001-age-estimation-runtime.md`
    - `0002-head-pose-coordinate-signs.md`
    - `0003-age-gender-and-skin-correction.md`
  - `model-cards/`
    - `mediapipe-face-landmarker-v1.md`
    - `mediapipe-selfie-segmenter.md`
    - `openvino-age-gender-retail-0013.md`
    - `symmetry-2d-v1.md`
- `scripts/`
  - `__init__.py`
  - `bootstrap_models.py`
  - `convert_demographic_model.py`
  - `derive_landmark_map.py`
  - `generate_structure.py`
  - `verify.py`
  - `verify_onnx_runtime_agreement.py`
  - `verify_openvino_parity.py`
- `src/`
  - `face_symmetry_lab/`
    - `adapters/`
      - `export/`
        - … (2 files)
      - `inference/`
        - … (5 files)
      - `io/`
        - … (3 files)
      - … (1 files)
    - `application/`
      - … (9 files)
    - `domain/`
      - … (15 files)
    - `runtime/`
      - … (2 files)
    - `ui/`
      - … (6 files)
    - `__init__.py`
    - `__main__.py`
- `tests/`
  - `adapters/`
    - `test_mediapipe_landmarker.py`
    - `test_mediapipe_segmenter.py`
    - `test_metric_config.py`
    - `test_model_store.py`
    - `test_opencv_demographic_estimator.py`
    - `test_session_exporter.py`
    - `test_tracking_zoom_landmarker.py`
  - `application/`
    - `test_analyzer.py`
    - `test_demographic_worker.py`
    - `test_image_input.py`
    - `test_quality.py`
    - `test_realtime.py`
    - `test_session.py`
    - `test_still_image.py`
  - `domain/`
    - `__init__.py`
    - `test_auto_tuning.py`
    - `test_demographics.py`
    - `test_export.py`
    - `test_frame_adjustments.py`
    - `test_geometry.py`
    - `test_landmarks.py`
    - `test_metrics.py`
    - `test_models.py`
    - `test_overlay.py`
    - `test_pose.py`
    - `test_skin_quality.py`
    - `test_smoothing.py`
    - `test_symmetry.py`
    - `test_tracking_zoom.py`
  - `model/`
    - `conftest.py`
    - `test_demographic_conversion.py`
    - `test_onnx_runtime_agreement.py`
    - `test_openvino_parity.py`
    - `test_real_demographic.py`
    - `test_real_landmarker.py`
    - `test_real_segmenter.py`
  - `repository/`
    - `test_generate_structure.py`
    - `test_testbox_documentation.py`
  - `runtime/`
    - `test_frame_scheduler.py`
  - `ui/`
    - `test_app.py`
    - `test_camera.py`
    - `test_camera_overlay.py`
    - `test_model_setup.py`
  - `__init__.py`
  - `test_cli.py`
- `.gitignore`
- `.python-version`
- `AGENTS.md`
- `CLAUDE.md`
- `CODEX.md`
- `NOTICE`
- `README.md`
- `STRUCTURE.md`
- `pyproject.toml`
- `requirement.md`
- `testbox.md`
- `uv.lock`
<!-- END GENERATED TREE -->
