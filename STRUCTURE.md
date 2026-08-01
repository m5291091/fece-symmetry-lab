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
| Runtime ports | `src/face_symmetry_lab/application/ports.py` |
| Frame admission gate | `src/face_symmetry_lab/runtime/frame_scheduler.py` |
| MediaPipe landmark adapter | `src/face_symmetry_lab/adapters/inference/mediapipe_landmarker.py` |
| ONNX age adapter | `src/face_symmetry_lab/adapters/inference/onnx_age_estimator.py` |
| Age model card | `docs/model-cards/age-mobilenetv2-utkface-v1.md` |
| Architecture decisions | `docs/adr/` |
| Repository verification | `scripts/verify.py` |

## Requirement routing

| Capability | Production | Tests | Requirements |
|---|---|---|---|
| Analysis DTO | `domain/models.py` | `tests/domain/test_models.py` | FR-008, FR-011 |
| Landmark DTO / pixel mapping | `domain/landmarks.py` | `tests/domain/test_landmarks.py` | FR-008, FR-009 |
| Landmark overlay painting | `domain/overlay.py` | `tests/domain/test_overlay.py` | FR-013 |
| Age-band mapping | `domain/age.py` | `tests/domain/test_age.py` | FR-022, FR-023 |
| Safe image input | `application/image_input.py` | `tests/application/test_image_input.py` | FR-006, FR-007 |
| Realtime frame pipeline | `application/realtime.py` | `tests/application/test_realtime.py` | FR-013, NFR-017 |
| Latest-frame-wins gate | `runtime/frame_scheduler.py` | `tests/runtime/test_frame_scheduler.py` | NFR-006, NFR-007 |
| Landmark detection | `adapters/inference/mediapipe_landmarker.py` | `tests/adapters/test_mediapipe_landmarker.py` | FR-008, FR-009 |
| ONNX age inference | `adapters/inference/onnx_age_estimator.py` | `tests/adapters/test_onnx_age_estimator.py` | FR-022〜024 |
| Camera composition | `ui/camera.py` | `tests/ui/test_camera.py` | FR-004, FR-005, NFR-009 |
| Streamlit flow | `ui/app.py` | future UI / E2E tests | AC-FLOW-01, AC-AGE-02 |

## Maintenance

- Curated tables above change when responsibilities change.
- The tree below is generated from tracked and non-ignored paths.
- Do not edit between the generated markers.
- Regenerate: `uv run python scripts/generate_structure.py --include-untracked`
- Verify: `uv run python scripts/generate_structure.py --include-untracked --check`

<!-- BEGIN GENERATED TREE -->
_File-list SHA-256: `74b4d7310103bf4a8242d3530a1baf30416e393972605ab348182cacf80b09d9`_

face-symmetry-lab/
- `.antigravity/`
  - `instructions.md`
- `.github/`
  - `workflows/`
    - `ci.yml`
  - `copilot-instructions.md`
- `.streamlit/`
  - `config.toml`
- `docs/`
  - `adr/`
    - `0001-age-estimation-runtime.md`
  - `model-cards/`
    - `age-mobilenetv2-utkface-v1.md`
- `scripts/`
  - `__init__.py`
  - `export_age_model.py`
  - `generate_structure.py`
  - `verify.py`
- `src/`
  - `face_symmetry_lab/`
    - `adapters/`
      - `inference/`
        - … (3 files)
      - … (1 files)
    - `application/`
      - … (4 files)
    - `domain/`
      - … (5 files)
    - `runtime/`
      - … (2 files)
    - `ui/`
      - … (3 files)
    - `__init__.py`
    - `__main__.py`
- `tests/`
  - `adapters/`
    - `test_mediapipe_landmarker.py`
    - `test_onnx_age_estimator.py`
  - `application/`
    - `test_image_input.py`
    - `test_realtime.py`
  - `domain/`
    - `__init__.py`
    - `test_age.py`
    - `test_landmarks.py`
    - `test_models.py`
    - `test_overlay.py`
  - `repository/`
    - `test_generate_structure.py`
    - `test_testbox_documentation.py`
  - `runtime/`
    - `test_frame_scheduler.py`
  - `ui/`
    - `test_app.py`
    - `test_camera.py`
  - `__init__.py`
  - `test_cli.py`
- `.gitignore`
- `.python-version`
- `AGENTS.md`
- `CLAUDE.md`
- `CODEX.md`
- `README.md`
- `STRUCTURE.md`
- `pyproject.toml`
- `requirement.md`
- `testbox.md`
- `uv.lock`
<!-- END GENERATED TREE -->
