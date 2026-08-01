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
| Domain models | `src/face_symmetry_lab/domain/` |
| Upload validation | `src/face_symmetry_lab/application/image_input.py` |
| Runtime ports | `src/face_symmetry_lab/application/ports.py` |
| ONNX age adapter | `src/face_symmetry_lab/adapters/inference/onnx_age_estimator.py` |
| Age model card | `docs/model-cards/age-mobilenetv2-utkface-v1.md` |
| Architecture decisions | `docs/adr/` |
| Repository verification | `scripts/verify.py` |

## Requirement routing

| Capability | Production | Tests | Requirements |
|---|---|---|---|
| Analysis DTO | `domain/models.py` | `tests/domain/test_models.py` | FR-008, FR-011 |
| Age-band mapping | `domain/age.py` | `tests/domain/test_age.py` | FR-022, FR-023 |
| Safe image input | `application/image_input.py` | `tests/application/test_image_input.py` | FR-006, FR-007 |
| ONNX age inference | `adapters/inference/onnx_age_estimator.py` | `tests/adapters/test_onnx_age_estimator.py` | FR-022〜024 |
| Streamlit flow | `ui/app.py` | future UI / E2E tests | AC-FLOW-01, AC-AGE-02 |

## Maintenance

- Curated tables above change when responsibilities change.
- The tree below is generated from tracked and non-ignored paths.
- Do not edit between the generated markers.
- Regenerate: `uv run python scripts/generate_structure.py --include-untracked`
- Verify: `uv run python scripts/generate_structure.py --include-untracked --check`

<!-- BEGIN GENERATED TREE -->
_File-list SHA-256: `6c9b1d86c795b974e824db071bd7dbd3d82a62215667edf178213a0037f54807`_

face-symmetry-lab/
- `.github/`
  - `workflows/`
    - `ci.yml`
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
        - … (2 files)
      - … (1 files)
    - `application/`
      - … (3 files)
    - `domain/`
      - … (3 files)
    - `ui/`
      - … (2 files)
    - `__init__.py`
    - `__main__.py`
- `tests/`
  - `adapters/`
    - `test_onnx_age_estimator.py`
  - `application/`
    - `test_image_input.py`
  - `domain/`
    - `__init__.py`
    - `test_age.py`
    - `test_models.py`
  - `repository/`
    - `test_generate_structure.py`
    - `test_testbox_documentation.py`
  - `ui/`
    - `test_app.py`
  - `__init__.py`
  - `test_cli.py`
- `.gitignore`
- `.python-version`
- `AGENTS.md`
- `README.md`
- `STRUCTURE.md`
- `pyproject.toml`
- `requirement.md`
- `testbox.md`
- `uv.lock`
<!-- END GENERATED TREE -->
