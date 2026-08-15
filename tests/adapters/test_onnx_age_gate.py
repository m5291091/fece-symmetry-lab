"""FR-038 / AC-AGEGATE-03: 年齢ゲートの spec 読み込みと ONNX adapter。

`model_store.py` の作法に合わせ、**版と hash が合わないものは読まない**。
adapter は表示文言も価値判断も持たず、確率をそのまま返すだけである。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from face_symmetry_lab.adapters.inference.onnx_age_gate import (
    OnnxAgeGateClassifier,
    load_age_gate_spec,
)
from face_symmetry_lab.domain.age_gate import FEATURE_NAMES

APP_ROOT = Path(__file__).resolve().parents[2]
REAL_SPEC = APP_ROOT / "configs" / "age-gate-v1.json"
REAL_ONNX = APP_ROOT / "models" / "age_gate_model.onnx"


def _spec_dict(**overrides: object) -> dict:
    spec = json.loads(REAL_SPEC.read_text(encoding="utf-8"))
    spec.update(overrides)
    return spec


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "age-gate-v1.json"
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_the_real_spec_bundle() -> None:
    spec = load_age_gate_spec(REAL_SPEC)

    assert spec.gate_version == "age-gate/1.0.0"
    assert spec.feature_spec_version == "age-gate-features/1.0.0"
    assert spec.spec_revision == 3
    assert spec.threshold == pytest.approx(0.20)
    assert spec.feature_names == FEATURE_NAMES
    assert spec.onnx_sha256 == "80bcfc05eec12243108e8502f88824e680d3d6f03e9ca6607a1dc711e86e02d3"


def test_rejects_a_feature_spec_version_mismatch(tmp_path: Path) -> None:
    """テスト 13: `feature_spec_version` が違えば読まない。"""
    path = _write_spec(tmp_path, _spec_dict(feature_spec_version="age-gate-features/2.0.0"))

    with pytest.raises(ValueError, match="feature_spec_version"):
        load_age_gate_spec(path)


def test_rejects_a_spec_revision_mismatch(tmp_path: Path) -> None:
    """テスト 13: 動作点の版が違えば読まない。

    再学習しなくても閾値が動けば `spec_revision` は上がる（§22.5）。ここを
    通してしまうと、想定と違う動作点で黙って動く。
    """
    path = _write_spec(tmp_path, _spec_dict(spec_revision=99))

    with pytest.raises(ValueError, match="spec_revision"):
        load_age_gate_spec(path)


def test_rejects_a_feature_name_list_that_disagrees_with_the_domain(tmp_path: Path) -> None:
    """spec と domain の並びがずれたら読み込みの時点で落とす。

    これを通すと、モデルは例外を投げずに違う確率を返す（最も静かな壊れ方）。
    """
    names = list(FEATURE_NAMES)
    names[0], names[1] = names[1], names[0]
    path = _write_spec(tmp_path, _spec_dict(feature_names=names))

    with pytest.raises(ValueError):
        load_age_gate_spec(path)


def test_rejects_a_truncated_feature_name_list(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, _spec_dict(feature_names=list(FEATURE_NAMES)[:-1]))

    with pytest.raises(ValueError):
        load_age_gate_spec(path)


def test_rejects_a_missing_spec_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_age_gate_spec(tmp_path / "absent.json")


@pytest.mark.model
def test_classifier_verifies_the_onnx_hash_before_loading(tmp_path: Path) -> None:
    """テスト 14: sha256 が spec と 1 byte でも違えばロードしない。"""
    tampered = tmp_path / "age_gate_model.onnx"
    tampered.write_bytes(REAL_ONNX.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="sha256"):
        OnnxAgeGateClassifier(model_path=tampered, spec=load_age_gate_spec(REAL_SPEC))


@pytest.mark.model
def test_classifier_returns_the_in_range_probability() -> None:
    """`probabilities[:, 1]` を返す。`label` は読まない。"""
    fixture = json.loads(
        (APP_ROOT / "tests" / "fixtures" / "age_gate_contract_fixture.json").read_text("utf-8")
    )
    classifier = OnnxAgeGateClassifier(model_path=REAL_ONNX, spec=load_age_gate_spec(REAL_SPEC))

    for case in fixture["cases"][:5]:
        actual = classifier.predict(tuple(float(v) for v in case["features"]))
        assert actual == pytest.approx(
            case["expected_probability_in_range"], abs=fixture["tolerance"]
        )
        assert 0.0 <= actual <= 1.0


@pytest.mark.model
def test_classifier_rejects_a_wrong_length_vector() -> None:
    classifier = OnnxAgeGateClassifier(model_path=REAL_ONNX, spec=load_age_gate_spec(REAL_SPEC))

    with pytest.raises(ValueError):
        classifier.predict(tuple(0.0 for _ in range(40)))


@pytest.mark.model
def test_classifier_reuses_one_session() -> None:
    """セッションを毎回作らない（1 件推論のたびにモデルを読み直さない）。"""
    classifier = OnnxAgeGateClassifier(model_path=REAL_ONNX, spec=load_age_gate_spec(REAL_SPEC))
    vector = tuple(0.0 for _ in range(41))

    first = classifier.predict(vector)
    session = classifier._session
    second = classifier.predict(vector)

    assert classifier._session is session
    assert first == pytest.approx(second)


def test_missing_model_file_raises_before_any_inference(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        OnnxAgeGateClassifier(
            model_path=tmp_path / "absent.onnx", spec=load_age_gate_spec(REAL_SPEC)
        )
