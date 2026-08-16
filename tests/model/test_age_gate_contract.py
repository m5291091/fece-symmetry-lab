"""FR-038 / AC-AGEGATE-01: 学習時とアプリ側で特徴量の並びが一致していること。

**この test がこの機能で最も重要である。** 41 個の特徴量の並びが 1 つでもずれると、
モデルは例外を投げず、shape も合ったまま、**黙って違う確率を返す**。他のどの test
にも現れない種類の壊れ方なので、学習時に固定した 24 ケースの期待確率と突き合わせて
毎回確かめる。

`models/age_gate_model.onnx` はリポジトリへ同梱されている（§22.5）ので、この test は
skip されない。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.model

APP_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = APP_ROOT / "models" / "age_gate_model.onnx"
SPEC_PATH = APP_ROOT / "configs" / "age-gate-v1.json"
FIXTURE_PATH = APP_ROOT / "tests" / "fixtures" / "age_gate_contract_fixture.json"


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_onnx_sha256_matches_every_artifact_that_claims_it() -> None:
    """テスト 2: hash が spec・fixture・manifest の 3 つと一致する。

    どれか 1 つでも古いまま残ると、「どの重みで測った期待確率なのか」が
    追えなくなる（不変条件 6）。
    """
    digest = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()
    manifest = json.loads((APP_ROOT / "configs" / "model-manifest.json").read_text("utf-8"))
    bundled = next(m for m in manifest["bundled_models"] if m["name"] == "age_gate")

    assert digest == _spec()["onnx_sha256"]
    assert digest == _fixture()["onnx_sha256"]
    assert digest == bundled["sha256"]
    assert bundled["source"] == "bundled"


def test_feature_order_is_identical_across_spec_fixture_and_domain() -> None:
    """テスト 3: 特徴量の順序が spec・fixture・domain 実装で完全一致する。

    **タプルとして比較する。** 集合や長さの比較では並べ替えを見逃す——そして
    並べ替えこそが、このモデルを黙って壊す唯一の失敗モードである。
    """
    from face_symmetry_lab.domain.age_gate import FEATURE_NAMES

    spec_names = tuple(_spec()["feature_names"])

    assert spec_names == FEATURE_NAMES
    assert tuple(_fixture()["feature_names"]) == spec_names
    assert len(FEATURE_NAMES) == 41


def test_threshold_is_the_same_in_spec_and_fixture() -> None:
    """動作点が 2 か所に書かれて食い違わないこと（§22.5）。"""
    assert _fixture()["threshold"] == _spec()["output"]["threshold"]


def test_spec_revision_and_gate_version_are_pinned() -> None:
    spec = _spec()

    assert spec["gate_version"] == "age-gate/1.0.0"
    assert spec["spec_revision"] == 3
    assert spec["output"]["threshold"] == pytest.approx(0.20)
    assert spec["output"]["probabilities_index_in_range"] == 1


def test_every_fixture_case_reproduces_its_expected_probability() -> None:
    """テスト 1（本命）: 24 ケースが絶対誤差 1e-5 以内で一致する。

    `probabilities` を読む。`label` は静的 shape `[1]` で宣言されており、
    値も使わない（モデルカード）。
    """
    import onnxruntime as ort

    fixture = _fixture()
    tolerance = float(fixture["tolerance"])
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    worst = 0.0
    for case in fixture["cases"]:
        vector = np.asarray([case["features"]], dtype=np.float32)
        assert vector.shape == (1, 41)
        probabilities = session.run(["probabilities"], {"features": vector})[0]
        actual = float(probabilities[0][1])
        expected = float(case["expected_probability_in_range"])
        worst = max(worst, abs(actual - expected))
        assert actual == pytest.approx(expected, abs=tolerance), case["case_id"]

    assert worst < tolerance


def test_the_gate_actually_rejects_someone() -> None:
    """テスト 8 の罠: 「常に範囲内を返す実装」ではこの test が落ちる。

    24 ケースのうち少なくとも 1 件は、凍結された閾値で「範囲外」にならなければ
    ならない。すべて範囲内になる fixture は、ゲートが働いている証拠を何も
    与えない。
    """
    import onnxruntime as ort

    fixture = _fixture()
    threshold = float(fixture["threshold"])
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    vectors = np.asarray([case["features"] for case in fixture["cases"]], dtype=np.float32)
    probabilities = session.run(["probabilities"], {"features": vectors})[0]
    decisions = [float(row[1]) > threshold for row in probabilities]

    assert not all(decisions), "凍結した閾値でどのケースも弾かれないなら罠が成立しない"
    assert any(decisions), "全件が範囲外になるなら閾値か特徴量が壊れている"


def test_batch_and_single_inference_agree() -> None:
    """1 件ずつ流しても 24 件まとめて流しても同じ確率になる。

    `label` 出力が静的 shape `[1]` なので、バッチで流すと onnxruntime が警告を
    出す。**`probabilities` だけを要求している限り結果は実質変わらない**ことを固定し、
    将来 `label` を読む実装へ変わったら気づけるようにする。

    許容は contract と同じ 1e-5。実測の差は最大 6e-8 程度（float32 の丸め由来）で、
    **完全一致ではない**——`abs=1e-9` で書くと落ちる。アプリは 1 件ずつしか
    推論しないので、この差が表示へ出ることはない。
    """
    import onnxruntime as ort

    fixture = _fixture()
    tolerance = float(fixture["tolerance"])
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    cases = fixture["cases"]

    batch = session.run(
        ["probabilities"],
        {"features": np.asarray([c["features"] for c in cases], dtype=np.float32)},
    )[0]
    singles = [
        session.run(["probabilities"], {"features": np.asarray([c["features"]], dtype=np.float32)})[
            0
        ][0][1]
        for c in cases
    ]

    for row, single in zip(batch, singles, strict=True):
        assert float(row[1]) == pytest.approx(float(single), abs=tolerance)
