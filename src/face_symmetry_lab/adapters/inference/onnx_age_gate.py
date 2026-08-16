"""FR-038: `age-gate/1.0.0` の spec 読み込みと ONNX 実行（§10.11, §22.5, ADR-0005）。

`adapters.io.model_store` と同じ規律に従う。**版と hash が合わないものは読まない。**
このアダプタは表示文言も価値判断も持たない。確率をそのまま返すだけで、利用者
向けの固定文言は `ui.age_gate_notice` が持つ。

`age-gate/1.0.0` は `ai.onnx.ml` ドメインの `TreeEnsembleClassifier`（LightGBM
由来の決定木集合）であり、`cv2.dnn.readNetFromONNX` は `nodes_modes`
（`BRANCH_LEQ` / `LEAF`）を解釈できず読み込みに失敗する（実機で確認済み、
ADR-0005）。ニューラルネット用の推論器では原理的に実行できないため、この
モデルに限り `onnxruntime` を用いる（§10.8）。ネットワークアクセスは行わない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import numpy as np

from face_symmetry_lab.domain.age_gate import FEATURE_NAMES

APP_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_AGE_GATE_SPEC_PATH = APP_ROOT / "configs" / "age-gate-v1.json"
DEFAULT_AGE_GATE_MODEL_PATH = APP_ROOT / "models" / "age_gate_model.onnx"

#: ADR-0005・§10.11-3・§10.11-8: 動作点を版として固定する凍結値。
#: これ以外の feature_spec_version / spec_revision の spec は読まない。
FEATURE_SPEC_VERSION: Final[str] = "age-gate-features/1.0.0"
SPEC_REVISION: Final[int] = 3

_OUTPUT_NAME: Final[str] = "probabilities"
_IN_RANGE_INDEX: Final[int] = 1
_CHUNK_BYTES: Final[int] = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AgeGateSpec:
    """凍結された特徴量仕様と動作点 1 件（`configs/age-gate-v1.json`）。"""

    gate_version: str
    feature_spec_version: str
    spec_revision: int
    threshold: float
    feature_names: tuple[str, ...]
    onnx_sha256: str


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_age_gate_spec(path: Path = DEFAULT_AGE_GATE_SPEC_PATH) -> AgeGateSpec:
    """凍結された特徴量仕様と動作点を読み込んで検証する。

    ファイル欠落・JSON 解析失敗・版不一致・特徴量順序の不一致は、すべて
    `ValueError` として失敗させる（想定外の版・並びで黙って動かないため）。
    """
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"age-gate の spec が見つかりません: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"age-gate の spec を解析できません: {path}") from error

    if not isinstance(raw, dict):
        raise ValueError("age-gate の spec の形式が不正です。")

    try:
        feature_spec_version = str(raw["feature_spec_version"])
        gate_version = str(raw["gate_version"])
        spec_revision = int(raw["spec_revision"])
        threshold = float(raw["output"]["threshold"])
        feature_names = tuple(str(name) for name in raw["feature_names"])
        onnx_sha256 = str(raw["onnx_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"age-gate の spec の内容が不正です: {error}") from error

    if feature_spec_version != FEATURE_SPEC_VERSION:
        raise ValueError(
            "age-gate の feature_spec_version が想定と一致しません: "
            f"expected={FEATURE_SPEC_VERSION!r}, actual={feature_spec_version!r}"
        )
    if spec_revision != SPEC_REVISION:
        raise ValueError(
            "age-gate の spec_revision が想定と一致しません: "
            f"expected={SPEC_REVISION!r}, actual={spec_revision!r}"
        )
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            "age-gate の feature_names が domain.age_gate.FEATURE_NAMES と"
            "タプルとして一致しません（並べ替え・欠落のいずれか）。"
        )

    return AgeGateSpec(
        gate_version=gate_version,
        feature_spec_version=feature_spec_version,
        spec_revision=spec_revision,
        threshold=threshold,
        feature_names=feature_names,
        onnx_sha256=onnx_sha256,
    )


class OnnxAgeGateClassifier:
    """`age-gate/1.0.0` の ONNX を 1 回だけロードし、範囲内確率を返す。"""

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_AGE_GATE_MODEL_PATH,
        spec: AgeGateSpec,
    ) -> None:
        if not model_path.is_file():
            raise ValueError(f"age-gate の ONNX が見つかりません: {model_path}")
        actual_sha256 = _file_sha256(model_path)
        if actual_sha256 != spec.onnx_sha256:
            raise ValueError(
                "age-gate の ONNX の sha256 が spec と一致しません: "
                f"expected={spec.onnx_sha256!r}, actual={actual_sha256!r}"
            )

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            raise ValueError(f"age-gate の ONNX を初期化できません: {error}") from error
        self._feature_count = len(spec.feature_names)

    def predict(self, vector: tuple[float, ...]) -> float:
        """`probabilities[:, 1]`（範囲内である確率）を返す。`label` は読まない。"""
        if len(vector) != self._feature_count:
            raise ValueError(
                f"age-gate の特徴ベクトルは {self._feature_count} 次元である必要があります: "
                f"actual={len(vector)}"
            )
        features = np.asarray([vector], dtype=np.float32)
        probabilities = self._session.run([_OUTPUT_NAME], {"features": features})[0]
        return float(probabilities[0][_IN_RANGE_INDEX])
