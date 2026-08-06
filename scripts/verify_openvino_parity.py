# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.4.6",
#   "opencv-contrib-python==5.0.0.93",
#   "openvino==2026.2.1",
# ]
# ///
"""公式 OpenVINO IR と同梱 ONNX の、固定入力に対する parity を測る。

開発時専用のツールで、OD-07 の「runtime parity」条項に答える。決定的な合成
tensor 群を、manifest で固定した OpenVINO IR（参照）と、``cv2.dnn.readNetFromONNX``
で読み込んだ manifest 固定の派生 ONNX（候補。``opencv_demographic_estimator``
が使うのとまったく同じ経路）へ通し、年齢と性別確率の最大差を報告する。

``openvino`` をアプリの runtime lock へ加えてはならない（ADR-0003, §5, §10.6）。
そのため本 script は PEP 723 で自前の依存を固定し、``uv sync`` とは別の隔離環境を
解決する ``uv run scripts/verify_openvino_parity.py`` 経由での実行を想定している。
``openvino`` を導入できる開発環境で実行すればよく、ここから ``pyproject.toml`` や
``uv.lock`` に触れることはない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = APP_ROOT / "configs" / "model-manifest.json"
DEFAULT_XML = APP_ROOT / "models" / "age-gender-recognition-retail-0013.xml"
DEFAULT_BIN = APP_ROOT / "models" / "age-gender-recognition-retail-0013.bin"
DEFAULT_ONNX = APP_ROOT / "models" / "age-gender-recognition-retail-0013.onnx"
SOURCE_NAMES = ("demographic_age_gender_xml", "demographic_age_gender_bin")
DERIVED_NAME = "demographic_age_gender_onnx"
OUTPUT_NAMES = ["fc3_a", "prob"]
INPUT_SHAPE = (1, 3, 62, 62)

#: model card の「Release gate」§1 の目標値（docs/model-cards/openvino-age-gender-retail-0013.md）。
DEFAULT_AGE_GATE_YEARS = 0.1
DEFAULT_GENDER_GATE_PROB = 1e-4

#: §21.4 は、実際の顔の切り出しよりも、再配布可能な合成 fixture を優先する。
FIXED_SEED = 20260801


class ParityError(RuntimeError):
    """固定 asset・manifest・runtime のいずれかを検証できなかった。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_asset(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file():
        raise ParityError(f"asset が見つかりません: {path}")
    if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
        raise ParityError(f"asset が manifest と一致しません: {path.name}")


def _manifest_entries(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {entry["name"]: entry for entry in raw["models"]}
    derived = {entry["name"]: entry for entry in raw.get("derived_models", [])}
    try:
        return ({name: sources[name] for name in SOURCE_NAMES}, derived[DERIVED_NAME])
    except KeyError as error:
        raise ParityError(f"manifest entry がありません: {error.args[0]}") from error


def fixed_input_samples() -> list[tuple[str, np.ndarray]]:
    """決定的な tensor 群。飽和端の値と、seed 付きの内部値からなる。

    上で固定した numpy の版があれば、どの標本も機械をまたいでビット単位で再現
    できる。そのため写真を一切配布せずに、2 つの開発環境の parity 実行結果を
    比較できる。
    """
    samples: list[tuple[str, np.ndarray]] = [
        ("min_saturation_0", np.zeros(INPUT_SHAPE, dtype=np.float32)),
        ("max_saturation_255", np.full(INPUT_SHAPE, 255.0, dtype=np.float32)),
        ("mid_gray_127_5", np.full(INPUT_SHAPE, 127.5, dtype=np.float32)),
    ]

    rng = np.random.default_rng(FIXED_SEED)
    for index in range(6):
        tensor = rng.uniform(40.0, 220.0, size=INPUT_SHAPE).astype(np.float32)
        samples.append((f"uniform_random_{index}", tensor))

    # 肌色としてありうる範囲に収めた BGR の空間傾斜（§8.6.1 の切り出し）。
    columns = np.linspace(60.0, 140.0, INPUT_SHAPE[3], dtype=np.float32)
    rows = np.linspace(90.0, 180.0, INPUT_SHAPE[2], dtype=np.float32)
    blue = np.tile(columns[None, :], (INPUT_SHAPE[2], 1))
    green = np.tile(rows[:, None], (1, INPUT_SHAPE[3]))
    red = np.linspace(120.0, 220.0, INPUT_SHAPE[3], dtype=np.float32)
    red = np.tile(red[None, :], (INPUT_SHAPE[2], 1))
    gradient = np.stack([blue, green, red])[np.newaxis, ...].astype(np.float32)
    samples.append(("skin_tone_gradient", gradient))

    for index in range(4):
        tensor = rng.normal(loc=140.0, scale=35.0, size=INPUT_SHAPE).astype(np.float32)
        np.clip(tensor, 0.0, 255.0, out=tensor)
        samples.append((f"gaussian_{index}", tensor))

    return samples


@dataclass(frozen=True, slots=True)
class SampleResult:
    name: str
    reference_age_years: float
    candidate_age_years: float
    age_diff_years: float
    reference_gender_prob: tuple[float, float]
    candidate_gender_prob: tuple[float, float]
    gender_prob_diff: float


def _run_reference(xml_path: Path, bin_path: Path, tensor: np.ndarray) -> tuple[float, np.ndarray]:
    from openvino import Core

    core = Core()
    model = core.read_model(model=str(xml_path), weights=str(bin_path))
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()
    result = request.infer({0: tensor})

    age = gender = None
    for value in result.values():
        array = np.asarray(value).reshape(-1)
        if array.size == 2:
            gender = array
        elif array.size == 1:
            age = array
    if age is None or gender is None:
        raise ParityError("OpenVINO reference の出力shapeが不正です。")
    return float(age[0]), gender


def _run_candidate(onnx_path: Path, tensor: np.ndarray) -> tuple[float, np.ndarray]:
    import cv2

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(tensor)
    age_output, gender_output = net.forward(OUTPUT_NAMES)
    age = np.asarray(age_output, dtype=np.float64).reshape(-1)
    gender = np.asarray(gender_output, dtype=np.float64).reshape(-1)
    if age.size != 1 or gender.size != 2:
        raise ParityError("候補ONNX の出力shapeが不正です。")
    return float(age[0]), gender


def evaluate(
    *,
    xml_path: Path,
    bin_path: Path,
    onnx_path: Path,
    manifest_path: Path,
) -> list[SampleResult]:
    sources, derived = _manifest_entries(manifest_path)
    _verify_asset(xml_path, sources[SOURCE_NAMES[0]])
    _verify_asset(bin_path, sources[SOURCE_NAMES[1]])
    _verify_asset(onnx_path, derived)

    results: list[SampleResult] = []
    for name, tensor in fixed_input_samples():
        reference_age, reference_gender = _run_reference(xml_path, bin_path, tensor)
        candidate_age, candidate_gender = _run_candidate(onnx_path, tensor)
        results.append(
            SampleResult(
                name=name,
                reference_age_years=reference_age * 100.0,
                candidate_age_years=candidate_age * 100.0,
                age_diff_years=abs(reference_age - candidate_age) * 100.0,
                reference_gender_prob=(float(reference_gender[0]), float(reference_gender[1])),
                candidate_gender_prob=(float(candidate_gender[0]), float(candidate_gender[1])),
                gender_prob_diff=float(np.max(np.abs(reference_gender - candidate_gender))),
            )
        )
    return results


def summarize(
    results: list[SampleResult],
    *,
    age_gate_years: float,
    gender_gate_prob: float,
) -> dict[str, Any]:
    max_age_diff = max(sample.age_diff_years for sample in results)
    max_gender_diff = max(sample.gender_prob_diff for sample in results)
    return {
        "sample_count": len(results),
        "max_age_diff_years": max_age_diff,
        "max_gender_prob_diff": max_gender_diff,
        "age_gate_years": age_gate_years,
        "gender_gate_prob": gender_gate_prob,
        "age_gate_passed": max_age_diff <= age_gate_years,
        "gender_gate_passed": max_gender_diff <= gender_gate_prob,
        "samples": [asdict(sample) for sample in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--age-gate-years", type=float, default=DEFAULT_AGE_GATE_YEARS)
    parser.add_argument("--gender-gate-prob", type=float, default=DEFAULT_GENDER_GATE_PROB)
    parser.add_argument("--report", type=Path, default=None, help="JSON report output path")
    args = parser.parse_args()

    try:
        results = evaluate(
            xml_path=args.xml,
            bin_path=args.bin,
            onnx_path=args.onnx,
            manifest_path=args.manifest,
        )
    except ParityError as error:
        parser.error(str(error))

    summary = summarize(
        results,
        age_gate_years=args.age_gate_years,
        gender_gate_prob=args.gender_gate_prob,
    )

    for sample in results:
        print(
            f"{sample.name:24s} "
            f"age_diff_years={sample.age_diff_years:.6f} "
            f"gender_prob_diff={sample.gender_prob_diff:.6e}"
        )
    print("---")
    print(f"sample_count       = {summary['sample_count']}")
    print(
        f"max_age_diff_years = {summary['max_age_diff_years']:.6f} (gate <= {args.age_gate_years})"
    )
    print(
        f"max_gender_prob_diff = {summary['max_gender_prob_diff']:.6e} "
        f"(gate <= {args.gender_gate_prob})"
    )
    print(f"age_gate_passed    = {summary['age_gate_passed']}")
    print(f"gender_gate_passed = {summary['gender_gate_passed']}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.report}")

    passed = summary["age_gate_passed"] and summary["gender_gate_passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
