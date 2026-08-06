# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.4.6",
#   "opencv-contrib-python==5.0.0.93",
#   "openvino==2026.2.1",
#   "onnxruntime==1.20.1",
# ]
# ///
"""OD-07 の due diligence: parity の差は変換の bug か、backend の特性か。

``verify_openvino_parity.py`` は、同梱 ONNX（アプリの実 runtime である
``cv2.dnn`` で実行）が OpenVINO 参照からどれだけずれるかを測る。その差は model
card の目標ゲート（docs/model-cards/openvino-age-gender-retail-0013.md）を
超えている。本 script は同じ固定入力に対して 3 つ目の独立した ONNX runtime
（``onnxruntime``）を加え、差の出どころを推測せずに絞り込めるようにする。

- ``onnxruntime`` も OpenVINO とおおむね同じだけ食い違うなら、ONNX グラフ自体は
  元の IR に忠実であり、ずれは OpenVINO CPU 固有の数値的特性（元の FP16 IR の
  扱い方など）である。変換手順を変えても埋まる見込みは薄い。
- 逆に ``onnxruntime`` が OpenVINO とよく一致し ``cv2.dnn`` だけが外れるなら、
  OpenCV 固有の実行差を示している。変換手順、またはアプリが ``cv2.dnn`` を
  呼び出す方法を、さらに調べる価値がある。

``verify_openvino_parity.py`` と同様、これは開発時専用である。``openvino`` と
``onnxruntime`` をアプリの runtime lock へ加えてはならない（ADR-0003, §5, §10.6）。
``uv sync`` とは別の隔離環境を解決する
``uv run scripts/verify_onnx_runtime_agreement.py`` 経由で実行する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


class RuntimeAgreementError(RuntimeError):
    """固定 asset・manifest・runtime のいずれかを検証できなかった。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_asset(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeAgreementError(f"asset が見つかりません: {path}")
    if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
        raise RuntimeAgreementError(f"asset が manifest と一致しません: {path.name}")


def _manifest_entries(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {entry["name"]: entry for entry in raw["models"]}
    derived = {entry["name"]: entry for entry in raw.get("derived_models", [])}
    try:
        return ({name: sources[name] for name in SOURCE_NAMES}, derived[DERIVED_NAME])
    except KeyError as error:
        raise RuntimeAgreementError(f"manifest entry がありません: {error.args[0]}") from error


@dataclass(frozen=True, slots=True)
class SampleAgreement:
    name: str
    ov_age_years: float
    cv_age_years: float
    ort_age_years: float
    ov_cv_age_diff: float
    ov_ort_age_diff: float
    cv_ort_age_diff: float
    ov_cv_gender_diff: float
    ov_ort_gender_diff: float
    cv_ort_gender_diff: float


def _run_openvino(xml_path: Path, bin_path: Path, tensor: np.ndarray) -> tuple[float, np.ndarray]:
    from openvino import Core

    core = Core()
    model = core.read_model(model=str(xml_path), weights=str(bin_path))
    compiled = core.compile_model(model, "CPU")
    result = compiled.create_infer_request().infer({0: tensor})
    age = gender = None
    for value in result.values():
        array = np.asarray(value).reshape(-1)
        if array.size == 2:
            gender = array
        elif array.size == 1:
            age = array
    if age is None or gender is None:
        raise RuntimeAgreementError("OpenVINO の出力shapeが不正です。")
    return float(age[0]) * 100.0, gender.astype(np.float64)


def _run_opencv(onnx_path: Path, tensor: np.ndarray) -> tuple[float, np.ndarray]:
    import cv2

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(tensor)
    age_out, gender_out = net.forward(OUTPUT_NAMES)
    return (
        float(np.asarray(age_out, dtype=np.float64).reshape(-1)[0]) * 100.0,
        np.asarray(gender_out, dtype=np.float64).reshape(-1),
    )


def _run_onnxruntime(onnx_path: Path, tensor: np.ndarray) -> tuple[float, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    age_out, gender_out = session.run(OUTPUT_NAMES, {input_name: tensor})
    return (
        float(np.asarray(age_out, dtype=np.float64).reshape(-1)[0]) * 100.0,
        np.asarray(gender_out, dtype=np.float64).reshape(-1),
    )


def evaluate(
    *,
    xml_path: Path,
    bin_path: Path,
    onnx_path: Path,
    manifest_path: Path,
) -> list[SampleAgreement]:
    sources, derived = _manifest_entries(manifest_path)
    _verify_asset(xml_path, sources[SOURCE_NAMES[0]])
    _verify_asset(bin_path, sources[SOURCE_NAMES[1]])
    _verify_asset(onnx_path, derived)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import verify_openvino_parity as parity
    finally:
        sys.path.remove(str(Path(__file__).resolve().parent))

    results: list[SampleAgreement] = []
    for name, tensor in parity.fixed_input_samples():
        ov_age, ov_gender = _run_openvino(xml_path, bin_path, tensor)
        cv_age, cv_gender = _run_opencv(onnx_path, tensor)
        ort_age, ort_gender = _run_onnxruntime(onnx_path, tensor)
        results.append(
            SampleAgreement(
                name=name,
                ov_age_years=ov_age,
                cv_age_years=cv_age,
                ort_age_years=ort_age,
                ov_cv_age_diff=abs(ov_age - cv_age),
                ov_ort_age_diff=abs(ov_age - ort_age),
                cv_ort_age_diff=abs(cv_age - ort_age),
                ov_cv_gender_diff=float(np.max(np.abs(ov_gender - cv_gender))),
                ov_ort_gender_diff=float(np.max(np.abs(ov_gender - ort_gender))),
                cv_ort_gender_diff=float(np.max(np.abs(cv_gender - ort_gender))),
            )
        )
    return results


def summarize(results: list[SampleAgreement]) -> dict[str, Any]:
    return {
        "sample_count": len(results),
        "max_ov_cv_age_diff_years": max(r.ov_cv_age_diff for r in results),
        "max_ov_ort_age_diff_years": max(r.ov_ort_age_diff for r in results),
        "max_cv_ort_age_diff_years": max(r.cv_ort_age_diff for r in results),
        "max_ov_cv_gender_diff": max(r.ov_cv_gender_diff for r in results),
        "max_ov_ort_gender_diff": max(r.ov_ort_gender_diff for r in results),
        "max_cv_ort_gender_diff": max(r.cv_ort_gender_diff for r in results),
        "samples": [asdict(r) for r in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=None, help="JSON report output path")
    args = parser.parse_args()

    try:
        results = evaluate(
            xml_path=args.xml,
            bin_path=args.bin,
            onnx_path=args.onnx,
            manifest_path=args.manifest,
        )
    except RuntimeAgreementError as error:
        parser.error(str(error))

    summary = summarize(results)
    print(
        f"{'sample':<20} {'ov-cv age':>10} {'ov-ort age':>11} {'cv-ort age':>11} "
        f"{'ov-cv gender':>13} {'ov-ort gender':>14} {'cv-ort gender':>14}"
    )
    for r in results:
        print(
            f"{r.name:<20} {r.ov_cv_age_diff:>10.4f} {r.ov_ort_age_diff:>11.4f} "
            f"{r.cv_ort_age_diff:>11.4f} {r.ov_cv_gender_diff:>13.4e} "
            f"{r.ov_ort_gender_diff:>14.4e} {r.cv_ort_gender_diff:>14.4e}"
        )
    print("---")
    print(
        f"max age diff:    ov-cv={summary['max_ov_cv_age_diff_years']:.6f}  "
        f"ov-ort={summary['max_ov_ort_age_diff_years']:.6f}  "
        f"cv-ort={summary['max_cv_ort_age_diff_years']:.6f}"
    )
    print(
        f"max gender diff: ov-cv={summary['max_ov_cv_gender_diff']:.6e}  "
        f"ov-ort={summary['max_ov_ort_gender_diff']:.6e}  "
        f"cv-ort={summary['max_cv_ort_gender_diff']:.6e}"
    )
    print(
        "interpretation: if cv-ort is far smaller than ov-cv/ov-ort, the two "
        "independent ONNX runtimes agree and OpenVINO's CPU execution of the "
        "original FP16 IR is the likely source of the gap, not the conversion."
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
