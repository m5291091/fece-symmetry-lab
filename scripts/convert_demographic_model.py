# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "ml-dtypes==0.5.4",
#   "numpy==2.5.1",
#   "onnx==1.22.0",
#   "protobuf==7.35.1",
# ]
# ///
"""固定された OpenVINO IR の年齢・性別グラフを、決定的な ONNX へ変換する。

開発時専用のツールであり、``age-gender-recognition-retail-0013`` が使う演算の
契約だけを意図的に対象とする。アプリの runtime は生成された ONNX を OpenCV で
読み込み、ONNX や OpenVINO の Python パッケージには依存しない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = APP_ROOT / "configs" / "model-manifest.json"
DEFAULT_XML = APP_ROOT / "models" / "age-gender-recognition-retail-0013.xml"
DEFAULT_BIN = APP_ROOT / "models" / "age-gender-recognition-retail-0013.bin"
DEFAULT_OUTPUT = APP_ROOT / "models" / "age-gender-recognition-retail-0013.onnx"
SOURCE_NAMES = ("demographic_age_gender_xml", "demographic_age_gender_bin")
DERIVED_NAME = "demographic_age_gender_onnx"
CONVERTER_VERSION = "ir11-to-onnx-opset13-v2"
SUPPORTED_OPERATIONS = {"Add", "Convolution", "MaxPool", "ReLU", "SoftMax"}


class ConversionError(RuntimeError):
    """入力元・グラフ契約・生成物のいずれかが不正である。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {entry["name"]: entry for entry in raw["models"]}
    derived = {entry["name"]: entry for entry in raw.get("derived_models", [])}
    try:
        return ({name: sources[name] for name in SOURCE_NAMES}, derived[DERIVED_NAME])
    except KeyError as error:
        raise ConversionError(f"manifest entryがありません: {error.args[0]}") from error


def _verify_source(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file():
        raise ConversionError(f"元assetが見つかりません: {path}")
    if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
        raise ConversionError(f"元assetがmanifestと一致しません: {path.name}")


def _integers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",")]


def _port_shape(layer: ET.Element, section: str) -> list[int]:
    ports = layer.find(section)
    if ports is None or len(ports) != 1:
        raise ConversionError(f"{layer.attrib['name']}の{section} contractが不正です。")
    return [int(dim.text or "") for dim in ports[0].findall("dim")]


def _load_constant(layer: ET.Element, weights: bytes) -> np.ndarray:
    data = layer.find("data")
    if data is None or data.attrib.get("element_type") != "f16":
        raise ConversionError(f"未対応のConstです: {layer.attrib['name']}")
    offset = int(data.attrib["offset"])
    size = int(data.attrib["size"])
    shape = tuple(_integers(data.attrib["shape"]))
    raw = weights[offset : offset + size]
    if len(raw) != size or int(np.prod(shape)) * 2 != size:
        raise ConversionError(f"Const範囲が不正です: {layer.attrib['name']}")
    return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)


def _make_node(
    layer: ET.Element,
    inputs: list[str],
    output: str,
) -> onnx.NodeProto:
    layer_type = layer.attrib["type"]
    name = layer.attrib["name"]
    data = layer.find("data")
    attrs = data.attrib if data is not None else {}
    if layer_type == "Convolution":
        return helper.make_node(
            "Conv",
            inputs,
            [output],
            name=name,
            strides=_integers(attrs["strides"]),
            dilations=_integers(attrs["dilations"]),
            pads=_integers(attrs["pads_begin"]) + _integers(attrs["pads_end"]),
        )
    if layer_type == "Add":
        return helper.make_node("Add", inputs, [output], name=name)
    if layer_type == "ReLU":
        return helper.make_node("Relu", inputs, [output], name=name)
    if layer_type == "MaxPool":
        return helper.make_node(
            "MaxPool",
            inputs,
            [output],
            name=name,
            kernel_shape=_integers(attrs["kernel"]),
            strides=_integers(attrs["strides"]),
            dilations=_integers(attrs["dilations"]),
            pads=_integers(attrs["pads_begin"]) + _integers(attrs["pads_end"]),
            ceil_mode=1 if attrs["rounding_type"] == "ceil" else 0,
        )
    if layer_type == "SoftMax":
        return helper.make_node("Softmax", inputs, [output], name=name, axis=int(attrs["axis"]))
    raise ConversionError(f"未対応の演算です: {name} ({layer_type})")


def build_onnx(xml_path: Path, bin_path: Path) -> onnx.ModelProto:
    """検証済みの OpenVINO IR v11 の組から、固定された ONNX グラフを構築する。"""
    root = ET.parse(xml_path).getroot()
    if root.attrib.get("version") != "11":
        raise ConversionError("OpenVINO IR v11だけを変換できます。")
    layers_parent = root.find("layers")
    edges_parent = root.find("edges")
    if layers_parent is None or edges_parent is None:
        raise ConversionError("IR graphにlayersまたはedgesがありません。")

    layers = {int(layer.attrib["id"]): layer for layer in layers_parent}
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge in edges_parent:
        incoming[int(edge.attrib["to-layer"])].append(
            (int(edge.attrib["to-port"]), int(edge.attrib["from-layer"]))
        )
    weights = bin_path.read_bytes()
    tensors: dict[int, str] = {}
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    graph_inputs: list[onnx.ValueInfoProto] = []
    graph_outputs: list[onnx.ValueInfoProto] = []

    for layer_id, layer in layers.items():
        layer_type = layer.attrib["type"]
        name = layer.attrib["name"]
        source_ids = [source for _, source in sorted(incoming.get(layer_id, []))]
        if layer_type == "Parameter":
            data = layer.find("data")
            shape = _integers(data.attrib.get("shape", "")) if data is not None else []
            if name != "data" or shape != [1, 3, 62, 62]:
                raise ConversionError("入力contractはdata / 1x3x62x62に固定されています。")
            tensors[layer_id] = name
            graph_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        elif layer_type == "Const":
            tensor_name = f"const_{layer_id}"
            tensors[layer_id] = tensor_name
            initializers.append(
                numpy_helper.from_array(_load_constant(layer, weights), tensor_name)
            )
        elif layer_type == "Convert":
            if len(source_ids) != 1:
                raise ConversionError(f"Convert入力が不正です: {name}")
            tensors[layer_id] = tensors[source_ids[0]]
        elif layer_type in SUPPORTED_OPERATIONS:
            output = f"layer_{layer_id}_out"
            inputs = [tensors[source_id] for source_id in source_ids]
            nodes.append(_make_node(layer, inputs, output))
            tensors[layer_id] = output
        elif layer_type == "Result":
            if len(source_ids) != 1:
                raise ConversionError(f"Result入力が不正です: {name}")
            output_name = name.removesuffix("/sink_port_0")
            # `name`（IR 自身の一意な層名、例 "prob/sink_port_0"）を ONNX の
            # node 名として使う。`output_name`（末尾を取り除いた形、例 "prob"）
            # は出力 *tensor* の名前にすぎない。`output_name` を node 名にも
            # 流用すると、すでにその名前を持つ別の node と衝突しうる（OpenVINO の
            # 命名規約は Result の sink を "<producer_name>/sink_port_0" と
            # 名付ける）。OpenCV の importer はこれを許容するが、厳格な ONNX
            # runtime は拒否する。
            nodes.append(
                helper.make_node("Identity", [tensors[source_ids[0]]], [output_name], name=name)
            )
            tensors[layer_id] = output_name
            graph_outputs.append(
                helper.make_tensor_value_info(
                    output_name, TensorProto.FLOAT, _port_shape(layer, "input")
                )
            )
        else:
            raise ConversionError(f"未対応のlayerです: {name} ({layer_type})")

    if [output.name for output in graph_outputs] != ["prob", "fc3_a"]:
        raise ConversionError("出力contractがprob / fc3_aと一致しません。")
    graph = helper.make_graph(
        nodes,
        "age-gender-recognition-retail-0013",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="face-symmetry-lab",
        producer_version=CONVERTER_VERSION,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    metadata = {
        "converter": "scripts/convert_demographic_model.py",
        "converter_version": CONVERTER_VERSION,
        "source_xml_sha256": _sha256(xml_path),
        "source_bin_sha256": _sha256(bin_path),
    }
    for key, value in sorted(metadata.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(model)
    return model


def convert(
    *,
    xml_path: Path,
    bin_path: Path,
    output_path: Path,
    manifest_path: Path,
    accept_new_output: bool,
) -> tuple[int, str]:
    sources, derived = _manifest_entries(manifest_path)
    _verify_source(xml_path, sources[SOURCE_NAMES[0]])
    _verify_source(bin_path, sources[SOURCE_NAMES[1]])
    model = build_onnx(xml_path, bin_path)
    payload = model.SerializeToString(deterministic=True)
    size = len(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if not accept_new_output and (size != derived["size_bytes"] or digest != derived["sha256"]):
        raise ConversionError(
            "生成ONNXがmanifestと一致しません。"
            "recipe更新時だけ --accept-new-output を使用してください。"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, output_path)
    return size, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--accept-new-output", action="store_true")
    args = parser.parse_args()
    try:
        size, digest = convert(
            xml_path=args.xml,
            bin_path=args.bin,
            output_path=args.output,
            manifest_path=args.manifest,
            accept_new_output=args.accept_new_output,
        )
    except (ConversionError, ET.ParseError, json.JSONDecodeError, OSError) as error:
        parser.error(str(error))
    print(f"generated: {args.output}")
    print(f"size_bytes: {size}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
