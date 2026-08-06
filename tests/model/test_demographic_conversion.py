from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from face_symmetry_lab.adapters.io.model_store import (
    DEFAULT_MODELS_DIR,
    InstallStatus,
    asset_status,
)
from face_symmetry_lab.ui.model_setup import demographic_asset, demographic_assets

APP_ROOT = Path(__file__).parents[2]
CONVERTER = APP_ROOT / "scripts" / "convert_demographic_model.py"

#: `age-gender-recognition-retail-0013` の入出力契約（入力 `data` 1x3x62x62、
#: 出力 `prob`/`fc3_a`）を 2 分岐で再現した、完全に合成の最小 OpenVINO IR v11 の
#: 組。1 つ目の分岐の計算 node は、実モデルの SoftMax と同じく意図的に "prob" と
#: 名付けてある。その `Result` sink は OpenVINO で一般的な `<name>/sink_port_0`
#: の自動命名を使うため、この接尾辞を取り除くと計算 node 自身の名前と衝突する。
#: 権利のある重みは一切含まない（§21.4 は実際の顔・モデル asset よりも合成
#: fixture を優先する）。
_SYNTHETIC_IR_XML = """<?xml version="1.0"?>
<net name="synthetic_naming_collision" version="11">
  <layers>
    <layer id="0" name="data" type="Parameter" version="opset1">
      <data shape="1,3,62,62" element_type="f32"/>
    </layer>
    <layer id="1" name="w1_const" type="Const" version="opset1">
      <data element_type="f16" shape="2,3,1,1" offset="0" size="12"/>
    </layer>
    <layer id="2" name="w1_convert" type="Convert" version="opset1">
      <data destination_type="f32"/>
    </layer>
    <layer id="3" name="prob" type="Convolution" version="opset1">
      <data strides="1,1" dilations="1,1" pads_begin="0,0" pads_end="0,0" auto_pad="explicit"/>
    </layer>
    <layer id="4" name="prob/sink_port_0" type="Result" version="opset1">
      <input>
        <port id="0">
          <dim>1</dim><dim>2</dim><dim>62</dim><dim>62</dim>
        </port>
      </input>
    </layer>
    <layer id="5" name="w2_const" type="Const" version="opset1">
      <data element_type="f16" shape="1,3,1,1" offset="12" size="6"/>
    </layer>
    <layer id="6" name="w2_convert" type="Convert" version="opset1">
      <data destination_type="f32"/>
    </layer>
    <layer id="7" name="age_op" type="Convolution" version="opset1">
      <data strides="1,1" dilations="1,1" pads_begin="0,0" pads_end="0,0" auto_pad="explicit"/>
    </layer>
    <layer id="8" name="fc3_a" type="Result" version="opset1">
      <input>
        <port id="0">
          <dim>1</dim><dim>1</dim><dim>62</dim><dim>62</dim>
        </port>
      </input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    <edge from-layer="2" from-port="0" to-layer="3" to-port="1"/>
    <edge from-layer="3" from-port="0" to-layer="4" to-port="0"/>
    <edge from-layer="5" from-port="0" to-layer="6" to-port="0"/>
    <edge from-layer="0" from-port="0" to-layer="7" to-port="0"/>
    <edge from-layer="6" from-port="0" to-layer="7" to-port="1"/>
    <edge from-layer="7" from-port="0" to-layer="8" to-port="0"/>
  </edges>
</net>
"""


def _write_synthetic_ir(tmp_path: Path) -> tuple[Path, Path, Path]:
    """合成 IR の組と、それらを固定する manifest を書き出し、path を返す。"""
    xml_path = tmp_path / "synthetic.xml"
    bin_path = tmp_path / "synthetic.bin"
    xml_path.write_text(_SYNTHETIC_IR_XML, encoding="utf-8")
    weights = np.concatenate(
        [
            np.zeros(6, dtype=np.float16),  # w1: 形状 (2,3,1,1) -> 6 要素
            np.zeros(3, dtype=np.float16),  # w2: 形状 (1,3,1,1) -> 3 要素
        ]
    )
    bin_path.write_bytes(weights.tobytes())

    def _entry(name: str, path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "name": name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    manifest = {
        "models": [
            _entry("demographic_age_gender_xml", xml_path),
            _entry("demographic_age_gender_bin", bin_path),
        ],
        "derived_models": [
            {
                "name": "demographic_age_gender_onnx",
                "size_bytes": 0,
                "sha256": "0" * 64,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return xml_path, bin_path, manifest_path


@pytest.mark.model
def test_conversion_reproduces_the_manifest_pinned_onnx(tmp_path: Path) -> None:
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("開発時変換packageのonnxが現在の環境にありません。")
    xml_asset, bin_asset = demographic_assets()
    xml_path = DEFAULT_MODELS_DIR / xml_asset.filename
    bin_path = DEFAULT_MODELS_DIR / bin_asset.filename
    if (
        asset_status(xml_path, xml_asset) is not InstallStatus.READY
        or asset_status(bin_path, bin_asset) is not InstallStatus.READY
    ):
        pytest.skip("manifest検証済みの公式OpenVINO IRが未配置です。")

    output_path = tmp_path / demographic_asset().filename
    completed = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--xml",
            str(xml_path),
            "--bin",
            str(bin_path),
            "--output",
            str(output_path),
        ],
        cwd=APP_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert asset_status(output_path, demographic_asset()) is InstallStatus.READY


def test_conversion_never_emits_duplicate_onnx_node_names(tmp_path: Path) -> None:
    """`<name>/sink_port_0` の Result が、同名の計算 node と衝突してはならない。

    ONNX の node 名の重複は OpenCV の importer では許容されるが、他の厳格な
    ONNX runtime は拒否する（onnxruntime はそうしたグラフの読み込みを拒む）。
    そのため、どの runtime が読むことになっても、生成物に重複が含まれては
    ならない。
    """
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("開発時変換packageのonnxが現在の環境にありません。")
    xml_path, bin_path, manifest_path = _write_synthetic_ir(tmp_path)
    output_path = tmp_path / "synthetic.onnx"

    completed = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--xml",
            str(xml_path),
            "--bin",
            str(bin_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--accept-new-output",
        ],
        cwd=APP_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    import onnx

    model = onnx.load(str(output_path))
    node_names = [node.name for node in model.graph.node]
    assert len(node_names) == len(set(node_names)), node_names
    assert [output.name for output in model.graph.output] == ["prob", "fc3_a"]
