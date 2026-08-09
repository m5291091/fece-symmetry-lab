"""凍結された指標 bundle を読み込む（§8.6, §16 `configs/metrics/`）。

bundle は 2 件並存する。`symmetry-2d/2.0.0` が §8.5 の正中線で折りたたむ距離を、
`symmetry-3d/1.0.0` が §8.5.1 の object symmetry 解析を定義する。版の切り替え
UI ではなく、**指標の定義が 2 つある**という意味である（§15.2, FR-036）。両者は
同じ `landmark_map_version` と同じ対応ペア表を共有するので、読み込み手順も同じ
である。表示・エクスポートが読むのは前者だけで、既定は変わらない。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from face_symmetry_lab.domain.metrics import (
    PART_NAMES,
    MetricConfig,
    MetricConfigError,
    PartMap,
    build_metric_config,
)

APP_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_METRIC_CONFIG_PATH = APP_ROOT / "configs" / "metrics" / "symmetry-2d-v1.json"
#: §8.5.1 の 3D object symmetry 用 bundle。既定にはしない（FR-036）。
SYMMETRY_3D_METRIC_CONFIG_PATH = APP_ROOT / "configs" / "metrics" / "symmetry-3d-v1.json"


def _pairs(raw: Any, part: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list):
        raise MetricConfigError(f"{part} の pairs が配列ではありません。")
    pairs: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            raise MetricConfigError(f"{part} の pairs は 2 要素の配列である必要があります。")
        pairs.append((int(entry[0]), int(entry[1])))
    return tuple(pairs)


def load_metric_config(path: Path = DEFAULT_METRIC_CONFIG_PATH) -> MetricConfig:
    """固定された指標設定を読み込んで検証する。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MetricConfigError(f"metric config が見つかりません: {path}") from error
    except json.JSONDecodeError as error:
        raise MetricConfigError(f"metric config を解析できません: {path}") from error

    if not isinstance(raw, dict):
        raise MetricConfigError("metric config の形式が不正です。")

    try:
        anchors = raw["eye_anchor"]
        parts_raw = raw["parts"]
        parts = {
            name: PartMap(
                pairs=_pairs(parts_raw[name]["pairs"], name),
                min_pairs=int(parts_raw[name]["min_pairs"]),
            )
            for name in PART_NAMES
            if name in parts_raw
        }
        return build_metric_config(
            metric_version=str(raw["metric_version"]),
            landmark_map_version=str(raw["landmark_map_version"]),
            landmark_count=int(raw["landmark_count"]),
            left_eye_corners=(
                int(anchors["left_corners"][0]),
                int(anchors["left_corners"][1]),
            ),
            right_eye_corners=(
                int(anchors["right_corners"][0]),
                int(anchors["right_corners"][1]),
            ),
            midline_indices=tuple(int(index) for index in raw["midline_indices"]),
            parts=parts,
        )
    except MetricConfigError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise MetricConfigError(f"metric config の内容が不正です: {error}") from error
