"""Load the frozen metric bundle (§8.6, §16 `configs/metrics/`)."""

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
    """Read and validate the pinned metric configuration."""
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
            sensitivity={name: float(value) for name, value in raw["sensitivity"].items()},
            weights={name: float(value) for name, value in raw["weights"].items()},
            sensitivity_frozen=bool(raw.get("sensitivity_frozen", False)),
        )
    except MetricConfigError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise MetricConfigError(f"metric config の内容が不正です: {error}") from error
