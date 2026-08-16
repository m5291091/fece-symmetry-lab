"""§9.5 / FR-037: `configs/reference/` の凍結参照アセットを読む adapter。

`adapters/io/model_store.py` の hash 検証と同じ規律に合わせる。**1 byte でも
違えば読まない**、そして **壊れた行を黙って捨てない**（不変条件 11）。ネットワーク
には一切接続しない（§10.9-3）——このモジュールは `pathlib` / `json` / `csv` /
`hashlib` の標準ライブラリだけでファイルを読む。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Final

from face_symmetry_lab.domain.reference_stats import (
    METRIC_NAMES,
    MetricMoments,
    ReferenceCell,
    ReferenceDataset,
    ReferenceDatasetMetadata,
)

APP_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REFERENCE_CELLS_PATH = (
    APP_ROOT / "configs" / "reference" / "utkface_symmetry3d_cells_v1.csv"
)

#: §9.5 のセルテーブルが宣言する schema。これ以外は読まない。
SUPPORTED_SCHEMA_VERSION: Final[str] = "utkface-symmetry-3d-cells/1.0.0"

_REQUIRED_META_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "metric_version",
    "landmark_map_version",
    "source_schema_version",
    "source_generated_at",
    "records_sha256",
    "cells_sha256",
    "total_count",
    "min_group_for_ci",
    "dataset_name",
    "dataset_url",
    "license_scope",
    "selection",
    "warning",
)

_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"race", "gender", "age", "count"}
    | {f"{metric}_{suffix}" for metric in METRIC_NAMES for suffix in ("mean", "m2")}
)


class ReferenceDatasetError(RuntimeError):
    """参照統計データの読み込みまたは検証に失敗した。"""


def _load_metadata(meta_path: Path, csv_bytes: bytes) -> ReferenceDatasetMetadata:
    try:
        raw_text = meta_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ReferenceDatasetError(
            f"参照統計の meta ファイルが見つかりません: {meta_path}"
        ) from error

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ReferenceDatasetError(f"参照統計の meta を解析できません: {meta_path}") from error
    if not isinstance(raw, dict):
        raise ReferenceDatasetError(f"参照統計の meta の形式が不正です: {meta_path}")

    missing_keys = [key for key in _REQUIRED_META_KEYS if key not in raw]
    if missing_keys:
        raise ReferenceDatasetError(
            f"参照統計の meta に必須キーが不足しています: {', '.join(missing_keys)}"
        )

    schema_version = raw["schema_version"]
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ReferenceDatasetError(
            "参照統計の schema_version が一致しません"
            f"（期待値={SUPPORTED_SCHEMA_VERSION!r}, 実際={schema_version!r}）。"
        )

    actual_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    expected_sha256 = raw["cells_sha256"]
    if expected_sha256 != actual_sha256:
        raise ReferenceDatasetError(
            "参照統計セル CSV の sha256 が meta の記録と一致しません"
            f"（meta={expected_sha256!r}, 実ファイル={actual_sha256!r}）。"
        )

    try:
        selection_raw = raw["selection"]
        if not isinstance(selection_raw, dict):
            raise TypeError("selection はオブジェクトである必要があります。")
        return ReferenceDatasetMetadata(
            schema_version=str(raw["schema_version"]),
            metric_version=str(raw["metric_version"]),
            landmark_map_version=str(raw["landmark_map_version"]),
            source_schema_version=str(raw["source_schema_version"]),
            source_generated_at=str(raw["source_generated_at"]),
            records_sha256=str(raw["records_sha256"]),
            cells_sha256=str(raw["cells_sha256"]),
            total_count=int(raw["total_count"]),
            min_group_for_ci=int(raw["min_group_for_ci"]),
            dataset_name=str(raw["dataset_name"]),
            dataset_url=str(raw["dataset_url"]),
            license_scope=str(raw["license_scope"]),
            selection={str(key): str(value) for key, value in selection_raw.items()},
            warning=str(raw["warning"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReferenceDatasetError(f"参照統計の meta の内容が不正です: {error}") from error


def _read_cells(csv_bytes: bytes, cells_path: Path) -> tuple[ReferenceCell, ...]:
    text = csv_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames
    if fieldnames is None or set(fieldnames) != _REQUIRED_COLUMNS:
        raise ReferenceDatasetError(
            f"参照統計セル CSV の列が不正です（{cells_path}）。"
            f"必要な列: {sorted(_REQUIRED_COLUMNS)}"
        )

    cells: list[ReferenceCell] = []
    for row_number, row in enumerate(reader, start=2):  # 1 行目はヘッダー
        try:
            moments = {
                metric: MetricMoments(
                    mean=float(row[f"{metric}_mean"]),
                    m2=float(row[f"{metric}_m2"]),
                )
                for metric in METRIC_NAMES
            }
            cell = ReferenceCell(
                race=row["race"],
                gender=row["gender"],
                age=int(row["age"]),
                count=int(row["count"]),
                moments=moments,
            )
        except (KeyError, TypeError, ValueError) as error:
            # `ReferenceStatsError` は `ValueError` のサブクラスなのでここに含まれる
            # （未知ラベル・範囲外年齢・count<=0・非有限値のいずれも、不変条件 11
            # のとおり黙って捨てず失敗にする）。
            raise ReferenceDatasetError(
                f"参照統計セル CSV の {row_number} 行目が不正です（{cells_path}）: {error}"
            ) from error
        cells.append(cell)

    if not cells:
        raise ReferenceDatasetError(f"参照統計セル CSV に行がありません: {cells_path}")
    return tuple(cells)


class CsvReferenceCellRepository:
    """`configs/reference/` の CSV + `.meta.json` を読む `ReferenceCellRepository`。"""

    def __init__(self, cells_path: Path = DEFAULT_REFERENCE_CELLS_PATH) -> None:
        self._cells_path = cells_path

    def load(self) -> ReferenceDataset:
        try:
            csv_bytes = self._cells_path.read_bytes()
        except FileNotFoundError as error:
            raise ReferenceDatasetError(
                f"参照統計セル CSV が見つかりません: {self._cells_path}"
            ) from error

        meta_path = self._cells_path.with_suffix(".meta.json")
        metadata = _load_metadata(meta_path, csv_bytes)
        cells = _read_cells(csv_bytes, self._cells_path)
        return ReferenceDataset(metadata=metadata, cells=cells)
