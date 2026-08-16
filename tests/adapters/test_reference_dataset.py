"""§9.5 / FR-037: `configs/reference/` の凍結アセットを読む adapter。

`adapters/io/model_store.py` の hash 検証と同じ作法に合わせる。**1 byte でも
違えば読まない**、そして**壊れた行を黙って捨てない**（不変条件 11）。

fixture は test の中で生成する。実データを test へ持ち込まないためであり、
そのぶん「正しい CSV とはどういう形か」がこのファイルだけで読める。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from face_symmetry_lab.adapters.io.reference_dataset import (
    SUPPORTED_SCHEMA_VERSION,
    CsvReferenceCellRepository,
    ReferenceDatasetError,
)
from face_symmetry_lab.domain.reference_stats import METRIC_NAMES

_COLUMNS = (
    "race",
    "gender",
    "age",
    "count",
    *[f"{m}_{s}" for m in METRIC_NAMES for s in ("mean", "m2")],
)

_ROWS = [
    ("Asian", "female", 30, 5, 40.0),
    ("Asian", "male", 30, 7, 41.0),
    ("Black", "female", 45, 3, 42.0),
    ("Black", "male", 45, 9, 43.0),
    ("Indian", "female", 22, 4, 44.0),
    ("Indian", "male", 22, 6, 45.0),
    ("Others", "female", 60, 2, 46.0),
    ("Others", "male", 60, 8, 47.0),
    ("White", "female", 70, 10, 48.0),
    ("White", "male", 70, 11, 49.0),
]


def _csv_text(rows: list[tuple[str, str, int, int, float]] | None = None) -> str:
    lines = [",".join(_COLUMNS)]
    for race, gender, age, count, mean in rows if rows is not None else _ROWS:
        values = [race, gender, str(age), str(count)]
        for _ in METRIC_NAMES:
            values.extend([f"{mean:.6f}", f"{float(count):.6f}"])
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def _meta(csv_text: str, **overrides: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "metric_version": "symmetry-3d/1.0.0",
        "landmark_map_version": "mediapipe-478-mirror/1.0.0",
        "source_schema_version": "utkface-symmetry-3d-labelled/3.0.0",
        "source_generated_at": "2026-08-14T03:38:57.977344+00:00",
        "records_sha256": "0" * 64,
        "cells_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        "total_count": sum(row[3] for row in _ROWS),
        "min_group_for_ci": 8,
        "dataset_name": "UTKFace",
        "dataset_url": "https://susanqq.github.io/UTKFace/",
        "license_scope": "non-commercial research only",
        "selection": {"age": "18-75", "max_abs_yaw_deg": "15.0"},
        "warning": "探索値であり、正常値・目標値・個人評価・人物間比較には使用できない。",
    }
    meta.update(overrides)
    return meta


def _write(tmp_path: Path, csv_text: str, meta: dict[str, object]) -> Path:
    cells_path = tmp_path / "utkface_symmetry3d_cells_v1.csv"
    cells_path.write_text(csv_text, encoding="utf-8")
    (tmp_path / "utkface_symmetry3d_cells_v1.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return cells_path


def test_loads_a_well_formed_dataset(tmp_path: Path) -> None:
    """テスト 15: 正常な fixture を読める。"""
    csv_text = _csv_text()
    path = _write(tmp_path, csv_text, _meta(csv_text))

    dataset = CsvReferenceCellRepository(path).load()

    assert len(dataset.cells) == len(_ROWS)
    assert dataset.metadata.schema_version == SUPPORTED_SCHEMA_VERSION
    assert dataset.metadata.license_scope == "non-commercial research only"
    assert dataset.metadata.cells_sha256 == hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    first = dataset.cells[0]
    assert (first.race, first.gender, first.age, first.count) == ("Asian", "female", 30, 5)
    assert first.moments["score_points"].mean == pytest.approx(40.0)


def test_rejects_a_schema_version_mismatch(tmp_path: Path) -> None:
    """テスト 12: `schema_version` が違えば読まない。"""
    csv_text = _csv_text()
    path = _write(
        tmp_path,
        csv_text,
        _meta(csv_text, schema_version="utkface-symmetry-3d-cells/2.0.0"),
    )

    with pytest.raises(ReferenceDatasetError, match="schema_version"):
        CsvReferenceCellRepository(path).load()


def test_rejects_a_sha256_mismatch(tmp_path: Path) -> None:
    """テスト 13: meta に記録された hash と実ファイルが 1 byte でも違えば読まない。"""
    csv_text = _csv_text()
    meta = _meta(csv_text)
    path = _write(tmp_path, csv_text, meta)
    path.write_text(csv_text.replace("40.000000", "40.000001", 1), encoding="utf-8")

    with pytest.raises(ReferenceDatasetError, match="sha256"):
        CsvReferenceCellRepository(path).load()


def test_rejects_a_row_containing_nan(tmp_path: Path) -> None:
    """テスト 14a: NaN の行を黙って捨てず、失敗にする（不変条件 11）。"""
    csv_text = _csv_text().replace("40.000000", "NaN", 1)
    path = _write(tmp_path, csv_text, _meta(csv_text))

    with pytest.raises(ReferenceDatasetError):
        CsvReferenceCellRepository(path).load()


def test_rejects_a_missing_column(tmp_path: Path) -> None:
    """テスト 14b: 欠列。"""
    lines = _csv_text().splitlines()
    lines[0] = lines[0].replace(",score_points_m2", "")
    csv_text = "\n".join(lines) + "\n"
    path = _write(tmp_path, csv_text, _meta(csv_text))

    with pytest.raises(ReferenceDatasetError):
        CsvReferenceCellRepository(path).load()


def test_rejects_a_non_numeric_value(tmp_path: Path) -> None:
    """テスト 14c: 型不正。"""
    csv_text = _csv_text().replace("40.000000", "とても対称", 1)
    path = _write(tmp_path, csv_text, _meta(csv_text))

    with pytest.raises(ReferenceDatasetError):
        CsvReferenceCellRepository(path).load()


def test_rejects_an_unknown_race_label(tmp_path: Path) -> None:
    csv_text = _csv_text().replace("Asian,female", "Martian,female", 1)
    path = _write(tmp_path, csv_text, _meta(csv_text))

    with pytest.raises(ReferenceDatasetError):
        CsvReferenceCellRepository(path).load()


def test_rejects_a_missing_cells_file(tmp_path: Path) -> None:
    with pytest.raises(ReferenceDatasetError):
        CsvReferenceCellRepository(tmp_path / "absent.csv").load()


def test_rejects_a_missing_meta_file(tmp_path: Path) -> None:
    cells_path = tmp_path / "utkface_symmetry3d_cells_v1.csv"
    cells_path.write_text(_csv_text(), encoding="utf-8")

    with pytest.raises(ReferenceDatasetError):
        CsvReferenceCellRepository(cells_path).load()


def test_rejects_a_meta_missing_required_keys(tmp_path: Path) -> None:
    csv_text = _csv_text()
    meta = _meta(csv_text)
    del meta["warning"]
    path = _write(tmp_path, csv_text, meta)

    with pytest.raises(ReferenceDatasetError):
        CsvReferenceCellRepository(path).load()


def test_rejects_an_empty_cell_table(tmp_path: Path) -> None:
    csv_text = ",".join(_COLUMNS) + "\n"
    path = _write(tmp_path, csv_text, _meta(csv_text))

    with pytest.raises(ReferenceDatasetError):
        CsvReferenceCellRepository(path).load()


def test_reader_does_not_open_the_network(tmp_path: Path) -> None:
    """§10.9-3: 参照データを実行時に取得しない。

    `socket.socket` を差し替えて、読み込み経路が 1 度も socket を作らないことを
    見る（罠: 将来 `requests` などを足すとここで落ちる）。
    """
    import socket

    csv_text = _csv_text()
    path = _write(tmp_path, csv_text, _meta(csv_text))

    original = socket.socket

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("参照データの読み込みが network へ触れた")

    socket.socket = _forbidden  # type: ignore[assignment,misc]
    try:
        CsvReferenceCellRepository(path).load()
    finally:
        socket.socket = original  # type: ignore[misc]
