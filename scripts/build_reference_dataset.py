#!/usr/bin/env python3
"""個票 → §9.5 の参照統計セルテーブル（dev 用、§10.9-3）。

UTKFace の個票（`research/utkface-symmetry/` の notebook が出力した 4 ファイル）
から `(race, gender, age)` 粒度のセルテーブルを生成し、`configs/reference/` へ
書く。アプリの実行経路からは呼ばれない——開発者が手元で 1 回実行し、成果物を
リポジトリへコミットする（ADR-0004）。

    uv run python scripts/build_reference_dataset.py

個票・欠列・非有限値のいずれも黙って捨てず、非ゼロ終了で知らせる（不変条件
11）。書き出す前に、生成したセルから 1 軸集計・全件集計を再構成し、
`group_means.csv` / `measurement_stats.csv` の該当値と突き合わせる自己検証を
行う。一致しなければファイルを書かずに非ゼロ終了する。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from face_symmetry_lab.adapters.io.reference_dataset import (  # noqa: E402
    DEFAULT_REFERENCE_CELLS_PATH,
    SUPPORTED_SCHEMA_VERSION,
)
from face_symmetry_lab.domain.reference_stats import (  # noqa: E402
    METRIC_NAMES,
    MetricMoments,
    ReferenceCell,
    ReferenceFilter,
    ReferenceStatsError,
    combine,
    select_cells,
)

DEFAULT_RECORDS_PATH = REPOSITORY_ROOT / "data" / "utkface" / "utkface_symmetry3d_records.csv"
DEFAULT_GROUP_MEANS_PATH = (
    REPOSITORY_ROOT / "data" / "utkface" / "utkface_symmetry3d_group_means.csv"
)
DEFAULT_MEASUREMENT_STATS_PATH = (
    REPOSITORY_ROOT / "data" / "utkface" / "utkface_symmetry3d_measurement_stats.csv"
)
DEFAULT_SUMMARY_PATH = REPOSITORY_ROOT / "data" / "utkface" / "utkface_symmetry3d_summary.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REFERENCE_CELLS_PATH.parent

#: §9.5: 浮動小数の丸め方を 1 か所に決める。決定論的な出力の前提。
_FLOAT_DECIMALS: Final[int] = 10

#: measurement_stats.csv の `measurement` 名は一部が別名になっている。
_MEASUREMENT_ALIAS: Final[dict[str, str]] = {
    "score_points": "overall_score_points",
    "s3d_distance": "distance_s3d_equal_weight",
}


class _BuildError(Exception):
    """入力の不正、または自己検証の不一致（非ゼロ終了で知らせる）。"""


def _format_float(value: float) -> str:
    return f"{value:.{_FLOAT_DECIMALS}f}"


def _parse_record_row(
    row: dict[str, str], row_number: int
) -> tuple[str, str, int, dict[str, float]]:
    try:
        race = row["race"]
        gender = row["gender"]
        age = int(row["age"])
        metrics = {name: float(row[name]) for name in METRIC_NAMES}
    except (KeyError, TypeError, ValueError) as error:
        raise _BuildError(f"records の {row_number} 行目が不正です: {error}") from error

    for name, value in metrics.items():
        if not math.isfinite(value):
            raise _BuildError(f"records の {row_number} 行目の {name} が非有限値です: {value!r}")
    return race, gender, age, metrics


def _aggregate_cells(records_path: Path) -> tuple[ReferenceCell, ...]:
    """個票を `(race, gender, age)` へ集計し、版付きの `ReferenceCell` を得る。

    18〜75 の範囲外・未知ラベルは `ReferenceCell.__post_init__` が
    `ReferenceStatsError` として拒否する。ここではそれを捕まえて `_BuildError`
    へ変換するだけで、検証ルールを重複させない。
    """
    groups: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    with records_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            race, gender, age, metrics = _parse_record_row(row, row_number)
            bucket = groups.setdefault((race, gender, age), {name: [] for name in METRIC_NAMES})
            for name, value in metrics.items():
                bucket[name].append(value)

    if not groups:
        raise _BuildError(f"records に有効な行がありません: {records_path}")

    cells: list[ReferenceCell] = []
    for race, gender, age in sorted(groups):
        values_by_metric = groups[(race, gender, age)]
        count = len(values_by_metric[METRIC_NAMES[0]])
        moments: dict[str, MetricMoments] = {}
        for name in METRIC_NAMES:
            values = values_by_metric[name]
            mean = statistics.fmean(values)
            m2 = sum((value - mean) ** 2 for value in values)
            moments[name] = MetricMoments(mean=mean, m2=m2)
        try:
            cell = ReferenceCell(race=race, gender=gender, age=age, count=count, moments=moments)
        except ReferenceStatsError as error:
            raise _BuildError(
                f"セル ({race}, {gender}, {age}) の検証に失敗しました: {error}"
            ) from error
        cells.append(cell)
    return tuple(cells)


def _render_cells_csv(cells: Sequence[ReferenceCell]) -> bytes:
    """§9.5: `race, gender, age, count` の後に指標ごとの `mean`/`m2`。

    `filename` / `image_id` は一切書かない（§10.9-2）。行順は
    `_aggregate_cells` がすでに race → gender → age 昇順に並べている。
    """
    header = ["race", "gender", "age", "count"]
    for metric in METRIC_NAMES:
        header.extend([f"{metric}_mean", f"{metric}_m2"])

    lines = [",".join(header)]
    for cell in cells:
        fields = [cell.race, cell.gender, str(cell.age), str(cell.count)]
        for metric in METRIC_NAMES:
            moment = cell.moments[metric]
            fields.append(_format_float(moment.mean))
            fields.append(_format_float(moment.m2))
        lines.append(",".join(fields))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_metadata(
    *,
    cells: tuple[ReferenceCell, ...],
    records_sha256: str,
    cells_sha256: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    try:
        dataset = summary["dataset"]
        protocol = summary["protocol"]
        reproducibility = summary["reproducibility"]
        metadata = {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "metric_version": str(reproducibility["metric_version"]),
            "landmark_map_version": str(reproducibility["landmark_map_version"]),
            "source_schema_version": str(summary["schema_version"]),
            "source_generated_at": str(summary["generated_at"]),
            "records_sha256": records_sha256,
            "cells_sha256": cells_sha256,
            "total_count": sum(cell.count for cell in cells),
            "min_group_for_ci": int(protocol["min_group_for_ci"]),
            "dataset_name": str(dataset["name"]),
            "dataset_url": str(dataset["official_page"]),
            "license_scope": str(dataset["license_scope"]),
            "selection": {
                "age": f"{int(protocol['minimum_age'])}-{int(protocol['maximum_age'])}",
                "max_abs_yaw_deg": str(protocol["max_abs_yaw_deg"]),
                "max_abs_pitch_deg": str(protocol["max_abs_pitch_deg"]),
                "max_abs_roll_deg": str(protocol["max_abs_roll_deg"]),
            },
            "warning": str(summary["warning"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise _BuildError(f"summary.json の内容が不正です: {error}") from error
    return metadata


def _read_group_means(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_measurement_stats(path: Path) -> dict[str, dict[str, str]]:
    alias_to_metric = {alias: metric for metric, alias in _MEASUREMENT_ALIAS.items()}
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = alias_to_metric.get(row["measurement"], row["measurement"])
            if metric in METRIC_NAMES:
                rows[metric] = row
    return rows


def _compare_aggregate(label: str, aggregate: Any, expected_row: dict[str, str]) -> list[str]:
    problems: list[str] = []
    if aggregate is None:
        return [f"{label}: 一致するセルがありません。"]

    expected_count = int(expected_row["count"])
    if aggregate.count != expected_count:
        problems.append(
            f"{label}: count が一致しません（期待={expected_count}, 実際={aggregate.count}）。"
        )

    for name in METRIC_NAMES:
        key = "mean_score_points" if name == "score_points" else f"mean_{name}"
        if key not in expected_row:
            continue
        expected_mean = float(expected_row[key])
        summary = aggregate.metrics.get(name)
        if summary is None or not math.isclose(summary.mean, expected_mean, rel_tol=1e-6):
            problems.append(f"{label}: {name} の平均が一致しません。")

    expected_sd_raw = expected_row.get("standard_deviation", "")
    if expected_sd_raw:
        expected_sd = float(expected_sd_raw)
        summary = aggregate.metrics.get("score_points")
        actual_sd = summary.sd if summary is not None else None
        if actual_sd is None or not math.isclose(actual_sd, expected_sd, rel_tol=1e-6):
            problems.append(f"{label}: score_points の標準偏差が一致しません。")
    return problems


def _compare_single_metric(
    label: str, aggregate: Any, *, metric: str, row: dict[str, str]
) -> list[str]:
    """measurement_stats.csv 用の突き合わせ。1 行が 1 指標分の全件統計を持つ。

    `_compare_aggregate` は 1 行に 7 指標ぶんの列が並ぶ group_means.csv 用で
    あり、行の形が異なる measurement_stats.csv には使えない（列名を使い回すと
    無関係な指標同士を比較してしまう）。
    """
    problems: list[str] = []
    if aggregate is None:
        return [f"{label}: 一致するセルがありません。"]

    expected_count = int(float(row["count"]))
    if aggregate.count != expected_count:
        problems.append(
            f"{label}: count が一致しません（期待={expected_count}, 実際={aggregate.count}）。"
        )

    expected_mean = float(row["mean"])
    summary = aggregate.metrics.get(metric)
    if summary is None or not math.isclose(summary.mean, expected_mean, rel_tol=1e-6):
        problems.append(f"{label}: {metric} の平均が一致しません。")

    expected_sd_raw = row.get("standard_deviation", "")
    if expected_sd_raw:
        expected_sd = float(expected_sd_raw)
        actual_sd = summary.sd if summary is not None else None
        if actual_sd is None or not math.isclose(actual_sd, expected_sd, rel_tol=1e-6):
            problems.append(f"{label}: {metric} の標準偏差が一致しません。")
    return problems


def _self_check(
    cells: tuple[ReferenceCell, ...],
    *,
    group_means_path: Path,
    measurement_stats_path: Path,
) -> list[str]:
    """生成したセルから 1 軸集計・全件集計を再構成し、notebook の出力と突き合わせる。"""
    problems: list[str] = []

    for row in _read_group_means(group_means_path):
        dimension = row["dimension"]
        group = row["group"]
        if dimension == "gender":
            query = ReferenceFilter(genders=frozenset({group}))
        elif dimension == "race":
            query = ReferenceFilter(races=frozenset({group}))
        elif dimension == "age_band":
            query = ReferenceFilter(age_bands=frozenset({group}))
        else:
            continue
        matched = select_cells(cells, query)
        aggregate = combine(matched, min_count=1)
        problems.extend(_compare_aggregate(f"{dimension}={group}", aggregate, row))

    full_aggregate = combine(cells, min_count=1)
    measurement_rows = _read_measurement_stats(measurement_stats_path)
    for name in METRIC_NAMES:
        row = measurement_rows.get(name)
        if row is None:
            problems.append(f"measurement_stats.csv に {name} がありません。")
            continue
        problems.extend(
            _compare_single_metric(f"overall/{name}", full_aggregate, metric=name, row=row)
        )

    return problems


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--group-means", type=Path, default=DEFAULT_GROUP_MEANS_PATH)
    parser.add_argument("--measurement-stats", type=Path, default=DEFAULT_MEASUREMENT_STATS_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    for label, path in (
        ("records", args.records),
        ("group-means", args.group_means),
        ("measurement-stats", args.measurement_stats),
        ("summary", args.summary),
    ):
        if not path.is_file():
            return _fail(f"入力ファイルが見つかりません（--{label}）: {path}")

    try:
        cells = _aggregate_cells(args.records)
    except _BuildError as error:
        return _fail(str(error))

    cells_bytes = _render_cells_csv(cells)
    records_sha256 = hashlib.sha256(args.records.read_bytes()).hexdigest()
    cells_sha256 = hashlib.sha256(cells_bytes).hexdigest()

    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return _fail(f"summary.json を解析できません: {error}")
    if not isinstance(summary, dict):
        return _fail(f"summary.json の形式が不正です: {args.summary}")

    try:
        metadata = _build_metadata(
            cells=cells,
            records_sha256=records_sha256,
            cells_sha256=cells_sha256,
            summary=summary,
        )
    except _BuildError as error:
        return _fail(str(error))

    try:
        problems = _self_check(
            cells,
            group_means_path=args.group_means,
            measurement_stats_path=args.measurement_stats,
        )
    except (KeyError, ValueError) as error:
        return _fail(f"自己検証の入力が不正です: {error}")

    if problems:
        print("自己検証に失敗しました。生成物は書き出しません:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    cells_path = args.output_dir / DEFAULT_REFERENCE_CELLS_PATH.name
    meta_path = cells_path.with_suffix(".meta.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells_path.write_bytes(cells_bytes)
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {cells_path} ({len(cells)} cells, sha256={cells_sha256})")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
