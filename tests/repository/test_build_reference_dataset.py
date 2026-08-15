"""§9.5 / §10.9-3: 個票 → セルテーブルの生成スクリプト（dev 用）。

アプリの実行経路からは呼ばれないが、**成果物がリポジトリへ入る**ので、
「同じ入力から常に同じ sha256 が出ること」と「集計が合わなければ止まること」を
test で押さえる。合わない生成物がコミットされる経路を塞ぐのがここの役目である。
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.build_reference_dataset import main

_AGE_BANDS = {
    "18-24": (18, 24),
    "25-34": (25, 34),
    "35-44": (35, 44),
    "45-54": (45, 54),
    "55-64": (55, 64),
    "65-75": (65, 75),
}

_METRICS = (
    "score_points",
    "s3d_distance",
    "symmetry_percent_contour",
    "symmetry_percent_eyes",
    "symmetry_percent_nose",
    "symmetry_percent_mouth",
    "symmetry_percent_overall",
)

#: 個票 fixture。`(race, gender, age, score_points)` を並べ、他の指標は
#: `score_points` から決定的に導く。件数の偏り（Asian/male が 3 件）を意図的に
#: 入れてあるので、件数で重み付けしない実装は自己検証で落ちる。
_RECORDS: tuple[tuple[str, str, int, float], ...] = (
    ("Asian", "male", 30, 40.0),
    ("Asian", "male", 30, 44.0),
    ("Asian", "male", 31, 48.0),
    ("Asian", "female", 30, 52.0),
    ("White", "male", 45, 56.0),
    ("White", "male", 45, 60.0),
    ("White", "female", 70, 64.0),
    ("Black", "female", 22, 68.0),
    ("Indian", "male", 22, 72.0),
    ("Others", "female", 60, 76.0),
)


def _metric_values(score: float) -> dict[str, float]:
    return {
        "score_points": score,
        "s3d_distance": 0.05 - score / 2500.0,
        "symmetry_percent_contour": 95.0 + score / 100.0,
        "symmetry_percent_eyes": 97.0 + score / 200.0,
        "symmetry_percent_nose": 97.0 + score / 300.0,
        "symmetry_percent_mouth": 96.0 + score / 400.0,
        "symmetry_percent_overall": 96.0 + score / 150.0,
    }


def _age_band(age: int) -> str:
    for name, (low, high) in _AGE_BANDS.items():
        if low <= age <= high:
            return name
    raise AssertionError(age)


def _write_records(path: Path) -> None:
    """実データと同じ列構成で書く。**`filename` と `image_id` も入れる。**

    出力へ漏れていないことを test で確かめるための罠であり、fixture が実データ
    より緩いせいで漏洩に気づけない状況を避ける。
    """
    header = ["image_id", "filename", "age", "gender", "race", "age_band", *_METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, (race, gender, age, score) in enumerate(_RECORDS, start=1):
            values = _metric_values(score)
            writer.writerow(
                [
                    index,
                    f"{age}_0_0_2017010320130800{index}.jpg.chip.jpg",
                    age,
                    gender,
                    race,
                    _age_band(age),
                    *[repr(values[name]) for name in _METRICS],
                ]
            )


def _subset(dimension: str, group: str) -> list[tuple[str, str, int, float]]:
    if dimension == "gender":
        return [row for row in _RECORDS if row[1] == group]
    if dimension == "race":
        return [row for row in _RECORDS if row[0] == group]
    if dimension == "age_band":
        return [row for row in _RECORDS if _age_band(row[2]) == group]
    raise AssertionError(dimension)


def _write_group_means(path: Path, *, corrupt: bool = False) -> None:
    header = [
        "dimension",
        "group",
        "count",
        "mean_score_points",
        "standard_deviation",
        *[f"mean_{name}" for name in _METRICS if name != "score_points"],
    ]
    rows: list[list[object]] = []
    for dimension, groups in (
        ("gender", sorted({row[1] for row in _RECORDS})),
        ("race", sorted({row[0] for row in _RECORDS})),
        ("age_band", sorted({_age_band(row[2]) for row in _RECORDS})),
    ):
        for group in groups:
            subset = _subset(dimension, group)
            scores = [row[3] for row in subset]
            row: list[object] = [
                dimension,
                group,
                len(subset),
                repr(statistics.fmean(scores)),
                repr(statistics.stdev(scores)) if len(scores) > 1 else "",
            ]
            for name in _METRICS:
                if name == "score_points":
                    continue
                values = [_metric_values(score)[name] for score in scores]
                row.append(repr(statistics.fmean(values)))
            rows.append(row)

    if corrupt:
        rows[0][3] = repr(float(rows[0][3]) + 5.0)  # type: ignore[arg-type]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_measurement_stats(path: Path) -> None:
    """実データと同じ `measurement` 名で全件統計を書く。"""
    alias = {"score_points": "overall_score_points", "s3d_distance": "distance_s3d_equal_weight"}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["measurement", "count", "mean", "standard_deviation"])
        for name in _METRICS:
            values = [_metric_values(row[3])[name] for row in _RECORDS]
            writer.writerow(
                [
                    alias.get(name, name),
                    float(len(values)),
                    repr(statistics.fmean(values)),
                    repr(statistics.stdev(values)),
                ]
            )


def _write_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "utkface-symmetry-3d-labelled/3.0.0",
                "generated_at": "2026-08-14T03:38:57.977344+00:00",
                "dataset": {
                    "name": "UTKFace",
                    "official_page": "https://susanqq.github.io/UTKFace/",
                    "license_scope": "non-commercial research only",
                },
                "protocol": {
                    "minimum_age": 18,
                    "maximum_age": 75,
                    "max_abs_yaw_deg": 15.0,
                    "max_abs_pitch_deg": 15.0,
                    "max_abs_roll_deg": 25.0,
                    "min_group_for_ci": 8,
                },
                "counts": {"valid_images": len(_RECORDS)},
                "reproducibility": {
                    "metric_version": "symmetry-3d/1.0.0",
                    "landmark_map_version": "mediapipe-478-mirror/1.0.0",
                },
                "warning": "探索値であり、正常値・目標値・個人評価・人物間比較には使用できない。",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path, *, corrupt_group_means: bool = False) -> list[str]:
    records = tmp_path / "records.csv"
    group_means = tmp_path / "group_means.csv"
    measurement_stats = tmp_path / "measurement_stats.csv"
    summary = tmp_path / "summary.json"
    _write_records(records)
    _write_group_means(group_means, corrupt=corrupt_group_means)
    _write_measurement_stats(measurement_stats)
    _write_summary(summary)
    return [
        "--records",
        str(records),
        "--group-means",
        str(group_means),
        "--measurement-stats",
        str(measurement_stats),
        "--summary",
        str(summary),
        "--output-dir",
        str(tmp_path / "out"),
    ]


def _cells_path(tmp_path: Path) -> Path:
    return tmp_path / "out" / "utkface_symmetry3d_cells_v1.csv"


def _meta_path(tmp_path: Path) -> Path:
    return tmp_path / "out" / "utkface_symmetry3d_cells_v1.meta.json"


def _read_cells(tmp_path: Path) -> list[dict[str, str]]:
    with _cells_path(tmp_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_builds_the_expected_cells(tmp_path: Path) -> None:
    """テスト 16a: 既知の個票から期待どおりのセルが出る。"""
    assert main(_inputs(tmp_path)) == 0

    rows = _read_cells(tmp_path)

    # 8 通りの (race, gender, age)。件数 0 のセルは出力しない。
    assert len(rows) == 8
    asian_male_30 = next(
        row for row in rows if (row["race"], row["gender"], row["age"]) == ("Asian", "male", "30")
    )
    assert asian_male_30["count"] == "2"
    assert float(asian_male_30["score_points_mean"]) == pytest.approx(42.0)
    # Σ(x − mean)² = (40−42)² + (44−42)² = 8
    assert float(asian_male_30["score_points_m2"]) == pytest.approx(8.0)


def test_cells_are_sorted_by_race_then_gender_then_age(tmp_path: Path) -> None:
    """§9.5: 行順を固定する。決定論性の前提。"""
    assert main(_inputs(tmp_path)) == 0

    rows = _read_cells(tmp_path)
    keys = [(row["race"], row["gender"], int(row["age"])) for row in rows]

    assert keys == sorted(keys)


def test_output_never_contains_per_image_identifiers(tmp_path: Path) -> None:
    """§10.9-2: `filename` / `image_id` を 1 文字も出力しない。"""
    assert main(_inputs(tmp_path)) == 0

    text = _cells_path(tmp_path).read_text(encoding="utf-8")

    assert "filename" not in text
    assert "image_id" not in text
    assert ".chip.jpg" not in text
    header = text.splitlines()[0].split(",")
    assert header[:4] == ["race", "gender", "age", "count"]
    assert set(header) == {"race", "gender", "age", "count"} | {
        f"{metric}_{suffix}" for metric in _METRICS for suffix in ("mean", "m2")
    }


def test_output_is_deterministic(tmp_path: Path) -> None:
    """テスト 16b: 2 回実行して同一 sha256。"""
    argv = _inputs(tmp_path)

    assert main(argv) == 0
    first = hashlib.sha256(_cells_path(tmp_path).read_bytes()).hexdigest()
    assert main(argv) == 0
    second = hashlib.sha256(_cells_path(tmp_path).read_bytes()).hexdigest()

    assert first == second


def test_meta_records_both_hashes_and_the_source_warning(tmp_path: Path) -> None:
    argv = _inputs(tmp_path)
    records = Path(argv[argv.index("--records") + 1])

    assert main(argv) == 0

    meta = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))

    assert meta["schema_version"] == "utkface-symmetry-3d-cells/1.0.0"
    assert meta["metric_version"] == "symmetry-3d/1.0.0"
    assert meta["source_schema_version"] == "utkface-symmetry-3d-labelled/3.0.0"
    assert meta["records_sha256"] == hashlib.sha256(records.read_bytes()).hexdigest()
    assert meta["cells_sha256"] == hashlib.sha256(_cells_path(tmp_path).read_bytes()).hexdigest(), (
        "meta の hash は自分が書いた CSV を指すこと"
    )
    assert meta["total_count"] == len(_RECORDS)
    assert meta["min_group_for_ci"] == 8
    assert meta["license_scope"] == "non-commercial research only"
    assert "個人評価" in meta["warning"], "summary.json の warning をそのまま引くこと"


def test_self_check_fails_on_a_mismatching_group_mean(tmp_path: Path) -> None:
    """テスト 17: 自己検証が壊れた集計を見つけて非ゼロ終了する。"""
    assert main(_inputs(tmp_path, corrupt_group_means=True)) != 0


def test_missing_records_file_fails_clearly(tmp_path: Path) -> None:
    """§10.9-3: 入力が無ければ明確に落ちる。暗黙にダウンロードも生成もしない。"""
    argv = _inputs(tmp_path)
    Path(argv[argv.index("--records") + 1]).unlink()

    assert main(argv) != 0


def test_rows_outside_the_published_age_range_are_rejected(tmp_path: Path) -> None:
    """18〜75 の外が個票に混じっていたら、黙って捨てず落ちる（不変条件 11）。"""
    argv = _inputs(tmp_path)
    records = Path(argv[argv.index("--records") + 1])
    text = records.read_text(encoding="utf-8")
    lines = text.splitlines()
    lines.append(lines[-1].replace(",30,", ",12,", 1))
    records.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert main(argv) != 0


def _corrupt_metric_column(tmp_path: Path, replacement: str) -> Sequence[str]:
    argv = _inputs(tmp_path)
    records = Path(argv[argv.index("--records") + 1])
    lines = records.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[6] = replacement
    lines[1] = ",".join(fields)
    records.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return argv


@pytest.mark.parametrize("replacement", ["nan", "inf", "対称"])
def test_non_finite_records_are_rejected(tmp_path: Path, replacement: str) -> None:
    argv = _corrupt_metric_column(tmp_path, replacement)

    assert main(list(argv)) != 0
