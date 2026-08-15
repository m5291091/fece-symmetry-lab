"""§9.5 / FR-037: 参照統計セルテーブルの純粋計算と immutable DTO。

`configs/reference/` に同梱する `(race, gender, age)` 粒度のセル集計を、絞り込み
（`select_cells`）と合併（`combine`）だけで扱う。個票（1 枚ごとの画像とスコア）は
一切保持しない——保持しているのは各セルの `count` と、7 指標それぞれの `mean` /
`m2`（二乗偏差和）だけである。

**`sum` と `sum_of_squares` ではなく `mean` と `m2` を持つ。** Chan らの並列分散
アルゴリズムに従って合併すると、任意のセル集合の和集合について平均と標本標準
偏差を厳密に再構成できる。`Σx²/n - mean²` 型の実装は、平均に対して分散が小さい
とき桁落ちして負の分散（`sqrt` で NaN）を返しうるため、意図的に避けている。

この module は streamlit / mediapipe / onnxruntime / cv2 / numpy のいずれも
import しない。標準ライブラリの `math` と `dataclasses` だけで足りる。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

AGE_BANDS: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {
        "18-24": (18, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55-64": (55, 64),
        "65-75": (65, 75),
    }
)

#: UTKFace の配布時注釈が使う人種ラベル（名前順）。データセットの注釈であり、
#: 本アプリは推定しない（不変条件 7）。
RACE_LABELS: Final[tuple[str, ...]] = ("Asian", "Black", "Indian", "Others", "White")

#: 同上、性別ラベル。
GENDER_LABELS: Final[tuple[str, ...]] = ("female", "male")

#: §9.5 のセルが持つ 7 指標。いずれも `symmetry-3d/1.0.0` に由来する。
METRIC_NAMES: Final[tuple[str, ...]] = (
    "score_points",
    "s3d_distance",
    "symmetry_percent_contour",
    "symmetry_percent_eyes",
    "symmetry_percent_nose",
    "symmetry_percent_mouth",
    "symmetry_percent_overall",
)

MIN_AGE: Final[int] = 18
MAX_AGE: Final[int] = 75

#: 元 notebook（`research/utkface-symmetry/`）の `min_group_for_ci` と揃えた値。
#: 合計件数がこれ未満のセル集合は、統計値として不安定なため統計値を出さない
#: （AC-REFERENCE-02）。ここを変える場合は notebook 側の定義も合わせて変更する
#: こと——2 か所が値だけずれると、`.meta.json` の記録と挙動が食い違う。
MIN_COUNT_FOR_STATISTICS: Final[int] = 8


class ReferenceStatsError(ValueError):
    """参照統計セルの検証、または絞り込み条件の解決に失敗した。"""


@dataclass(frozen=True, slots=True)
class MetricMoments:
    """1 セル・1 指標分の、合併可能な要約統計。

    `m2` は二乗偏差和 `Σ(x - mean)²` であり、分散でも標準偏差でもない
    （§9.5 の Chan の式がこの形を要求する）。
    """

    mean: float
    m2: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ReferenceStatsError(f"mean は有限値である必要があります: {self.mean!r}")
        if not math.isfinite(self.m2):
            raise ReferenceStatsError(f"m2 は有限値である必要があります: {self.m2!r}")
        if self.m2 < 0:
            raise ReferenceStatsError(f"m2 は 0 以上である必要があります: {self.m2!r}")


@dataclass(frozen=True, slots=True)
class ReferenceCell:
    """`(race, gender, age)` 粒度のセル 1 件（§9.5）。"""

    race: str
    gender: str
    age: int
    count: int
    moments: Mapping[str, MetricMoments]

    def __post_init__(self) -> None:
        if self.race not in RACE_LABELS:
            raise ReferenceStatsError(f"未知の race です: {self.race!r}")
        if self.gender not in GENDER_LABELS:
            raise ReferenceStatsError(f"未知の gender です: {self.gender!r}")
        if not MIN_AGE <= self.age <= MAX_AGE:
            raise ReferenceStatsError(
                f"age は {MIN_AGE}〜{MAX_AGE} の範囲である必要があります: {self.age!r}"
            )
        if self.count <= 0:
            raise ReferenceStatsError(f"count は正の値である必要があります: {self.count!r}")
        moment_names = set(self.moments)
        if moment_names != set(METRIC_NAMES):
            missing = sorted(set(METRIC_NAMES) - moment_names)
            extra = sorted(moment_names - set(METRIC_NAMES))
            raise ReferenceStatsError(
                "moments は METRIC_NAMES と過不足なく一致する必要があります"
                f"（不足: {missing}, 余剰: {extra}）。"
            )
        # frozen dataclass だが、渡された Mapping を凍結した写しへ差し替える。
        # 呼び出し側が持つ元の dict を後から変更しても、この DTO は影響を受けない。
        object.__setattr__(self, "moments", MappingProxyType(dict(self.moments)))


@dataclass(frozen=True, slots=True)
class ReferenceFilter:
    """絞り込み条件。空集合・`None` の軸は「絞り込まない」（全件が対象）。

    ラベルの妥当性はここでは検証しない。`races`/`genders` は `select_cells` が
    素通しの集合演算として扱うため未知ラベルは単に 0 件になり、`age_bands` だけ
    `AGE_BANDS` の key として解決する必要があるため `select_cells` が検証する。
    UI からの入力検証は `application.reference_browser.validate_filter` が行う。
    """

    races: frozenset[str] = frozenset()
    genders: frozenset[str] = frozenset()
    age_bands: frozenset[str] = frozenset()
    age_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """合併後の 1 指標分の要約。"""

    mean: float
    sd: float | None


@dataclass(frozen=True, slots=True)
class ReferenceAggregate:
    """`combine` の結果。しきい値未満のときは `metrics` が空になる。"""

    count: int
    metrics: Mapping[str, MetricSummary]

    @property
    def has_statistics(self) -> bool:
        return bool(self.metrics)


@dataclass(frozen=True, slots=True)
class ReferenceDatasetMetadata:
    """`.meta.json`（§9.5）の内容。"""

    schema_version: str
    metric_version: str
    landmark_map_version: str
    source_schema_version: str
    source_generated_at: str
    records_sha256: str
    cells_sha256: str
    total_count: int
    min_group_for_ci: int
    dataset_name: str
    dataset_url: str
    license_scope: str
    selection: Mapping[str, str]
    warning: str


@dataclass(frozen=True, slots=True)
class ReferenceDataset:
    """読み込み済みの参照統計データ 1 式。"""

    metadata: ReferenceDatasetMetadata
    cells: tuple[ReferenceCell, ...]


def select_cells(
    cells: Iterable[ReferenceCell],
    query: ReferenceFilter,
) -> tuple[ReferenceCell, ...]:
    """絞り込む。空集合・`None` の軸は絞り込まない（全件を通す）。

    同一軸の複数値は和集合（例: race = {Asian, Indian} はどちらか一方で可）、
    軸をまたぐと積集合（race かつ gender かつ age）になる。`age_bands` と
    `age_range` が両方指定された場合も同様に積集合——両方を満たすセルだけ残す。

    入力の順序を保つ（`cells` に現れた順で結果を返す）。
    """
    age_band_ranges: tuple[tuple[int, int], ...] | None = None
    if query.age_bands:
        try:
            age_band_ranges = tuple(AGE_BANDS[name] for name in query.age_bands)
        except KeyError as error:
            raise ReferenceStatsError(f"未知の年齢帯です: {error.args[0]!r}") from error

    selected: list[ReferenceCell] = []
    for cell in cells:
        if query.races and cell.race not in query.races:
            continue
        if query.genders and cell.gender not in query.genders:
            continue
        if age_band_ranges is not None and not any(
            low <= cell.age <= high for low, high in age_band_ranges
        ):
            continue
        if query.age_range is not None:
            low, high = query.age_range
            if not low <= cell.age <= high:
                continue
        selected.append(cell)
    return tuple(selected)


def combine(
    cells: Iterable[ReferenceCell],
    *,
    min_count: int = MIN_COUNT_FOR_STATISTICS,
) -> ReferenceAggregate | None:
    """一致したセルを合併し、件数・平均・標本標準偏差を得る。

    * 0 件なら `None`（「一致なし」と「一致したが統計を出さない」を区別する）。
    * 合計 `count` が `min_count` 未満なら、件数だけを持つ集計を返す
      （`metrics` は空、`has_statistics` は `False`）。
    * それ以外は §9.5 の Chan の式で `mean`/`m2` を合併し、
      `sd = sqrt(m2 / (n - 1))` を返す。`n < 2` では `sd` を欠測（`None`）にする
      ——0.0 で埋めない（不変条件 11）。

    **`sum_of_squares` を経由しない。** `mean`/`m2` のペアだけを使って合併するため、
    平均が大きく分散が小さい入力でも桁落ちしない（テストが 1e8 オフセットで検証）。
    """
    cell_tuple = tuple(cells)
    if not cell_tuple:
        return None

    total_count = sum(cell.count for cell in cell_tuple)
    if total_count < min_count:
        return ReferenceAggregate(count=total_count, metrics=MappingProxyType({}))

    metrics: dict[str, MetricSummary] = {}
    for name in METRIC_NAMES:
        merged_n = 0
        merged_mean = 0.0
        merged_m2 = 0.0
        for cell in cell_tuple:
            moment = cell.moments[name]
            if merged_n == 0:
                merged_n = cell.count
                merged_mean = moment.mean
                merged_m2 = moment.m2
                continue
            delta = moment.mean - merged_mean
            next_n = merged_n + cell.count
            merged_mean = merged_mean + delta * cell.count / next_n
            merged_m2 = merged_m2 + moment.m2 + delta * delta * merged_n * cell.count / next_n
            merged_n = next_n
        sd = math.sqrt(merged_m2 / (merged_n - 1)) if merged_n >= 2 else None
        metrics[name] = MetricSummary(mean=merged_mean, sd=sd)

    return ReferenceAggregate(count=total_count, metrics=MappingProxyType(metrics))
