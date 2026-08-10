#!/usr/bin/env python3
"""1 本のテイクの中での姿勢応答を実測する（docs §15）。

`docs/symmetry-metric-measurements.md` §14.5 は、3D の表示値中央値で外れているのが
take03 だけであり、同じ向きに外れている共変量が **pitch と顔領域の黒つぶれ率の
2 つしかない**ことを記録した。ただしテイクは 3 本しかなく、そこでは両者を分け
られない。

本スクリプトは **1 本のテイクの中で** 姿勢と表示値の関係を測る。テイクを跨がない
ので、照明・顔の大きさ・その日の顔はすべて定数になり、姿勢だけが動く。

    T: within-take の応答  d_3d と現行 2D を pitch / yaw / roll へ回帰する（全フレーム）
    Y: yaw を層別した応答  セクション H と同じ帯で yaw を止め、pitch だけを残す
    X: 検算                T の感度で、テイク間の表示値差を説明できるか

§4 とセクション H は |yaw| だけを調べており、pitch と d_3d の関係は一度も測って
いない。本スクリプトはそこだけを埋める。

    uv run python scripts/measure_pose_response.py --self-test
    uv run python scripts/measure_pose_response.py --video take01.mov \
        --video take02.mov --video take03.mov --max-frames 600

検出とフレーム集合は `measure_session_variation.py` の `capture_take` をそのまま
呼ぶので、§14 と同一である。指標・正規化・表示式・yaw の層別は
`measure_metric_noise.py`（= `domain` の実装）から import している。**その 2 つの
スクリプトへは一切触れない。** 本スクリプトが持つ独自の計算は、最小二乗と相関と
自己相関の補正だけである。

`src/` は読むだけで一切変更しない。画像と landmark 座標は JSON へ書き出さず、
統計量だけを残す（CLAUDE.md §5 のプライバシー不変条件）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.measure_metric_noise import (  # noqa: E402
    CONFIG,
    CONFIG_3D,
    IPD_MM,
    MIN_USEFUL_FRAMES,
    YAW_BAND_DEGREES,
    Series,
    _correlation,
    heading,
    percent_values,
    resolve_model,
    stable_yaw_mask,
    yaw_values,
)
from scripts.measure_session_variation import (  # noqa: E402
    DEFAULT_FPS,
    DEFAULT_MAX_FRAMES,
    METRIC_LABELS,
    MIN_TAKES,
    Take,
    capture_take,
    column_stats,
    spread_across_takes,
)

#: 95% 区間に使う正規近似の係数。枚数が数百あるので t 分布との差は無視できる。
NORMAL_95: float = 1.959964

#: 回帰の傾きを出す最小の枚数。相関だけなら 3 枚で出るが、残差の標準誤差には
#: n - 2 の自由度が要る。
MIN_FIT_FRAMES: int = 10

JSON_SCHEMA: str = "measure-pose-response/1.0.0"

SECTIONS: dict[str, str] = {
    "T": "within-take の姿勢応答（全フレーム）",
    "Y": "yaw を層別したときの pitch 応答",
    "X": "検算 — within-take の感度でテイク間差を説明できるか",
}

#: 測る姿勢の軸。pitch が主題で、yaw は既知の対照、roll は除外の根拠に使う。
AXES: tuple[str, ...] = ("pitch", "yaw", "roll")


# --------------------------------------------------------------------------
# 系列の取り出し
# --------------------------------------------------------------------------
def pose_series(series: Series, axis: str) -> NDArray[np.float64]:
    """フレームごとの姿勢角（度）。姿勢を取れなかったフレームは NaN。

    `measure_metric_noise.yaw_values` を 3 軸へ一般化しただけである。両者が yaw で
    一致することは self-test が毎回確認する。
    """
    field = f"{axis}_deg"
    return np.asarray(
        [float("nan") if pose is None else float(getattr(pose, field)) for pose in series.poses],
        dtype=np.float64,
    )


def display_series(series: Series, metric: str) -> NDArray[np.float64]:
    """フレームごとの表示値（pt）。§8.6.3 の式そのもので、添字は `poses` と揃う。

    `measure_session_variation._display_values` は 3D 側で欠測を先に落とすが、
    ここでは姿勢との添字対応を保つために NaN のまま残す。落としてから読み替えても
    読み替えてから落としても、有限な要素の集合は同じである。
    """
    distances = series.distances if metric == "current_2d" else series.distances3d
    return percent_values(distances)


def adjacent_mask(series: Series, delta_seconds: float) -> NDArray[np.bool_]:
    """隣り合う 2 枚が撮影間隔ちょうどで並んでいる組（第 1 階差に使える組）。"""
    if series.count < 2:
        return np.zeros(0, dtype=np.bool_)
    steps = np.diff(series.times)
    return np.asarray(np.abs(steps - delta_seconds) <= 0.5 * delta_seconds, dtype=np.bool_)


# --------------------------------------------------------------------------
# 直線あてはめ
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Fit:
    """1 本の最小二乗あてはめ。`slope` の単位は「y の単位 ÷ x の単位」である。"""

    count: int
    x_std: float
    y_std: float
    correlation: float | None
    slope: float | None
    intercept: float | None
    #: 傾きの標準誤差。独立標本を仮定した素の値。
    stderr: float | None
    #: 自己相関で補正した実効枚数と、それに対応する標準誤差。
    effective_count: float | None
    stderr_effective: float | None

    @property
    def half_width(self) -> float | None:
        """95% 区間の半幅。補正後の標準誤差を使う。"""
        if self.stderr_effective is None:
            return None
        return NORMAL_95 * self.stderr_effective

    @property
    def separated_from_zero(self) -> bool | None:
        """95% 区間が 0 を跨がないか。跨ぐなら「感度は出ていない」と読む。"""
        half = self.half_width
        if self.slope is None or half is None:
            return None
        return abs(self.slope) > half


def _lag_one(values: NDArray[np.float64]) -> float | None:
    """ラグ 1 の自己相関。定数列や 3 点未満では求めない。"""
    if values.size < 3:
        return None
    centred = values - values.mean()
    denominator = float((centred**2).sum())
    if denominator <= 0.0:
        return None
    return float((centred[:-1] * centred[1:]).sum() / denominator)


def _effective_count(count: int, x_lag: float | None, residual_lag: float | None) -> float:
    """自己相関を考慮した実効枚数（Bartlett / Quenouille の近似）。

        n_eff = n × (1 - r1x·r1e) / (1 + r1x·r1e)

    30fps の隣接フレームは強く相関しているので、素の n をそのまま使うと傾きの
    区間が実際よりずっと狭く出る。**この補正も近似であって、区間の厳密な被覆率を
    主張するものではない。** 相関の積が負になる場合は補正しない（区間を素の値
    より狭めない）。
    """
    if x_lag is None or residual_lag is None:
        return float(count)
    product = x_lag * residual_lag
    if product <= 0.0:
        return float(count)
    reduced = count * (1.0 - product) / (1.0 + product)
    return float(min(float(count), max(3.0, reduced)))


def fit_line(x: NDArray[np.float64], y: NDArray[np.float64]) -> Fit | None:
    """y = a + b·x の最小二乗。欠測は組ごと落とす。x が定数なら傾きを出さない。"""
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    xs, ys = left[keep], right[keep]
    count = int(xs.size)
    if count < 3:
        return None
    x_std = float(xs.std(ddof=1))
    y_std = float(ys.std(ddof=1))
    correlation = _correlation(xs, ys)
    if x_std <= 0.0 or count < MIN_FIT_FRAMES:
        return Fit(count, x_std, y_std, correlation, None, None, None, None, None)

    centred = xs - xs.mean()
    scatter = float((centred**2).sum())
    slope = float((centred * (ys - ys.mean())).sum() / scatter)
    intercept = float(ys.mean() - slope * xs.mean())
    residual = ys - (intercept + slope * xs)
    variance = float((residual**2).sum()) / (count - 2)
    stderr = sqrt(variance / scatter) if variance > 0.0 else 0.0
    effective = _effective_count(count, _lag_one(xs), _lag_one(residual))
    stderr_effective = stderr * sqrt(count / effective) if effective > 0.0 else None
    return Fit(
        count=count,
        x_std=x_std,
        y_std=y_std,
        correlation=correlation,
        slope=slope,
        intercept=intercept,
        stderr=stderr,
        effective_count=effective,
        stderr_effective=stderr_effective,
    )


def fit_json(fit: Fit | None) -> dict[str, Any] | None:
    if fit is None:
        return None
    return {
        "frames": fit.count,
        "x_std": fit.x_std,
        "y_std": fit.y_std,
        "correlation": fit.correlation,
        "slope_pt_per_deg": fit.slope,
        "intercept_pt": fit.intercept,
        "stderr_naive": fit.stderr,
        "effective_frames": fit.effective_count,
        "stderr_effective": fit.stderr_effective,
        "half_width_95": fit.half_width,
        "separated_from_zero": fit.separated_from_zero,
    }


# --------------------------------------------------------------------------
# 出力の体裁
# --------------------------------------------------------------------------
def _number(value: float | None, digits: int) -> str:
    return "—" if value is None or not isfinite(value) else f"{value:.{digits}f}"


def _signed(value: float | None, digits: int) -> str:
    return "—" if value is None or not isfinite(value) else f"{value:+.{digits}f}"


def _interval(fit: Fit | None) -> str:
    """傾きの 95% 区間。0 を跨ぐかどうかがそのまま読めるように併記する。"""
    if fit is None or fit.slope is None or fit.half_width is None:
        return "—"
    low = fit.slope - fit.half_width
    high = fit.slope + fit.half_width
    mark = "" if fit.separated_from_zero else " *0"
    return f"[{low:+.4f}, {high:+.4f}]{mark}"


def _median_of(values: NDArray[np.float64]) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.median(finite))


# --------------------------------------------------------------------------
# セクション T — within-take の姿勢応答
# --------------------------------------------------------------------------
RESPONSE_NOTE = """
  何を測るか

  1 本のテイクの中だけで、フレームごとの表示値を姿勢角へ回帰する。

      表示値(pt) = a + b × 姿勢角(度)

  b が「感度」であり、単位は pt / 度である。1pt = 0.01 IOD なので、この量は IPD の
  仮定に依存しない（§12.0）。**符号付きの角度と |角度| の両方で出す。**
  §8.5.1 の剛体不変性は数学的に厳密なので、真の landmark が剛体で回っただけなら
  d_3d は動かない。符号付きで単調に動くなら不変性の側、|角度| で動くなら検出器の
  誤差の側を疑う材料になる。

  現行 2D は姿勢依存が強いと分かっている（§4 で |yaw| 相関 +0.854）。**2D で明確な
  応答が出ることが「この測定が効いている」ことの確認**になる。2D でも出なければ、
  そのテイクの姿勢の変動幅が足りないということである。

  第 1 階差（隣り合うフレームの差どうしの回帰）も併記する。20 秒のテイクでは表示値
  も姿勢もゆっくり漂うので、水準どうしの相関は「たまたま同じ向きに漂った」だけでも
  立つ。階差を取ると漂いが消えるので、瞬間的な結び付きだけが残る。**ただし階差は
  ノイズを増幅するので、傾きは 0 の側へ引っ張られる。小さい値を「無い」と読まない。**
"""


def _pose_profile(takes: list[Take], report: dict[str, Any]) -> None:
    """1) 姿勢がどれだけ動いたか。感度を測れる前提があるかを先に見る。"""
    print("  1) テイク内で姿勢がどれだけ動いたか（感度を測れる前提）")
    print(
        f"     {'テイク':<10}{'軸':<8}{'枚数':>6}{'中央値':>10}{'標準偏差':>10}"
        f"{'p05':>10}{'p95':>10}{'p95-p05':>10}{'最小':>10}{'最大':>10}"
    )
    print("     " + "-" * 96)
    rows: list[dict[str, Any]] = []
    for take in takes:
        for axis in AXES:
            values = pose_series(take.series, axis)
            stats = column_stats([float(value) for value in values])
            if stats is None:
                print(f"     {take.name:<10}{axis:<8}{'—':>6}")
                continue
            print(
                f"     {take.name:<10}{axis:<8}{int(stats['count']):6d}"
                f"{stats['median']:10.3f}{stats['std']:10.3f}"
                f"{stats['p05']:10.3f}{stats['p95']:10.3f}"
                f"{stats['p95'] - stats['p05']:10.3f}"
                f"{stats['min']:10.3f}{stats['max']:10.3f}"
            )
            rows.append({"take": take.name, "axis": axis, **stats})
        print("     " + "-" * 96)
    report["pose_profile"] = rows
    print()

    print("  2) 軸どうしの相関（pitch を単独で読めるか）")
    print(f"     {'テイク':<10}{'pitch-yaw':>12}{'pitch-roll':>12}{'yaw-roll':>12}")
    print("     " + "-" * 48)
    pairs: list[dict[str, Any]] = []
    for take in takes:
        angles = {axis: pose_series(take.series, axis) for axis in AXES}
        entry: dict[str, Any] = {"take": take.name}
        cells: list[str] = []
        for first, second in (("pitch", "yaw"), ("pitch", "roll"), ("yaw", "roll")):
            keep = np.isfinite(angles[first]) & np.isfinite(angles[second])
            value = _correlation(angles[first][keep], angles[second][keep])
            entry[f"{first}_{second}"] = value
            cells.append("—" if value is None else f"{value:+.3f}")
        print(f"     {take.name:<10}" + "".join(f"{cell:>12}" for cell in cells))
        pairs.append(entry)
    report["axis_correlation"] = pairs
    print()
    print("     pitch と yaw が強く相関しているテイクでは、水準の回帰で 2 つを分け")
    print("     られない。そのためにセクション Y で yaw を止める。")
    print()


def _response_rows(take: Take, metric: str) -> list[dict[str, Any]]:
    """1 テイク × 1 指標ぶんの応答。軸ごとに 1 行。"""
    display = display_series(take.series, metric)
    steps = adjacent_mask(take.series, take.delta_seconds)
    rows: list[dict[str, Any]] = []
    for axis in AXES:
        angles = pose_series(take.series, axis)
        signed = fit_line(angles, display)
        absolute = fit_line(np.abs(angles), display)
        if steps.size:
            difference = fit_line(np.diff(angles)[steps], np.diff(display)[steps])
        else:
            difference = None
        rows.append(
            {
                "take": take.name,
                "metric": metric,
                "axis": axis,
                "signed": signed,
                "absolute": absolute,
                "difference": difference,
            }
        )
    return rows


def _print_response_table(rows: list[dict[str, Any]]) -> None:
    print(
        f"     {'テイク':<13}{'軸':<7}{'枚数':>6}{'角度std':>9}"
        f"{'r(符号付)':>11}{'r(|角|)':>10}{'傾き pt/度':>12}"
        f"{'95%区間 pt/度':>24}{'n_eff':>8}{'r(Δ)':>9}{'傾きΔ':>10}"
    )
    print("     " + "-" * 119)
    for row in rows:
        signed: Fit | None = row["signed"]
        absolute: Fit | None = row["absolute"]
        difference: Fit | None = row["difference"]
        print(
            f"     {row.get('label', row['take']):<13}{row['axis']:<7}"
            f"{(0 if signed is None else signed.count):6d}"
            f"{_number(None if signed is None else signed.x_std, 3):>9}"
            f"{_signed(None if signed is None else signed.correlation, 3):>11}"
            f"{_signed(None if absolute is None else absolute.correlation, 3):>10}"
            f"{_signed(None if signed is None else signed.slope, 4):>12}"
            f"{_interval(signed):>24}"
            f"{_number(None if signed is None else signed.effective_count, 1):>8}"
            f"{_signed(None if difference is None else difference.correlation, 3):>9}"
            f"{_signed(None if difference is None else difference.slope, 4):>10}"
        )


def section_t(takes: list[Take], report: dict[str, Any]) -> None:
    heading("T", "within-take の姿勢応答 — 1 本の中で姿勢が動いたとき表示値は動くか")
    print(RESPONSE_NOTE)

    _pose_profile(takes, report)

    print("  3) 表示値の姿勢応答（水準の回帰。傾きは符号付きの角度に対するもの）")
    print()
    collected: dict[str, list[dict[str, Any]]] = {}
    for metric, label in METRIC_LABELS.items():
        print(f"     指標: {label}")
        rows = [row for take in takes for row in _response_rows(take, metric)]
        _print_response_table(rows)
        collected[metric] = rows
        print()
    print("     95%区間 は自己相関で補正した標準誤差 × 1.96。末尾の *0 は区間が 0 を")
    print("     跨いでいる印で、その行では「感度は出ていない」と読む。")
    print("     n_eff は自己相関を考慮した実効枚数。30fps の隣接フレームは強く相関する。")
    print("     r(Δ) / 傾きΔ は隣り合うフレームの差どうしの回帰である。")
    print()

    report["within_take_response"] = {
        "section": "T",
        "takes": [take.name for take in takes],
        "rows": [
            {
                "take": row["take"],
                "metric": row["metric"],
                "axis": row["axis"],
                "signed": fit_json(row["signed"]),
                "absolute": fit_json(row["absolute"]),
                "difference": fit_json(row["difference"]),
            }
            for metric in METRIC_LABELS
            for row in collected[metric]
        ],
    }
    report["_fits"] = collected


# --------------------------------------------------------------------------
# セクション Y — yaw を層別してからの pitch 応答
# --------------------------------------------------------------------------
STRATIFY_NOTE = """
  何を測るか

  take01 は yaw 標準偏差 2.210° と pitch 標準偏差 1.864° が同時に動く（§14.1）。
  水準の回帰だけでは、pitch に付いた傾きが yaw の影響を拾っている可能性を排除
  できない。セクション H の `stable_yaw_mask` と同じ帯（±{bands}°、
  {frames} 枚に届いた時点で確定）で yaw を止めてから、pitch の相関と傾きを出し直す。

  take02 / take03 は yaw 標準偏差が 0.33 / 0.23° で元から静かなので、層別しても
  ほとんど枚数が減らない。**この 2 本は対照である。**

  帯を絞ると pitch の変動幅まで一緒に痩せることがある。痩せた場合は感度が出なく
  なるのが当然なので、層別後の pitch 標準偏差を必ず並べて読むこと。
"""


def section_y(takes: list[Take], report: dict[str, Any]) -> None:
    heading("Y", "yaw を層別したときの pitch 応答 — 2 つの軸を分ける")
    print(
        STRATIFY_NOTE.format(
            bands="/".join(f"{band:.1f}" for band in YAW_BAND_DEGREES),
            frames=MIN_USEFUL_FRAMES,
        )
    )

    print("  1) 層別の効き（帯・枚数・残った pitch の変動幅）")
    print(
        f"     {'テイク':<10}{'帯':>10}{'yaw中央値':>12}{'枚数':>8}{'元の枚数':>10}"
        f"{'pitch std(層別)':>17}{'pitch std(全体)':>17}"
    )
    print("     " + "-" * 86)
    selections: dict[str, tuple[NDArray[np.bool_], float, float] | None] = {}
    strata: list[dict[str, Any]] = []
    for take in takes:
        selection = stable_yaw_mask(take.series)
        selections[take.name] = selection
        if selection is None:
            print(f"     {take.name:<10}{'—':>10}")
            continue
        mask, band, median = selection
        pitch = pose_series(take.series, "pitch")
        inside = pitch[mask & np.isfinite(pitch)]
        whole = pitch[np.isfinite(pitch)]
        inside_std = float(inside.std(ddof=1)) if inside.size > 1 else None
        whole_std = float(whole.std(ddof=1)) if whole.size > 1 else None
        print(
            f"     {take.name:<10}{f'±{band:.1f}°':>10}{median:12.3f}"
            f"{int(mask.sum()):8d}{take.series.count:10d}"
            f"{_number(inside_std, 3):>17}{_number(whole_std, 3):>17}"
        )
        strata.append(
            {
                "take": take.name,
                "band_deg": band,
                "median_yaw_deg": median,
                "frames": int(mask.sum()),
                "frames_total": take.series.count,
                "pitch_std_subset": inside_std,
                "pitch_std_full": whole_std,
            }
        )
    print()

    print("  2) 層別後の pitch 応答")
    print()
    stratified: dict[str, list[dict[str, Any]]] = {}
    for metric, label in METRIC_LABELS.items():
        print(f"     指標: {label}")
        rows: list[dict[str, Any]] = []
        for take in takes:
            selection = selections[take.name]
            if selection is None:
                continue
            mask, band, _median = selection
            display = display_series(take.series, metric)
            for axis in ("pitch", "roll"):
                angles = pose_series(take.series, axis)
                blocked = np.where(mask, angles, np.nan)
                rows.append(
                    {
                        "take": take.name,
                        "label": f"{take.name} ±{band:.1f}",
                        "axis": axis,
                        "signed": fit_line(blocked, display),
                        "absolute": fit_line(np.abs(blocked), display),
                        "difference": None,
                    }
                )
        _print_response_table(rows)
        stratified[metric] = rows
        print()
    print("     層別後は隣り合うフレームが飛び飛びになるので、第 1 階差は出していない。")
    print("     n_eff の補正も、連続していない部分列に対する近似である。")
    print()

    report["stratified_response"] = {
        "section": "Y",
        "yaw_bands_deg": list(YAW_BAND_DEGREES),
        "min_useful_frames": MIN_USEFUL_FRAMES,
        "strata": strata,
        "rows": [
            {
                "take": row["take"],
                "label": row["label"],
                "metric": metric,
                "axis": row["axis"],
                "signed": fit_json(row["signed"]),
                "absolute": fit_json(row["absolute"]),
            }
            for metric in METRIC_LABELS
            for row in stratified[metric]
        ],
    }
    report["_stratified_fits"] = stratified


# --------------------------------------------------------------------------
# セクション X — 検算
# --------------------------------------------------------------------------
CHECK_NOTE = """
  何を検算するか

  本作業の要は次の 1 行である。

      within-take で測った感度（pt/度） × テイク間の姿勢差（度）
          = §14.4 のセッション間全幅（pt）を説明できるか

  近ければ、テイク間で表示値が動いた分は姿勢で説明が付く。桁が違えば、姿勢は
  主犯ではなく、別の要因（照明・黒つぶれ・未測定の何か）が残っていることになる。

  **この節は数値と、数値から直接言えることまでしか書かない。** どの閾値をどう
  締めるかの判断はここでは出さない。
"""


def _take_medians(takes: list[Take]) -> dict[str, dict[str, float | None]]:
    """テイクごとの中央値。姿勢は `Series` 由来で、回帰に使ったものと同じ母数である。"""
    medians: dict[str, dict[str, float | None]] = {}
    for take in takes:
        entry: dict[str, float | None] = {}
        for axis in AXES:
            angles = pose_series(take.series, axis)
            entry[axis] = _median_of(angles)
            entry[f"abs_{axis}"] = _median_of(np.abs(angles))
        for metric in METRIC_LABELS:
            entry[metric] = _median_of(display_series(take.series, metric))
        medians[take.name] = entry
    return medians


def _verify_against_profiles(
    takes: list[Take], medians: dict[str, dict[str, float | None]]
) -> None:
    """0) 姿勢の中央値が §14.1 の母数（単一顔フレーム）と一致するかを確かめる。"""
    print("  0) 測定系の確認（§14 と同じ数字が出ているか）")
    print(
        f"     {'テイク':<10}{'pitch(本節)':>14}{'pitch(§14.1 母数)':>20}"
        f"{'表示値 3D':>12}{'表示値 2D':>12}"
    )
    print("     " + "-" * 70)
    for take in takes:
        single = [item.pitch_deg for item in take.profiles if item.single_face]
        stats = column_stats(single)
        print(
            f"     {take.name:<10}{_number(medians[take.name]['pitch'], 3):>14}"
            f"{_number(None if stats is None else stats['median'], 3):>20}"
            f"{_number(medians[take.name]['symmetry_3d'], 3):>12}"
            f"{_number(medians[take.name]['current_2d'], 3):>12}"
        )
    print()
    print("     左 2 列が一致すれば、本節の姿勢は §14.1 と同じ母数である。")
    print("     表示値の 2 列は §14.3 の中央値と一致していなければならない。")
    print()


def _slope_of(rows: list[dict[str, Any]], take: str, axis: str) -> float | None:
    for row in rows:
        if row["take"] == take and row["axis"] == axis:
            fit: Fit | None = row["signed"]
            return None if fit is None else fit.slope
    return None


def section_x(takes: list[Take], report: dict[str, Any]) -> None:
    heading("X", "検算 — within-take の感度で、テイク間の表示値差を説明できるか")
    print(CHECK_NOTE)

    fits: dict[str, list[dict[str, Any]]] | None = report.get("_fits")
    if fits is None:
        print("  セクション T を実行していないので、検算できません。")
        return
    if len(takes) < MIN_TAKES:
        print(f"  テイクが {len(takes)} 本しかないので、テイク間の差を出せません。")
        return

    names = [take.name for take in takes]
    medians = _take_medians(takes)
    _verify_against_profiles(takes, medians)

    print("  1) テイク間で実際に動いた量")
    print(
        f"     {'量':<16}{'単位':<8}" + "".join(f"{name:>12}" for name in names) + f"{'全幅':>12}"
    )
    print("     " + "-" * (24 + 12 * len(names) + 12))
    spans: dict[str, float | None] = {}
    for key, label, unit in (
        ("pitch", "pitch", "度"),
        ("abs_pitch", "|pitch|", "度"),
        ("yaw", "yaw", "度"),
        ("roll", "roll", "度"),
        ("symmetry_3d", "表示値（3D）", "pt"),
        ("current_2d", "表示値（現行 2D）", "pt"),
    ):
        column = [medians[name][key] for name in names]
        across = spread_across_takes(column)
        span = None if across is None else across["span"]
        spans[key] = span
        print(
            f"     {label:<16}{unit:<8}"
            + "".join(f"{_number(value, 3):>12}" for value in column)
            + f"{_number(span, 3):>12}"
        )
    print()

    stratified: dict[str, list[dict[str, Any]]] = report.get("_stratified_fits") or {}
    populations: list[tuple[str, dict[str, list[dict[str, Any]]], tuple[str, ...]]] = [
        ("全フレーム", fits, AXES),
    ]
    if stratified:
        populations.append(("yaw層別", stratified, ("pitch", "roll")))

    print("  2) 検算 — 感度 × テイク間の姿勢差 は、表示値の全幅に届くか")
    print(
        f"     {'指標':<8}{'母集団':<11}{'軸':<7}{'傾き中央値':>12}{'傾きの範囲':>20}"
        f"{'姿勢の全幅':>12}{'予測 pt':>10}{'予測の範囲':>18}{'実測 pt':>10}{'予測÷実測':>12}"
    )
    print("     " + "-" * 116)
    verdicts: list[dict[str, Any]] = []
    for metric, label in METRIC_LABELS.items():
        observed = spans[metric]
        for population, source, axes in populations:
            for axis in axes:
                slopes = [_slope_of(source[metric], take.name, axis) for take in takes]
                usable = [value for value in slopes if value is not None and isfinite(value)]
                middle = float(np.median(usable)) if usable else None
                low = min(abs(value) for value in usable) if usable else None
                high = max(abs(value) for value in usable) if usable else None
                angle_span = spans[axis]
                predicted = (
                    None if middle is None or angle_span is None else abs(middle) * abs(angle_span)
                )
                bounds = (
                    None
                    if low is None or high is None or angle_span is None
                    else (low * abs(angle_span), high * abs(angle_span))
                )
                ratio = (
                    None
                    if predicted is None or observed is None or abs(observed) < 1e-12
                    else predicted / observed
                )
                print(
                    f"     {label:<8}{population:<11}{axis:<7}{_signed(middle, 4):>12}"
                    f"{('—' if low is None else f'{low:.4f}〜{high:.4f}'):>20}"
                    f"{_number(angle_span, 3):>12}{_number(predicted, 3):>10}"
                    f"{('—' if bounds is None else f'{bounds[0]:.3f}〜{bounds[1]:.3f}'):>18}"
                    f"{_number(observed, 3):>10}{_number(ratio, 3):>12}"
                )
                verdicts.append(
                    {
                        "metric": metric,
                        "population": population,
                        "axis": axis,
                        "slopes_by_take": dict(zip(names, slopes, strict=True)),
                        "slope_median_pt_per_deg": middle,
                        "slope_abs_min_pt_per_deg": low,
                        "slope_abs_max_pt_per_deg": high,
                        "pose_span_deg": angle_span,
                        "predicted_span_pt": predicted,
                        "predicted_span_bounds_pt": None if bounds is None else list(bounds),
                        "observed_span_pt": observed,
                        "ratio": ratio,
                    }
                )
        print("     " + "-" * 116)
    print()
    print("     予測 = |テイクごとの傾きの中央値| × |姿勢の全幅|。符号は無視して大きさだけ。")
    print("     「傾きの範囲」はテイクごとの傾きの絶対値の最小〜最大で、「予測の範囲」は")
    print("     それに姿勢の全幅を掛けたものである。**中央値 1 つを額面で読まないための欄。**")
    print("     yaw層別 の行は、yaw を止めた部分集合（セクション Y）の傾きを使っている。")
    print("     yaw 自身は層別で動かなくなるので、その母集団には yaw の行が無い。")
    print()

    print("  3) 符号まで合うか（pitch。傾きの中央値で各テイクを予測して残差を見る）")
    print(
        f"     {'指標':<8}{'母集団':<11}{'テイク':<10}{'pitch 中央値':>14}{'実測 pt':>10}"
        f"{'予測 pt':>10}{'残差 pt':>10}"
    )
    print("     " + "-" * 75)
    residual_rows: list[dict[str, Any]] = []
    for metric, label in METRIC_LABELS.items():
        for population, source, _axes in populations:
            _print_residuals(
                metric,
                label,
                population,
                source,
                takes,
                medians,
                residual_rows,
            )
        print("     " + "-" * 75)
    print()
    print("     残差が実測の全幅と同じ大きさなら、pitch だけでは並び方を説明できていない。")
    print("     **テイクは 3 本、切片は平均へ合わせただけである。当てはまりの良さを")
    print("     統計として読んではならない。**")
    print()

    report["cross_check"] = {
        "section": "X",
        "takes": names,
        "medians_by_take": medians,
        "spans": spans,
        "prediction": verdicts,
        "residuals_pitch": residual_rows,
    }


def _print_residuals(
    metric: str,
    label: str,
    population: str,
    source: dict[str, list[dict[str, Any]]],
    takes: list[Take],
    medians: dict[str, dict[str, float | None]],
    residual_rows: list[dict[str, Any]],
) -> None:
    """1 母集団ぶんの残差表。切片はテイク平均へ合わせ、動き方だけを比べる。"""
    names = [take.name for take in takes]
    slopes = [_slope_of(source[metric], take.name, "pitch") for take in takes]
    usable = [value for value in slopes if value is not None and isfinite(value)]
    middle = float(np.median(usable)) if usable else None
    actual = [medians[name][metric] for name in names]
    pitches = [medians[name]["pitch"] for name in names]
    if middle is None or any(value is None for value in actual + pitches):
        print(f"     {label:<8}{population:<11}{'—':>10}")
        return
    actual_array = np.asarray(actual, dtype=np.float64)
    pitch_array = np.asarray(pitches, dtype=np.float64)
    # 切片はテイク平均に合わせる。比べたいのは水準ではなく、動き方だけである。
    predicted = actual_array.mean() + middle * (pitch_array - pitch_array.mean())
    for order, name in enumerate(names):
        print(
            f"     {label:<8}{population:<11}{name:<10}{pitch_array[order]:14.3f}"
            f"{actual_array[order]:10.3f}{predicted[order]:10.3f}"
            f"{actual_array[order] - predicted[order]:+10.3f}"
        )
        residual_rows.append(
            {
                "metric": metric,
                "population": population,
                "take": name,
                "slope_pt_per_deg": middle,
                "pitch_deg": float(pitch_array[order]),
                "observed_pt": float(actual_array[order]),
                "predicted_pt": float(predicted[order]),
                "residual_pt": float(actual_array[order] - predicted[order]),
            }
        )


# --------------------------------------------------------------------------
# 実写なしの健全性確認
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str


def _synthetic_series(
    *,
    count: int = 200,
    slope: float = -0.02,
    noise: float = 0.0,
) -> tuple[Series, NDArray[np.float64]]:
    """既知の傾きを持つ合成系列。検出器も画像も要らない。

    d = base + slope_d × pitch という関係を入れておくと、表示値 pt = 100(1 - d) は
    pitch に対して -100 × slope_d の傾きを持つ。これで単位を検査できる。
    """
    from face_symmetry_lab.domain.pose import FacePose

    generator = np.random.default_rng(20260809)
    times = np.arange(count, dtype=np.float64) / 30.0
    pitch = np.linspace(-3.0, 3.0, count)
    distances = 0.02 + slope * pitch / 100.0
    if noise > 0.0:
        distances = distances + generator.normal(0.0, noise, size=count)
    poses = tuple(
        FacePose(yaw_deg=float(value) * 0.5, pitch_deg=float(value), roll_deg=-float(value))
        for value in pitch
    )
    series = Series(
        times=times,
        distances=distances,
        normalized=np.zeros((count, 1, 2), dtype=np.float64),
        poses=poses,
        normalized3d=np.zeros((count, 1, 3), dtype=np.float64),
        distances3d=distances,
        distances2dp=distances,
    )
    return series, pitch


def _check_fit() -> list[Check]:
    """最小二乗が閉じた値を返すこと。"""
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = 3.0 + 2.5 * x
    fit = fit_line(x, y)
    constant = fit_line(np.ones(20), np.arange(20, dtype=np.float64))
    short = fit_line(np.arange(2, dtype=np.float64), np.arange(2, dtype=np.float64))
    with_nan = fit_line(np.asarray([np.nan, *x]), np.asarray([1.0, *y]))
    return [
        Check(
            "傾きと切片は閉じた値",
            fit is not None
            and fit.slope is not None
            and abs(fit.slope - 2.5) < 1e-12
            and fit.intercept is not None
            and abs(fit.intercept - 3.0) < 1e-12,
            "—" if fit is None or fit.slope is None else f"傾き {fit.slope:.6f} 切片 3.0",
        ),
        Check(
            "完全な直線では相関 +1、標準誤差 0",
            fit is not None
            and fit.correlation is not None
            and abs(fit.correlation - 1.0) < 1e-12
            and fit.stderr is not None
            and fit.stderr == 0.0,
            "—" if fit is None or fit.correlation is None else f"r={fit.correlation:+.6f}",
        ),
        Check(
            "x が定数なら傾きを出さない",
            constant is not None and constant.slope is None and constant.correlation is None,
            "傾き None / 相関 None",
        ),
        Check(
            "3 枚未満では何も返さない",
            short is None,
            f"MIN_FIT_FRAMES={MIN_FIT_FRAMES}",
        ),
        Check(
            "欠測は組ごと落とす",
            with_nan is not None
            and with_nan.count == x.size
            and with_nan.slope is not None
            and abs(with_nan.slope - 2.5) < 1e-12,
            "—" if with_nan is None else f"枚数 {with_nan.count} / 傾き {with_nan.slope:.6f}",
        ),
    ]


def _check_units() -> list[Check]:
    """傾きの単位が pt / 度であること（表示式は A〜L と同じ関数を通る）。"""
    series, pitch = _synthetic_series(slope=-0.02)
    display = display_series(series, "symmetry_3d")
    fit = fit_line(pitch, display)
    raw = fit_line(pitch, series.distances3d)
    expected = 0.02  # d の傾き -0.02/100 度 → pt では +0.02
    return [
        Check(
            "pt の傾きは d の傾きの -100 倍（1pt = 0.01 IOD）",
            fit is not None
            and raw is not None
            and fit.slope is not None
            and raw.slope is not None
            and abs(fit.slope - (-100.0 * raw.slope)) < 1e-9,
            "—"
            if fit is None or fit.slope is None or raw is None or raw.slope is None
            else f"pt {fit.slope:+.6f} / d {raw.slope:+.8f}",
        ),
        Check(
            "既知の傾きをそのまま復元する",
            fit is not None and fit.slope is not None and abs(fit.slope - expected) < 1e-9,
            "—" if fit is None or fit.slope is None else f"{fit.slope:+.6f} 期待={expected:+.6f}",
        ),
        Check(
            "表示値は §8.6.3 の式そのもの（percent_values を呼ぶだけ）",
            bool(np.allclose(display, percent_values(series.distances3d))),
            f"d=0.0200 → {float(percent_values(np.asarray([0.02]))[0]):.3f}pt",
        ),
    ]


def _check_pose_series() -> list[Check]:
    """姿勢の取り出しが `measure_metric_noise` と一致すること。"""
    series, pitch = _synthetic_series()
    mine = pose_series(series, "yaw")
    theirs = yaw_values(series)
    return [
        Check(
            "yaw の取り出しは yaw_values と一致（3 軸へ一般化しただけ）",
            bool(np.array_equal(mine, theirs)),
            f"{mine.size} 枚が一致",
        ),
        Check(
            "pitch / roll も同じ規約で取れる",
            bool(np.allclose(pose_series(series, "pitch"), pitch))
            and bool(np.allclose(pose_series(series, "roll"), -pitch)),
            "pitch は入力そのもの、roll は符号反転で作った合成値",
        ),
    ]


def _check_adjacency() -> list[Check]:
    """第 1 階差が、撮影間隔で隣り合う組だけを使うこと。"""
    series, _pitch = _synthetic_series(count=10)
    times = series.times.copy()
    times[5:] += 1.0  # 5 枚目と 6 枚目のあいだにだけ隙間を作る
    gapped = Series(
        times=times,
        distances=series.distances,
        normalized=series.normalized,
        poses=series.poses,
        normalized3d=series.normalized3d,
        distances3d=series.distances3d,
        distances2dp=series.distances2dp,
    )
    mask = adjacent_mask(gapped, 1.0 / 30.0)
    return [
        Check(
            "隙間を跨ぐ組は第 1 階差から外す",
            mask.size == 9 and int(mask.sum()) == 8 and not bool(mask[4]),
            f"{int(mask.sum())} / {mask.size} 組を採用（5 枚目と 6 枚目の間を除外）",
        ),
        Check(
            "1 枚しかなければ組は作れない",
            adjacent_mask(
                Series(
                    times=np.zeros(1),
                    distances=np.zeros(1),
                    normalized=np.zeros((1, 1, 2)),
                    poses=(None,),
                    normalized3d=np.zeros((1, 1, 3)),
                    distances3d=np.zeros(1),
                    distances2dp=np.zeros(1),
                ),
                1.0 / 30.0,
            ).size
            == 0,
            "0 組",
        ),
    ]


def _check_effective_count() -> list[Check]:
    """自己相関の補正が、独立系列では効かず、強い相関では効くこと。"""
    generator = np.random.default_rng(7)
    independent = generator.normal(size=400)
    walk = np.cumsum(generator.normal(size=400))
    independent_lag = _lag_one(independent)
    walk_lag = _lag_one(walk)
    plain = _effective_count(400, independent_lag, independent_lag)
    correlated = _effective_count(400, walk_lag, walk_lag)
    return [
        Check(
            "独立な系列では実効枚数がほぼ減らない",
            independent_lag is not None and plain > 300.0,
            f"r1={independent_lag:+.3f} → n_eff={plain:.1f} / n=400"
            if independent_lag is not None
            else "—",
        ),
        Check(
            "強く相関した系列では実効枚数が大きく減る",
            walk_lag is not None and correlated < 50.0,
            f"r1={walk_lag:+.3f} → n_eff={correlated:.1f} / n=400" if walk_lag is not None else "—",
        ),
        Check(
            "補正は素の枚数を超えない（区間を狭めない）",
            _effective_count(400, 0.9, -0.9) == 400.0,
            "相関の積が負なら補正しない",
        ),
    ]


def _check_interval() -> list[Check]:
    """ノイズだけの系列で、区間が 0 を跨ぐと判定されること。"""
    generator = np.random.default_rng(11)
    x = generator.normal(size=300)
    y = generator.normal(size=300)
    noise_only = fit_line(x, y)
    series, pitch = _synthetic_series(slope=-0.02)
    clean = fit_line(pitch, display_series(series, "symmetry_3d"))
    return [
        Check(
            "無関係な 2 本では 95% 区間が 0 を跨ぐ",
            noise_only is not None and noise_only.separated_from_zero is False,
            "—"
            if noise_only is None
            else f"傾き {noise_only.slope:+.4f} 区間 {_interval(noise_only)}",
        ),
        Check(
            "きれいな直線では 0 を跨がない",
            clean is not None and clean.separated_from_zero is True,
            "—" if clean is None else f"傾き {clean.slope:+.4f} 区間 {_interval(clean)}",
        ),
    ]


def run_self_test() -> list[Check]:
    checks: list[Check] = []
    checks.extend(_check_fit())
    checks.extend(_check_units())
    checks.extend(_check_pose_series())
    checks.extend(_check_adjacency())
    checks.extend(_check_effective_count())
    checks.extend(_check_interval())
    return checks


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def build_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=None,
        help="動画ファイル（複数指定可。1 本 = 1 テイク）",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="実写なしで、最小二乗と単位と系列の取り出しを確認する",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="処理枚数の上限")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="fps 取得失敗時の仮定")
    parser.add_argument("--model", type=Path, default=None, help="landmark モデルの明示指定")
    parser.add_argument(
        "--mode",
        choices=("image", "video"),
        default="image",
        help="検出モード。§12 / §14 と揃えるなら image",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=sorted(SECTIONS),
        help="実行するセクション（省略時は全部）",
    )
    parser.add_argument("--json", type=Path, default=None, help="統計量の保存先")
    return parser.parse_args(argv)


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_frames < 1:
        raise SystemExit("--max-frames は 1 以上である必要があります。")
    if not isfinite(arguments.fps) or arguments.fps <= 0:
        raise SystemExit("--fps は正の有限値である必要があります。")
    if arguments.model is not None and not arguments.model.is_file():
        raise SystemExit(f"モデルが見つかりません: {arguments.model}")
    if arguments.self_test:
        return
    if not arguments.video:
        raise SystemExit("--video か --self-test のどちらかが必要です。")
    for path in arguments.video:
        if not path.is_file():
            raise SystemExit(f"動画が見つかりません: {path}")


def main(argv: list[str] | None = None) -> int:
    arguments = build_arguments(argv)
    validate_arguments(arguments)
    selected = arguments.section or sorted(SECTIONS)

    print(f"metric  : {CONFIG.metric_version} / {CONFIG_3D.metric_version}")
    print(f"landmark: {CONFIG.landmark_map_version}")
    print(f"IPD     : {IPD_MM:.0f}mm （両眼間距離比 × {IPD_MM:.0f} で mm 換算）")
    report: dict[str, Any] = {
        "schema": JSON_SCHEMA,
        "metric_version": CONFIG.metric_version,
        "metric_version_3d": CONFIG_3D.metric_version,
        "landmark_map_version": CONFIG.landmark_map_version,
        "ipd_mm": IPD_MM,
        "max_frames": arguments.max_frames,
        "mode": arguments.mode,
    }

    if not arguments.self_test:
        model_path, model_version = resolve_model(arguments.model)
        report["model_version"] = model_version
        print(f"model   : {model_version}")
        print(f"入力    : {', '.join(path.name for path in arguments.video)}")
        print()
        takes = [
            capture_take(
                path,
                model_path=model_path,
                model_version=model_version,
                mode=arguments.mode,
                max_frames=arguments.max_frames,
                fallback_fps=arguments.fps,
            )
            for path in arguments.video
        ]
        report["takes"] = [
            {
                "name": take.name,
                "frames_seen": take.frames_seen,
                "frames_single_face": len(take.capture.detections),
                "frames_with_distance": take.series.count,
                "fps": take.fps,
                "running_mode": take.capture.running_mode,
                "detector_class": take.capture.detector_class,
                "model_sha256": take.capture.model_sha256,
            }
            for take in takes
        ]
        short = [take.name for take in takes if take.series.count < MIN_USEFUL_FRAMES]
        if short:
            print(f"  ※ {MIN_USEFUL_FRAMES} 枚未満のテイクがある: {', '.join(short)}")
        if "T" in selected:
            section_t(takes, report)
        if "Y" in selected:
            section_y(takes, report)
        if "X" in selected:
            section_x(takes, report)

    failures = 0
    if arguments.self_test:
        heading("S", "健全性確認 — 実写なしで、最小二乗・単位・系列の取り出しを検証する")
        for check in run_self_test():
            mark = "PASS" if check.passed else "FAIL"
            failures += 0 if check.passed else 1
            print(f"  [{mark}] {check.name}")
            print(f"         {check.detail}")

    # 中間の Fit オブジェクトは JSON へ出さない（fit_json で整形済みの写しがある）。
    report.pop("_fits", None)
    report.pop("_stratified_fits", None)
    if arguments.json is not None:
        # プライバシー: 画像も landmark 座標も書かない。統計量だけ（不変条件 1）。
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n統計量を書き出しました: {arguments.json}")

    print()
    print("=" * 100)
    print("解説: docs/symmetry-metric-measurements.md §15")
    print("=" * 100)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
