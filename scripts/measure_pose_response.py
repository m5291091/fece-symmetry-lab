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
    S: 形と交絡            take07（pitch を意図的に掃引したテイク）で、応答の形
                          （線形 / |pitch| 型 / pitch=0 で折れる型）と、IOD・
                          黒つぶれ率との交絡を測る（docs §16）

§4 とセクション H は |yaw| だけを調べており、pitch と d_3d の関係は一度も測って
いない。本スクリプトはそこだけを埋める。**ただし T/Y/X は take01〜03（pitch が
1.3〜6.3° しか動かない）でしか測れておらず、傾きを 1 個しか出さないので応答の
形が分からない。** セクション S はその空白を埋める。

    uv run python scripts/measure_pose_response.py --self-test
    uv run python scripts/measure_pose_response.py --video data/takes/take01.mov \
        --video data/takes/take02.mov --video data/takes/take03.mov --max-frames 600
    uv run python scripts/measure_pose_response.py --video data/takes/take07.mov \
        --section S --max-frames 600

検出とフレーム集合は `measure_session_variation.py` の `capture_take` をそのまま
呼ぶので、§14 と同一である。指標・正規化・表示式・yaw の層別は
`measure_metric_noise.py`（= `domain` の実装）から import している。**その 2 つの
スクリプトへは一切触れない。** 本スクリプトが持つ独自の計算は、最小二乗と相関と
自己相関の補正だけである。セクション S は姿勢ゲートの既定値
（`face_symmetry_lab.application.quality.QualityThresholds`）を読むだけで、
`src/` はそれ以外含め一切変更しない。

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

from face_symmetry_lab.application.quality import QualityThresholds

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.measure_metric_noise import (  # noqa: E402
    CONFIG,
    CONFIG_3D,
    IPD_MM,
    MIN_USEFUL_FRAMES,
    YAW_BAND_DEGREES,
    Capture,
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
    FrameProfile,
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
    "S": "pitch 応答の形と、見かけの距離との交絡",
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
# セクション S — pitch 応答の形と、見かけの距離との交絡
# --------------------------------------------------------------------------
#: pitch ビンの幅（度）。take07 の 49.7° 幅（-39.9〜+9.7°）を 10 ビンに割ると
#: 各ビン 28〜101 枚になり、中央値が安定する最小の幅である。これより狭いビン幅
#: では数枚しか入らないビンが出て、中央値が不安定になる。
PITCH_BIN_DEGREES: float = 5.0

#: S-4 で IOD を層別する帯の数（5 分位）。
IOD_BANDS: int = 5

#: `fit_multiple` が係数を出す最小枚数。切片 + 2 説明変数でも自由度が残る枚数。
MIN_MULTI_FRAMES: int = 10

SHAPE_NOTE = """
  何を測るか

  §15（T/Y/X）は take01〜03（pitch が 1.3〜6.3° しか動かない）で傾きを 1 個だけ
  出し、それを姿勢ゲート 30° 幅へ 24 倍外挿していた（§15.7）。take07 は pitch を
  -39.9〜+9.7°（49.7° 幅、0 を跨ぐ）で意図的に掃引したテイクである。本節は
  その中で、傾き 1 個では見えない「形」（線形か、|pitch| 型か、pitch=0 で折れる
  型か）と、掃引でだけ顕在化する IOD（見かけの両眼間隔）との交絡を測る。

      S-1  pitch をビンへ切り、表示値の中央値と四分位を並べる（形を目で見る）
      S-2  符号付き線形 / |pitch| 線形 / 折れ線の 3 モデルを当てはめて比べる
      S-3  姿勢ゲート内（|pitch| <= pitch_inner_deg）に絞って S-2 をやり直す
      S-4  IOD・黒つぶれ率・輝度が pitch と一緒に動いていないかを見る

  **本節も結論を断定しない。** 出すのは中央値・係数・残差 RMS・R^2・相関・区間と、
  数値から直接言えることまでである。
"""


def _bin_edges(values: NDArray[np.float64], width: float) -> NDArray[np.float64]:
    """半開区間 `[left, right)` のビン境界。最小値を含み、最大値も必ずどれかのビンへ入る。"""
    low = width * float(np.floor(values.min() / width))
    high = width * (float(np.floor(values.max() / width)) + 1.0)
    if high <= low:
        high = low + width
    count = max(1, round((high - low) / width))
    return low + width * np.arange(count + 1, dtype=np.float64)


def _bin_index(values: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
    """`values` が属するビン番号。境界ちょうどの値は右のビン（半開区間の側）へ入る。"""
    index = np.searchsorted(edges, values, side="right") - 1
    return np.clip(index, 0, edges.size - 2).astype(np.int64)


@dataclass(frozen=True, slots=True)
class BinRow:
    """1 ビンぶんの表示値の要約。"""

    left: float
    right: float
    count: int
    median: float
    p25: float
    p75: float


def pitch_bins(
    pitch: NDArray[np.float64],
    display: NDArray[np.float64],
    width: float = PITCH_BIN_DEGREES,
) -> list[BinRow]:
    """pitch でビン分割した表示値の中央値と四分位。枚数 0 のビンは含めない。"""
    keep = np.isfinite(pitch) & np.isfinite(display)
    xs, ys = pitch[keep], display[keep]
    if xs.size == 0:
        return []
    edges = _bin_edges(xs, width)
    bucket = _bin_index(xs, edges)
    rows: list[BinRow] = []
    for order in range(edges.size - 1):
        values = ys[bucket == order]
        if values.size == 0:
            continue
        rows.append(
            BinRow(
                left=float(edges[order]),
                right=float(edges[order + 1]),
                count=int(values.size),
                median=float(np.median(values)),
                p25=float(np.percentile(values, 25.0)),
                p75=float(np.percentile(values, 75.0)),
            )
        )
    return rows


def _fit_quality(
    x: NDArray[np.float64], y: NDArray[np.float64], fit: Fit | None
) -> tuple[float | None, float | None]:
    """`fit_line` が出した (a, b) から残差 RMS と R^2 を計算する。

    新しい最小二乗は解かない —— `fit_line` の係数へ予測させて残差を測るだけである。
    """
    if fit is None or fit.slope is None or fit.intercept is None:
        return None, None
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    xs, ys = left[keep], right[keep]
    if xs.size == 0:
        return None, None
    residual = ys - (fit.intercept + fit.slope * xs)
    rms = float(sqrt(float((residual**2).mean())))
    total = float(((ys - ys.mean()) ** 2).sum())
    r_squared = None if total <= 0.0 else float(1.0 - float((residual**2).sum()) / total)
    return rms, r_squared


@dataclass(frozen=True, slots=True)
class MultiFit:
    """複数の説明変数を持つ最小二乗あてはめ。切片は自動で加える。

    `fit_line` と違い自己相関の補正はしない —— 2 変数以上での近似が確立していない
    ためである。**傾きの信頼区間は出さない。** 区間が要る単回帰には `fit_line` を
    使うこと（S-2 / S-3 の 1・2 番目のモデルと、S-4 の IOD 帯ごとの傾きはそちら）。
    """

    count: int
    intercept: float
    #: 説明変数を渡した順の係数。切片は含まない。
    coefficients: tuple[float, ...]
    residual_rms: float
    r_squared: float | None


def fit_multiple(
    columns: tuple[NDArray[np.float64], ...], y: NDArray[np.float64]
) -> MultiFit | None:
    """y = a + Σ bᵢ·xᵢ の最小二乗。欠測は行ごと落とす。"""
    arrays = [np.asarray(column, dtype=np.float64) for column in columns]
    target = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(target)
    for array in arrays:
        keep &= np.isfinite(array)
    count = int(keep.sum())
    if count < MIN_MULTI_FRAMES:
        return None
    design = np.column_stack([np.ones(count, dtype=np.float64), *(array[keep] for array in arrays)])
    target_kept = target[keep]
    solution = np.linalg.lstsq(design, target_kept, rcond=None)[0]
    residual = target_kept - design @ solution
    rms = float(sqrt(float((residual**2).mean())))
    total = float(((target_kept - target_kept.mean()) ** 2).sum())
    r_squared = None if total <= 0.0 else float(1.0 - float((residual**2).sum()) / total)
    return MultiFit(
        count=count,
        intercept=float(solution[0]),
        coefficients=tuple(float(value) for value in solution[1:]),
        residual_rms=rms,
        r_squared=r_squared,
    )


def fit_hinge(pitch: NDArray[np.float64], display: NDArray[np.float64]) -> MultiFit | None:
    """pitch=0 で連続な折れ線 `y = a + b_neg・min(pitch,0) + b_pos・max(pitch,0)`。

    連続性は `min` / `max` の定義そのものから来る（pitch=0 ではどちらの項も 0 になる）
    ので、別立てで制約を課す必要はない。`coefficients` は `(b_neg, b_pos)`。
    """
    pitch_array = np.asarray(pitch, dtype=np.float64)
    return fit_multiple((np.minimum(pitch_array, 0.0), np.maximum(pitch_array, 0.0)), display)


def hinge_value(fit: MultiFit, pitch: float) -> float:
    """折れ線を任意の pitch で評価する。"""
    b_neg, b_pos = fit.coefficients
    return fit.intercept + b_neg * min(pitch, 0.0) + b_pos * max(pitch, 0.0)


def hinge_reading(fit: MultiFit | None) -> str:
    """b_neg / b_pos の符号から、劣化型か偏り型かを読む。判定できなければ `—`。"""
    if fit is None or len(fit.coefficients) != 2:
        return "—"
    b_neg, b_pos = fit.coefficients
    if b_neg == 0.0 or b_pos == 0.0:
        return "—"
    return "|pitch|型（劣化）" if (b_neg > 0.0) != (b_pos > 0.0) else "符号付き型（偏り）"


def gate_mask(pitch: NDArray[np.float64], thresholds: QualityThresholds) -> NDArray[np.bool_]:
    """`QualityThresholds.pitch_inner_deg` と同じ規約（境界含む）で姿勢ゲート内を選ぶ。

    `application.quality.evaluate_frame` は `abs(pose.pitch_deg) > limit` で弾く
    （超えたら弾く、ちょうどは通す）ので、ゲート内は `abs(pitch) <= limit` である。
    """
    return np.isfinite(pitch) & (np.abs(pitch) <= thresholds.pitch_inner_deg)


def quantile_bands(values: NDArray[np.float64], bands: int = IOD_BANDS) -> list[NDArray[np.intp]]:
    """値を昇順に並べ、ほぼ均等な `bands` 組へ分ける。重複せず全体を覆う。

    分位点で値そのものを切ると同じ値が境界を跨ぐことがあるので、順位で分けている。
    返り値は各帯に属する要素の、入力配列内での位置（インデックス）。
    """
    order = np.argsort(values, kind="stable")
    return [np.asarray(part, dtype=np.intp) for part in np.array_split(order, bands)]


def profile_field(take: Take, field: str) -> NDArray[np.float64]:
    """`take.series` の各行に対応する `FrameProfile` の 1 フィールド。

    `Series` は元のフレーム番号を保持しないので、`times = index * delta_seconds`
    から逆算する。`take.profiles` は生フレームを 0 始まりの連番で密に持つ
    （`capture_take` が毎フレーム 1 件ずつ積む）ので、逆算した index でそのまま
    引ける。値の無い欄（顔が 1 つに定まらなかった等）は NaN で埋める。
    """
    if take.series.count == 0 or take.delta_seconds <= 0.0 or not take.profiles:
        return np.zeros(0, dtype=np.float64)
    dense = np.full(len(take.profiles), np.nan, dtype=np.float64)
    for position, profile in enumerate(take.profiles):
        value = getattr(profile, field)
        if value is not None:
            dense[position] = float(value)
    indices = np.clip(
        np.rint(take.series.times / take.delta_seconds).astype(np.int64), 0, dense.size - 1
    )
    return dense[indices]


@dataclass(frozen=True, slots=True)
class ModelComparisonRow:
    """1 テイク × 1 母集団 × 1 指標ぶんの 3 モデル比較。"""

    take: str
    population: str
    metric: str
    frames: int
    signed: Fit | None
    signed_rms: float | None
    signed_r2: float | None
    absolute: Fit | None
    absolute_rms: float | None
    absolute_r2: float | None
    hinge: MultiFit | None


def _model_comparison_row(
    take: str,
    population: str,
    metric: str,
    pitch: NDArray[np.float64],
    display: NDArray[np.float64],
) -> ModelComparisonRow:
    signed = fit_line(pitch, display)
    absolute = fit_line(np.abs(pitch), display)
    signed_rms, signed_r2 = _fit_quality(pitch, display, signed)
    absolute_rms, absolute_r2 = _fit_quality(np.abs(pitch), display, absolute)
    hinge = fit_hinge(pitch, display)
    frames = int(np.count_nonzero(np.isfinite(pitch) & np.isfinite(display)))
    return ModelComparisonRow(
        take=take,
        population=population,
        metric=metric,
        frames=frames,
        signed=signed,
        signed_rms=signed_rms,
        signed_r2=signed_r2,
        absolute=absolute,
        absolute_rms=absolute_rms,
        absolute_r2=absolute_r2,
        hinge=hinge,
    )


def _model_row_json(row: ModelComparisonRow) -> dict[str, Any]:
    hinge = row.hinge
    return {
        "take": row.take,
        "population": row.population,
        "metric": row.metric,
        "frames": row.frames,
        "signed": fit_json(row.signed),
        "signed_residual_rms": row.signed_rms,
        "signed_r_squared": row.signed_r2,
        "absolute": fit_json(row.absolute),
        "absolute_residual_rms": row.absolute_rms,
        "absolute_r_squared": row.absolute_r2,
        "hinge": None
        if hinge is None
        else {
            "frames": hinge.count,
            "intercept": hinge.intercept,
            "b_neg_pt_per_deg": hinge.coefficients[0],
            "b_pos_pt_per_deg": hinge.coefficients[1],
            "residual_rms": hinge.residual_rms,
            "r_squared": hinge.r_squared,
            "reading": hinge_reading(hinge),
        },
    }


def _print_model_row(row: ModelComparisonRow) -> None:
    label = METRIC_LABELS[row.metric]
    print(f"     {row.take:<10}{row.population:<14}{label:<8}{row.frames:>5}枚")
    print(
        f"         符号付き {_signed(None if row.signed is None else row.signed.slope, 4):>10} "
        f"{_interval(row.signed):<26} RMS {_number(row.signed_rms, 4):>8} "
        f"R^2 {_number(row.signed_r2, 4):>7}"
    )
    print(
        f"         |pitch|  {_signed(None if row.absolute is None else row.absolute.slope, 4):>10} "
        f"{_interval(row.absolute):<26} RMS {_number(row.absolute_rms, 4):>8} "
        f"R^2 {_number(row.absolute_r2, 4):>7}"
    )
    hinge = row.hinge
    if hinge is None:
        print("         折れ線   —")
    else:
        b_neg, b_pos = hinge.coefficients
        print(
            f"         折れ線   負{b_neg:+.4f} 正{b_pos:+.4f}  "
            f"RMS {_number(hinge.residual_rms, 4)} R^2 {_number(hinge.r_squared, 4)}"
            f"   → {hinge_reading(hinge)}"
        )


PITCH_BIN_NOTE = """
  S-1) pitch ビン別の表示値中央値

  ビン幅は {width:.1f}°（`PITCH_BIN_DEGREES`）で固定。区間は半開 `[left, right)` —
  境界ちょうどの値は右のビンへ入るので、1 枚が 2 つのビンへ二重に入ることはない。
  枚数 0 のビンは行ごと飛ばす。
"""


def _s1_pitch_bins(takes: list[Take], report: dict[str, Any]) -> None:
    print(PITCH_BIN_NOTE.format(width=PITCH_BIN_DEGREES))
    by_take: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for take in takes:
        pitch = pose_series(take.series, "pitch")
        print(f"  ● {take.name}")
        by_take[take.name] = {}
        for metric, label in METRIC_LABELS.items():
            display = display_series(take.series, metric)
            rows = pitch_bins(pitch, display)
            print(f"     指標: {label}")
            print(f"     {'pitch帯（度）':<16}{'枚数':>6}{'中央値':>10}{'p25':>10}{'p75':>10}")
            print("     " + "-" * 52)
            if not rows:
                print("     —")
            for row in rows:
                bucket = f"{row.left:+.0f}〜{row.right:+.0f}"
                print(
                    f"     {bucket:<16}{row.count:>6}{row.median:>10.3f}"
                    f"{row.p25:>10.3f}{row.p75:>10.3f}"
                )
            by_take[take.name][metric] = [
                {
                    "left": row.left,
                    "right": row.right,
                    "count": row.count,
                    "median": row.median,
                    "p25": row.p25,
                    "p75": row.p75,
                }
                for row in rows
            ]
            print()
    report["pitch_bins"] = {
        "section": "S-1",
        "bin_width_deg": PITCH_BIN_DEGREES,
        "by_take": by_take,
    }


MODEL_NOTE = """
  S-2) 3 モデルの当てはめ比較（全域 / yaw 層別）

  同じ (pitch, 表示値) に 3 つのモデルを当てはめる。

      1) 符号付き線形     y = a + b・pitch                       （`fit_line` そのもの）
      2) |pitch| 線形      y = a + b・|pitch|                     （`fit_line` そのもの）
      3) 折れ線（hinge）  y = a + b_neg・min(pitch,0) + b_pos・max(pitch,0)

  3) は pitch=0 で自動的に連続になる（min/max がどちらも 0 になるため、別立ての
  制約は要らない）。**b_neg と b_pos の符号が逆なら |pitch| 型（劣化）、同じなら
  符号付き型（偏り）と読める。ただし 3 モデルの R^2 が横並びのときはこの判定は
  できない**（take07 全域・3D は 0.2941 / 0.3077 / 0.3105 で横並びだった）。

  「全域」は全フレーム、「yaw層別」はセクション Y と同じ `stable_yaw_mask`。
  残差 RMS と R^2 は、1) と 2) は `fit_line` が出した (a, b) から計算するだけで、
  別の最小二乗は解かない。3) だけは 2 つの説明変数を同時に持つので、切片込みで
  新しく最小二乗を解く（`fit_multiple`）。
"""


def _s2_model_comparison(takes: list[Take], report: dict[str, Any]) -> None:
    print(MODEL_NOTE)
    rows: list[ModelComparisonRow] = []
    for take in takes:
        pitch = pose_series(take.series, "pitch")
        selection = stable_yaw_mask(take.series)
        print(f"  ● {take.name}")
        for metric in METRIC_LABELS:
            display = display_series(take.series, metric)
            full_row = _model_comparison_row(take.name, "全域", metric, pitch, display)
            rows.append(full_row)
            _print_model_row(full_row)
            if selection is not None:
                mask, band, _median = selection
                stratified = _model_comparison_row(
                    take.name,
                    f"yaw層別±{band:.1f}",
                    metric,
                    np.where(mask, pitch, np.nan),
                    display,
                )
                rows.append(stratified)
                _print_model_row(stratified)
        print()
    report["model_comparison"] = {"section": "S-2", "rows": [_model_row_json(row) for row in rows]}


GATE_NOTE = """
  S-3) 姿勢ゲート内に絞った再計算

  `face_symmetry_lab.application.quality.QualityThresholds` の既定値
  `pitch_inner_deg = {inner:.1f}` を使い、|pitch| <= {inner:.1f}° のフレームだけで
  S-2 と同じ 3 モデルをやり直す。ここはプロダクトの既定をそのまま import して
  いるので、既定が変われば本節の「ゲート内」も一緒に動く。
"""


def _s3_gated_recalculation(
    takes: list[Take], thresholds: QualityThresholds, report: dict[str, Any]
) -> None:
    print(GATE_NOTE.format(inner=thresholds.pitch_inner_deg))
    rows: list[ModelComparisonRow] = []
    for take in takes:
        pitch = pose_series(take.series, "pitch")
        mask = gate_mask(pitch, thresholds)
        print(f"  ● {take.name}   ゲート内 {int(mask.sum())} / {take.series.count} 枚")
        for metric in METRIC_LABELS:
            display = display_series(take.series, metric)
            row = _model_comparison_row(
                take.name,
                f"ゲート内±{thresholds.pitch_inner_deg:.1f}",
                metric,
                np.where(mask, pitch, np.nan),
                display,
            )
            rows.append(row)
            _print_model_row(row)
        print()
    report["gated_model_comparison"] = {
        "section": "S-3",
        "pitch_inner_deg": thresholds.pitch_inner_deg,
        "rows": [_model_row_json(row) for row in rows],
    }


IOD_NOTE = """
  S-4) IOD（見かけの両眼間隔）との交絡

  `measure_session_variation.FrameProfile.interocular_px` を使う。pitch を掃引
  すると見かけの顔の大きさ（IOD）も一緒に動きうる。動いていれば、S-2/S-3 の
  pitch 係数の一部は IOD の効きを拾っている可能性がある。

      1) r(符号付き pitch, IOD) と r(|pitch|, IOD)（全域・ゲート内）
      2) IOD を 5 分位で 5 帯に切り、帯の中だけで pitch の傾きと区間を見る
      3) pitch と IOD を同時に入れた重回帰の係数と R^2（全域・ゲート内）
      4) 黒つぶれ率・顔輝度も、pitch と一緒に動いていないかを相関で見る

  傾きと区間は `fit_line` を使う。2) と 3) は同じ関数（`fit_line` / `fit_multiple`）
  を複数の部分集合へ当てているだけで、自己相関補正を外した別の最小二乗は書かない。
"""


def _print_correlation_line(
    prefix: str, signed: float | None, absolute: float | None, extra: str
) -> None:
    print(
        f"        {prefix:<10} r(符号付pitch) {_signed(signed, 3):>8}   "
        f"r(|pitch|) {_signed(absolute, 3):>8}{extra}"
    )


def _s4_iod_confound(
    takes: list[Take], thresholds: QualityThresholds, report: dict[str, Any]
) -> None:
    print(IOD_NOTE)
    entries: list[dict[str, Any]] = []
    for take in takes:
        pitch = pose_series(take.series, "pitch")
        iod = profile_field(take, "interocular_px")
        clip_low = profile_field(take, "face_clip_low")
        clip_high = profile_field(take, "face_clip_high")
        luminance = profile_field(take, "face_luminance")
        full_mask = np.isfinite(pitch) & np.isfinite(iod)
        gated_mask = full_mask & gate_mask(pitch, thresholds)

        print(f"  ● {take.name}")
        print("     1) r(pitch, IOD)")
        correlations: dict[str, Any] = {}
        for population, mask in (("全域", full_mask), ("ゲート内", gated_mask)):
            count = int(mask.sum())
            if count < 3:
                print(f"        {population:<8} 枚数不足")
                correlations[population] = None
                continue
            signed_r = _correlation(pitch[mask], iod[mask])
            absolute_r = _correlation(np.abs(pitch[mask]), iod[mask])
            values = iod[mask]
            extra = (
                f"   IOD {values.min():.1f}〜{values.max():.1f}px "
                f"中央値 {float(np.median(values)):.1f}"
            )
            _print_correlation_line(population, signed_r, absolute_r, extra)
            correlations[population] = {
                "signed": signed_r,
                "absolute": absolute_r,
                "iod_min": float(values.min()),
                "iod_max": float(values.max()),
                "iod_median": float(np.median(values)),
                "frames": count,
            }
        print()

        print("     2) IOD の 5 分位帯ごとの pitch 傾き")
        band_entries: dict[str, list[dict[str, Any]]] = {}
        for metric, label in METRIC_LABELS.items():
            display = display_series(take.series, metric)
            print(f"        指標: {label}")
            band_rows: list[dict[str, Any]] = []
            if int(full_mask.sum()) >= IOD_BANDS:
                base_pitch = pitch[full_mask]
                base_display = display[full_mask]
                base_iod = iod[full_mask]
                for order, positions in enumerate(quantile_bands(base_iod), start=1):
                    fit = fit_line(base_pitch[positions], base_display[positions])
                    band_iod = base_iod[positions]
                    print(
                        f"           帯{order} {band_iod.min():.1f}〜{band_iod.max():.1f}px"
                        f"（{positions.size}枚）  "
                        f"傾き {_signed(None if fit is None else fit.slope, 4)}"
                        f"  区間 {_interval(fit)}"
                    )
                    band_rows.append(
                        {
                            "band": order,
                            "frames": int(positions.size),
                            "iod_min": float(band_iod.min()),
                            "iod_max": float(band_iod.max()),
                            "fit": fit_json(fit),
                        }
                    )
            else:
                print("           枚数不足のため 5 帯へ分けられません。")
            band_entries[metric] = band_rows
        print()

        print("     3) 重回帰 pitch + IOD → 表示値")
        regressions: dict[str, dict[str, Any]] = {}
        for metric, label in METRIC_LABELS.items():
            display = display_series(take.series, metric)
            print(f"        指標: {label}")
            regressions[metric] = {}
            for population, mask in (("全域", full_mask), ("ゲート内", gated_mask)):
                fit = fit_multiple((pitch[mask], iod[mask]), display[mask])
                if fit is None:
                    print(f"           {population:<8} —")
                    regressions[metric][population] = None
                    continue
                pitch_coefficient, iod_coefficient = fit.coefficients
                print(
                    f"           {population:<8} pitch {pitch_coefficient:+.4f} pt/度"
                    f"   IOD {iod_coefficient:+.4f} pt/px   R^2 {_number(fit.r_squared, 4)}"
                    f"   （{fit.count}枚）"
                )
                regressions[metric][population] = {
                    "frames": fit.count,
                    "pitch_pt_per_deg": pitch_coefficient,
                    "iod_pt_per_px": iod_coefficient,
                    "r_squared": fit.r_squared,
                }
        print()

        print("     4) pitch と黒つぶれ率・輝度・IOD との相関（全域）")
        covariate_correlations: dict[str, dict[str, float | None]] = {}
        for label, values in (
            ("face_clip_low", clip_low),
            ("face_clip_high", clip_high),
            ("face_luminance", luminance),
            ("interocular_px", iod),
        ):
            mask = np.isfinite(pitch) & np.isfinite(values)
            if int(mask.sum()) < 3:
                print(f"        {label:<18} 枚数不足")
                covariate_correlations[label] = {"signed": None, "absolute": None}
                continue
            signed_r = _correlation(pitch[mask], values[mask])
            absolute_r = _correlation(np.abs(pitch[mask]), values[mask])
            print(
                f"        {label:<18} r(符号付) {_signed(signed_r, 3):>8}   "
                f"r(|pitch|) {_signed(absolute_r, 3):>8}"
            )
            covariate_correlations[label] = {"signed": signed_r, "absolute": absolute_r}
        print()

        entries.append(
            {
                "take": take.name,
                "correlation_with_iod": correlations,
                "iod_band_pitch_slope": band_entries,
                "regression_pitch_iod": regressions,
                "covariate_correlation": covariate_correlations,
            }
        )

    report["iod_confound"] = {
        "section": "S-4",
        "pitch_inner_deg": thresholds.pitch_inner_deg,
        "iod_bands": IOD_BANDS,
        "entries": entries,
    }


def section_s(takes: list[Take], report: dict[str, Any]) -> None:
    heading("S", "pitch 応答の形と、見かけの距離との交絡 — take07 の掃引で埋める空白")
    print(SHAPE_NOTE)
    thresholds = QualityThresholds()
    _s1_pitch_bins(takes, report)
    _s2_model_comparison(takes, report)
    _s3_gated_recalculation(takes, thresholds, report)
    _s4_iod_confound(takes, thresholds, report)


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


def _check_pitch_bins() -> list[Check]:
    """ビン分割が半開区間であること（境界値は右のビンへ入る）。枚数 0 のビンが消えること。"""
    # 幅 5 の境界（-35）ちょうどの値と、その隣（-40）を混ぜる。半開なら -35 は
    # 右のビン [-35,-30) へ入り、[-40,-35) には -40 だけが残る。
    boundary_pitch = np.asarray([-40.0, -35.0], dtype=np.float64)
    boundary_display = np.asarray([10.0, 20.0], dtype=np.float64)
    boundary_rows = pitch_bins(boundary_pitch, boundary_display, width=5.0)

    # 中間の帯を素通りさせ、枚数 0 のビンが出力に現れないことを確かめる。
    sparse_pitch = np.asarray([-10.0, -10.0, 8.0, 8.0], dtype=np.float64)
    sparse_display = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sparse_rows = pitch_bins(sparse_pitch, sparse_display, width=5.0)
    sparse_lefts = {row.left for row in sparse_rows}

    return [
        Check(
            "境界ちょうどの値は左のビンではなく右のビンへ入る（半開区間）",
            len(boundary_rows) == 2
            and boundary_rows[0].left == -40.0
            and boundary_rows[0].count == 1
            and boundary_rows[1].left == -35.0
            and boundary_rows[1].count == 1,
            "—"
            if len(boundary_rows) != 2
            else f"[-40,-35)={boundary_rows[0].count}枚 [-35,-30)={boundary_rows[1].count}枚",
        ),
        Check(
            "枚数 0 のビンは出力に現れない",
            len(sparse_rows) == 2 and sparse_lefts == {-10.0, 5.0},
            f"出力に現れたビンの左端: {sorted(sparse_lefts)}（帯は [-10,10) を 4 分割）",
        ),
    ]


def _check_fit_quality() -> list[Check]:
    """`_fit_quality` が `fit_line` の係数から RMS / R^2 を計算し、外れ値に反応すること。"""
    x = np.linspace(-5.0, 5.0, 50)
    y = 2.0 + 3.0 * x
    fit = fit_line(x, y)
    rms, r_squared = _fit_quality(x, y, fit)

    noisy_y = y.copy()
    noisy_y[0] += 100.0
    noisy_fit = fit_line(x, noisy_y)
    noisy_rms, noisy_r_squared = _fit_quality(x, noisy_y, noisy_fit)
    return [
        Check(
            "完全な直線では RMS 0・R^2 1",
            rms is not None
            and rms < 1e-9
            and r_squared is not None
            and abs(r_squared - 1.0) < 1e-9,
            f"RMS={rms} R^2={r_squared}",
        ),
        Check(
            "外れ値を混ぜると RMS が増え R^2 が下がる",
            None not in (rms, r_squared, noisy_rms, noisy_r_squared)
            and noisy_rms > rms  # type: ignore[operator]
            and noisy_r_squared < r_squared,  # type: ignore[operator]
            f"RMS {rms:.4f}→{noisy_rms:.4f}  R^2 {r_squared:.4f}→{noisy_r_squared:.4f}"
            if None not in (rms, r_squared, noisy_rms, noisy_r_squared)
            else "—",
        ),
    ]


def _hinge_sample(
    *, kind: str, count: int = 200
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`y = c - k|x|`（kind="abs"）または `y = c + k·x`（kind="signed"）の純粋な合成データ。

    ノイズなし。|x| = max(x,0) - min(x,0)、x = max(x,0) + min(x,0) が恒等式なので、
    折れ線の係数は閉じた値（k または ±k）へ厳密に収束する。
    """
    x = np.linspace(-10.0, 10.0, count)
    y = 5.0 - 0.3 * np.abs(x) if kind == "abs" else 5.0 + 0.3 * x
    return x, y


def _coefficients_text(fit: MultiFit | None) -> str:
    """`MultiFit.coefficients` を安全に文字列化する。長さが 2 でなくても壊れない。

    罠テスト（`fit_multiple` が説明変数を取りこぼす等）で `coefficients` の長さが
    想定と違っても、self-test 自体が例外で落ちずに `FAIL` を報告できるようにする。
    """
    if fit is None:
        return "—"
    parts = (f"b_{index}={value:+.6f}" for index, value in enumerate(fit.coefficients))
    return ", ".join(parts)


def _check_hinge_shape() -> list[Check]:
    """折れ線が |pitch| 型（劣化）と符号付き型（偏り）を b_neg / b_pos の符号だけで見分けること。"""
    x_abs, y_abs = _hinge_sample(kind="abs")
    x_signed, y_signed = _hinge_sample(kind="signed")
    degraded = fit_hinge(x_abs, y_abs)
    biased = fit_hinge(x_signed, y_signed)
    return [
        Check(
            "y = c - k|x| では b_neg と b_pos が符号が逆で、大きさがほぼ一致",
            degraded is not None
            and len(degraded.coefficients) == 2
            and (degraded.coefficients[0] > 0.0) != (degraded.coefficients[1] > 0.0)
            and abs(abs(degraded.coefficients[0]) - abs(degraded.coefficients[1])) < 1e-6,
            _coefficients_text(degraded),
        ),
        Check(
            "y = c + k・x では b_neg と b_pos が同符号で、ほぼ一致",
            biased is not None
            and len(biased.coefficients) == 2
            and (biased.coefficients[0] > 0.0) == (biased.coefficients[1] > 0.0)
            and abs(biased.coefficients[0] - biased.coefficients[1]) < 1e-6,
            _coefficients_text(biased),
        ),
    ]


def _check_hinge_continuity() -> list[Check]:
    """折れ線は pitch=0 で連続（左右の極限が切片に一致すること）。"""
    x, y = _hinge_sample(kind="abs", count=300)
    fit = fit_hinge(x, y)
    if fit is None:
        return [Check("折れ線は pitch=0 で連続", False, "フィットが None")]
    if len(fit.coefficients) != 2:
        return [
            Check(
                "折れ線は pitch=0 で連続",
                False,
                f"coefficients の長さが 2 でない: {_coefficients_text(fit)}",
            )
        ]
    left = hinge_value(fit, -1e-6)
    right = hinge_value(fit, 1e-6)
    return [
        Check(
            "折れ線は pitch=0 で連続（左右の極限が一致し、切片とも一致）",
            abs(left - right) < 1e-4
            and abs(left - fit.intercept) < 1e-4
            and abs(right - fit.intercept) < 1e-4,
            f"左極限={left:.6f} 右極限={right:.6f} 切片={fit.intercept:.6f}",
        ),
    ]


def _check_hinge_reading() -> list[Check]:
    """`hinge_reading` が符号の一致・不一致から正しい文言を選ぶこと。"""
    degraded = fit_hinge(*_hinge_sample(kind="abs"))
    biased = fit_hinge(*_hinge_sample(kind="signed"))
    return [
        Check(
            "読み: 符号が逆なら |pitch| 型、同じなら符号付き型",
            hinge_reading(degraded) == "|pitch|型（劣化）"
            and hinge_reading(biased) == "符号付き型（偏り）",
            f"degraded→{hinge_reading(degraded)} / biased→{hinge_reading(biased)}",
        ),
    ]


def _check_multi_regression() -> list[Check]:
    """既知の 2 変数合成 y = a + b1・x1 + b2・x2 の係数を `fit_multiple` が復元すること。"""
    generator = np.random.default_rng(20260810)
    x1 = generator.normal(size=300)
    x2 = generator.normal(size=300)
    y = 4.0 + 1.5 * x1 - 2.5 * x2
    fit = fit_multiple((x1, x2), y)
    ok = (
        fit is not None
        and len(fit.coefficients) == 2
        and abs(fit.intercept - 4.0) < 1e-8
        and abs(fit.coefficients[0] - 1.5) < 1e-8
        and abs(fit.coefficients[1] - (-2.5)) < 1e-8
    )
    return [
        Check(
            "既知の 2 変数合成 y=a+b1・x1+b2・x2 の係数を復元する",
            ok,
            "—"
            if fit is None
            else f"a={fit.intercept:.6f} 係数={tuple(round(v, 6) for v in fit.coefficients)}",
        ),
    ]


def _check_gate_mask() -> list[Check]:
    """姿勢ゲート内マスクが `pitch_inner_deg` と厳密一致すること（境界 ±15.0 を含む）。"""
    thresholds = QualityThresholds()
    pitch = np.asarray(
        [-20.0, -15.0, -14.999, 0.0, 14.999, 15.0, 20.0, float("nan")], dtype=np.float64
    )
    mask = gate_mask(pitch, thresholds)
    expected = np.asarray([False, True, True, True, True, True, False, False])
    return [
        Check(
            "ゲート内は abs(pitch) <= pitch_inner_deg（境界 ±15.0 ちょうどを含む）",
            bool(np.array_equal(mask, expected)),
            f"pitch_inner_deg={thresholds.pitch_inner_deg} mask={mask.tolist()}",
        ),
    ]


def _check_quantile_bands() -> list[Check]:
    """5 分位帯が全フレームを重複なく覆い、大きさがほぼ均等であること。"""
    values = np.arange(23, dtype=np.float64)  # 5 で割り切れない枚数で試す
    bands = quantile_bands(values, bands=5)
    sizes = [int(part.size) for part in bands]
    covered = np.concatenate(bands) if bands else np.zeros(0, dtype=np.intp)
    return [
        Check(
            "5 分位帯は重複せず全フレームを覆う",
            covered.size == values.size and set(covered.tolist()) == set(range(values.size)),
            f"帯サイズ={sizes} 合計={covered.size}/{values.size}",
        ),
        Check(
            "帯サイズはほぼ均等（最大と最小の差は 1 以下）",
            bool(sizes) and max(sizes) - min(sizes) <= 1,
            f"帯サイズ={sizes}",
        ),
    ]


def _check_profile_field() -> list[Check]:
    """`profile_field` が、逆算した元フレーム番号で `FrameProfile` を正しく引くこと。"""
    series, _pitch = _synthetic_series(count=5)
    profiles = tuple(
        FrameProfile(
            index=index,
            time_seconds=index / 30.0,
            face_count=1,
            frame_luminance=0.0,
            frame_clip_high=0.0,
            frame_clip_low=0.0,
            interocular_px=100.0 + index,
        )
        for index in range(5)
    )
    take = Take(
        name="synthetic",
        path=Path("synthetic.mov"),
        fps=30.0,
        frames_seen=5,
        profiles=profiles,
        capture=Capture(
            detections=(),
            frames_seen=5,
            delta_seconds=1.0 / 30.0,
            source_kind="video",
            mode="image",
            model_version="test",
            model_sha256="0" * 64,
        ),
        series=series,
    )
    values = profile_field(take, "interocular_px")
    expected = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
    return [
        Check(
            "profile_field は逆算した元フレーム番号で FrameProfile の欄を引く",
            bool(np.allclose(values, expected)),
            f"{values.tolist()}",
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
    checks.extend(_check_pitch_bins())
    checks.extend(_check_fit_quality())
    checks.extend(_check_hinge_shape())
    checks.extend(_check_hinge_continuity())
    checks.extend(_check_hinge_reading())
    checks.extend(_check_multi_regression())
    checks.extend(_check_gate_mask())
    checks.extend(_check_quantile_bands())
    checks.extend(_check_profile_field())
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
        if "S" in selected:
            section_s(takes, report)

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
    print("解説: docs/symmetry-metric-measurements.md §15（T/Y/X） / §16（S）")
    print("=" * 100)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
