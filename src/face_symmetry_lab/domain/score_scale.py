"""「全体のスコア」の表示スケール（§8.6.7, `display-scale/1.0.0`, AC-DISPLAY-05）。

§8.6.3 の `100 × (1 - d)` は、**目盛りが実測の分布に対して粗すぎる**。
`docs/symmetry-metric-measurements.md` の実測では、同一人物 3 セッションの表示
中央値が 98.280 / 98.249 / 98.523pt に集まり（§14.4）、文献上の「対称な成人」と
「非対称な成人」の差も 2.11pt しかない（§12.0）。0〜100 の目盛りのうち動くのは
上端の 2〜3pt であり、**98% は「ほぼ上限」ではなく目盛りが粗いことの帰結**である。
`ui.hud` の折れ線グラフが既に y 軸を 95〜100% に固定しているのは同じ理由で、
本 module はそのグラフ側だけで行っていた拡大を数値そのものへ適用する。

    score = clamp( 100 × (D_ZERO - s_3d) / (D_ZERO - D_FULL), 0, 100 )

`domain` に置くのは、アンカーが版に凍結された固定値であり、表示の都合で動かして
よい定数ではないためである（不変条件 10）。`ui` は式を持たず、ここを呼ぶ。

**この変換は距離を変えない。** `s_3d` の定義（§8.6.5）も `metric_version` も
エクスポート列（§9.2）も、ここでは一切触らない。§9.2 は
`distance3d_parts_equal_raw` を記録し続け、本式で一意に再現できる。

**アンカーは母集団の分位ではない。** 82 点は「上位 18%」を意味しない。本アプリは
母集団分布を持たない（OD-09、被写体 1 名）ため、§8.6 の「母集団内順位、正常範囲、
美的品質、人物の優劣や望ましさを意味しない」という制限はそのまま掛かる。参照分布
の分位点へ置き換える案は、本 module を差し替える新しい版として評価する（§8.6.7、
`docs/beauty-page-plan.md` §S）。
"""

from __future__ import annotations

import math
from typing import Final

#: §8.6.7: 目盛りの版。距離指標の `metric_version` とは独立に動く。
DISPLAY_SCALE_VERSION: Final[str] = "display-scale/1.0.0"

#: 100 点に対応するずれ（両眼間隔の 1.0%）。実測の最良フレームでも 1.29% 止まり
#: （§12.1 の take01 最大 98.71pt）なので、100 点は実写でまず到達しない上限側の
#: 端点である。**完全対称の証明ではない。**
DISTANCE_AT_FULL_SCORE: Final[float] = 0.010

#: 0 点に対応するずれ（両眼間隔の 5.0%）。文献の「非対称と判定された成人」
#: （2.85mm ≒ 4.52%、§12.0）の外側に置き、実写が下端へ張り付かないようにする。
#: **群の境界ではない。** 文献値は `symmetry-2d` 系の測り方によるもので、`s_3d`
#: と同じ尺度上にある保証はない（§8.6.7）。
DISTANCE_AT_ZERO_SCORE: Final[float] = 0.050

#: ずれが 1 ポイント（両眼間隔の 1%）動いたときに点が動く量。§8.6.7 の
#: `score_points = 25 × (symmetry_percent - 95)` と同じ傾きである。
#:
#: **アンカー 2 点から導き、`25.0` を直接書かない。** 直接書くと、アンカーを
#: 差し替えた版（`display-scale/2.0.0`）で画面の注記だけが黙って嘘になる。
#: 計算過程の開示（§8.6.3 のステップ 3）と `domain.score_explanation` は
#: どちらもここを読む。
POINTS_PER_DEVIATION_PERCENT: Final[float] = (
    0.01 * 100.0 / (DISTANCE_AT_ZERO_SCORE - DISTANCE_AT_FULL_SCORE)
)


def score_points(distance: float | None) -> float | None:
    """部位等重み距離 `s_3d` を、§8.6.7 の 0〜100 点へ写す。

    **戻り値の単位は「点」であって % ではない。** 呼び出し側は `%` を付けず、
    同じ画面へ全体の対称度（%）を併記すること（§8.6.7）。

    欠測（`None`）は欠測のまま返す。**0 点で埋めない。** 0 点は「ずれが 5.0%
    以上」という測定結果であり、値が無いこととは別である（不変条件 11）。
    非有限値も同じ理由で欠測として扱う——`min`/`max` による clamp は NaN を
    黙って 0.0 へ落とすため、clamp より前に弾く。
    """
    if distance is None or not math.isfinite(distance):
        return None
    span = DISTANCE_AT_ZERO_SCORE - DISTANCE_AT_FULL_SCORE
    points = 100.0 * (DISTANCE_AT_ZERO_SCORE - distance) / span
    return min(100.0, max(0.0, points))


def symmetry_percent(distance: float | None) -> float | None:
    """§8.6.3 の対称度変換（`100% - ずれ%`）を、文字列ではなく数値のまま返す。

    **この式の唯一の実装である。** 読み手は 3 つある——`ui.readings`（テキストと
    HUD の折れ線グラフ）、`domain.session_summary`（FR-021 の中央値と四分位）、
    そして FR-042 のレポート。式を 2 か所に書くと、画面上の別々の場所が同じ測定値
    から別の数を出す。`ui.readings.symmetry_percent` はここへ委譲するだけである
    （`score_points` と `ui.readings.score_points_value` の関係と同じ）。

    線形かつ可逆で、較正された係数を持たない。したがって同じ測定値の読む向きを
    変えるだけであり、採点ではない（§8.6.3）。

    ずれが両眼間隔そのものを超える退化ケースのために 0% で下限を設ける。欠測と
    非有限値は欠測のまま返す——`min`/`max` の clamp は NaN を黙って 0.0 へ落とす
    ため、clamp より前に弾く（`score_points` と同じ理由）。
    """
    if distance is None or not math.isfinite(distance):
        return None
    return min(100.0, max(0.0, (1.0 - distance) * 100.0))
