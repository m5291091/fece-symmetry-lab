"""Camera と静止画の両セクションが共有する詳細結果の描画。

文言は §5.3 に従う。指標は実験的な左右差 *距離* として導入し、`HOLD` は次に
取るべき操作へ言い換え、値の大小に価値判断を結び付けない。§8.6 で 0〜100 の
変換を廃止したため、ここに出る対称度はすべて両眼間隔に対する正規化距離を
百分率で示したものである。小さいという事実は対応点どうしが近いことだけを
意味し、決して「良い」を意味しない。

**この module は「描画」と「文言」だけを持つ。** 表示値の解決と整形は
`ui.readings`、計算過程の算術は `domain.score_explanation`、対称化パネルは
`ui.symmetrization_panel` にある。§8.6.3（H4）でこの 3 つを外へ出したのは、
以前ここが「文言・整形・値の解決・算術・描画」を 1 つの module に抱えており、
HUD や吹き出しが数値を整形するためだけにこの描画 module へ依存していたため
である。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import Final

import streamlit as st

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate, Status
from face_symmetry_lab.domain.score_explanation import ScoreExplanation, explain_score
from face_symmetry_lab.domain.score_scale import (
    DISTANCE_AT_FULL_SCORE,
    DISTANCE_AT_ZERO_SCORE,
    POINTS_PER_DEVIATION_PERCENT,
)
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.i18n import t
from face_symmetry_lab.ui.readings import (
    format_distance,
    format_score,
    format_symmetry,
    gender_label,
    part_label,
    resolve_equal_weight_distance,
    resolve_overall_distance,
    resolve_part_distance_3d,
)

SCORE_HEADING = "対称度（実験値・両眼距離比）"

#: §5.2: 距離は両眼間隔に対する比なので、小数 2 桁の百分率が、精度を偽らずに
#: 読める形になる。§5.3/§12.1（`docs/symmetry-metric-measurements.md`）: 実写の
#: 表示値はテイク内でも数パーセントポイントの狭い範囲に集まるため、「100% は
#: 目標値ではない」という既存の注意の延長として、その狭い範囲の中の違いを
#: 対称・非対称の判定として読ませない注意文を添える。
#: §5.3.1（2026-08-16）: 常時表示は 1 行まで。式の説明・「100% は目標値ではない」
#: ・「狭い範囲に集まる」といった根拠は About と `help` が持つ。以前はこの 5 文が
#: 画面に出続けており、その下に同種の注意がさらに 3 本並んでいた。
DISTANCE_UNIT_CAPTION = (
    "対称度は両眼間隔に対するずれを 100% から引いた値です。目標値ではありません。"
)

#: 上の 1 行に収まらない説明。`help=` として渡す——常時は見えないが、その場で読める。
DISTANCE_UNIT_HELP: Final[str] = (
    "対称度は、両眼の間隔を 100% としたときの左右対応点のずれを 100% から引いた値です。"
    "100% は対応点距離がすべて 0 になる数学上の上限で、目標値ではありません。"
    "高い・低いは良し悪しを表しません。"
    "実写では表示値が数パーセントポイントの狭い範囲に集まりやすく、"
    "その中の違いを対称・非対称の判定として読まないでください。"
)


#: §8.6.3/§8.6.5: `s_3d` のラベル。**ユーザー指定により「部位平均の対称度」
#: ではなく「全体のスコア」とする**（2026-08-11、W5-fix）。実体は §8.6.3 と
#: 同じ線形・無パラメータの変換で読んだ対称度であり、`s_3d` の重みは全部位に
#: ついて 1 に固定され調整の余地を持たないため、§8.6 が禁じる較正されていない
#: 自由パラメータには当たらない（§8.6.3「表示語としてのスコア」・§8.6.5）。
#: 採点・評価ではないことは help（`PART_EQUAL_WEIGHT_HELP`）で明示する。
PART_EQUAL_WEIGHT_LABEL: Final[str] = "全体のスコア"

#: 計算過程の `st.expander` の見出し。`{label}` へ `t(PART_EQUAL_WEIGHT_LABEL)`
#: を渡す（f-string を `t()` へ渡さないため、テンプレート定数へ切り出す）。
SCORE_CALCULATION_EXPANDER_TITLE_TEMPLATE: Final[str] = "{label}の計算過程を見る"

#: §8.6.3/§8.6.5/§8.6.7: 「全体のスコア」の help。全体の対称度（対応点を 1 票ずつ
#: 数える `d_3d`）とは重みも目盛りも異なる並列の総合量であることと、採点でも順位
#: でもないことを明示する。**アンカー 2 点を必ず含める。** これが画面に無いと、
#: §8.6.3 が要求する可逆性（点から `s_3d` へ戻せること）が利用者側で成立しない。
PART_EQUAL_WEIGHT_HELP: Final[str] = (
    "部位を 1 票ずつ数えた対称度を、実測が集まる範囲へ合わせた目盛りで読み直した値です。"
    "鼻のようにペア数の少ない部位も、目のように多い部位と同じ重みで入ります。"
    "ずれが両眼間隔の 1.0% で 100 点、5.0% で 0 点になります。"
    "全体の対称度（対応点を 1 票ずつ）とは重みも目盛りも違うため、数字の大小を直接比べられません。"
    "較正された感度係数や部位の重み付けを持たない値であり、採点ではありません。"
    "母集団の分布を持たないため、順位や正常範囲を表しません。"
    "高い・低いは良し悪しを表しません。"
)

#: §8.6.7: 目盛りを引き伸ばしたことの帰結を開示する注記。**分解能は上がるが情報は
#: 増えない。** 25 倍細かくすると、同じ 25 倍でばらつきも拡大する。実測
#: （`docs/symmetry-metric-measurements.md` §14.4 のセッション間全幅 0.273pt =
#: 6.8 点）を数字で示し、その幅より小さい点差を人物間の差として読ませない。
#: §5.3.1: 1 行まで。根拠（アンカーの由来・被写体 1 名の測定であること）は
#: About と `help` が持つ。**撮り直しで動く量だけは残す**——数字が動かないものだと
#: 読まれると、この目盛りは誤解の元になる。
SCORE_SCALE_CAPTION: Final[str] = "撮り直すだけで 6.8 点ほど動く目盛りです。"

SCORE_SCALE_HELP: Final[str] = (
    "この目盛りは、実測が集まる範囲（ずれ 1.0〜5.0%）へ合わせて引き伸ばしてあります。"
    "引き伸ばした分だけばらつきも拡大し、同じ人を同じ条件で撮り直しただけでも 6.8 点ほど動きます"
    "（`docs/symmetry-metric-measurements.md` §14.4、被写体 1 名）。"
    "それより小さい点差を、人と人の違いとして読まないでください。"
)

#: §5.3 が義務付ける「この数値をどう読むか」の留保。**畳まない。**
#: `SCORE_SCALE_CAPTION` と同じ理由で、開かないと見えない場所へ置くと数字だけが
#: 独り歩きする。以前はこれらが詳細結果の末尾に無名の `st.caption` として 3 本
#: 続き、その後ろに指標の版がもう 1 本続いていた——区切りの無い灰色の壁であり、
#: 読まれることを前提にできない形だった。見出しを持つ 1 つの箱へまとめ、走査
#: できる並びにする。**内容は 1 つも減らしていない。**
READING_LIMITS_HEADING: Final[str] = "この数値の読み方"

#: §5.3.1: 3 本の長文から 1 行へ。**残したのは「何に使えないか」だけ**で、
#: 式の内訳（部位別を平均しても全体にならない等）は About が持つ。名前を付けて
#: 単独の定数にしてあるのは、`READING_LIMITS[0]`（タプルの添字）は `t()` の
#: AST 走査（`tests/ui/test_i18n.py`）が解決できる形ではないため——名前を経由
#: すれば `t(READING_LIMITS_NOTE)` として訳せる。
READING_LIMITS_NOTE: Final[str] = (
    "同一人物・同一環境の中で傾向を見るための値です。他人との比較や医学的評価には使えません。"
)
READING_LIMITS: Final[tuple[str, ...]] = (READING_LIMITS_NOTE,)

#: §5.3・§14.7（同 docs）: 部位別の値は同一条件で撮り直すだけでも動く幅がある。
#: この注記は 4 部位すべてに等しく掛かる形にし、特定の部位（鼻など）だけを
#: 名指しで「使えない」と述べない。測定は被写体 1 名によるため、部位ごとの
#: 個人差は未検証である（OD-09）。
#: §5.3.1: `help=` へ移した。常時表示ではなく、部位別を見ている人がその場で
#: 開ける場所に置く。
PART_LEVEL_VARIATION_CAPTION: Final[str] = (
    "部位別の値は、同じ条件で撮り直すだけでも動きます。"
    "撮影のたびに動く幅より小さい変化は、部位別の値だけでは左右差の変化として読み取れません。"
    "この幅は部位ごとに異なり、個人差は未検証です。"
)


# --- §8.6.3(H4): 「全体のスコア」の計算過程を 3 ステップで開示する -------------
#
# 式を 1 つの塊で見せると、記号の意味・途中経過・目盛りの読み替えが同時に
# 押し寄せて追えない。各ステップは「何をするか」の平叙文を先に置き、そのあとに
# 実数値を代入した計算を続ける。ステップの区切りは表示の構造であり、計算そのもの
# （`domain.score_explanation`）は変えない。
#
# 文言には価値判断語（§5.3・不変条件 8）も、母集団の分布を持たないため基準値
# （「◯%以上なら対称」）も書かない。§5.2 が禁じる進捗バー・ゲージ・達成度の
# 演出も用いず、目盛りの端点は数値と文章で示す。

#: `SCORE_STEP_TITLES`（タプル）の添字は `t()` の AST 走査が解決できる形では
#: ないため、各見出しに名前を付けておく——訳す箇所は `t(SCORE_STEP_TITLE_*)` を
#: 使う。タプル自体は既存の呼び出し側（テストを含む）のために残す。
SCORE_STEP_TITLE_DEVIATIONS: Final[str] = "ステップ 1 — 4 つの部位それぞれの「ずれ」を読む"
SCORE_STEP_TITLE_COMBINE: Final[str] = "ステップ 2 — 4 つのずれを 1 つにまとめる"
SCORE_STEP_TITLE_SCALE: Final[str] = "ステップ 3 — まとめたずれを「点」へ置き換える"

SCORE_STEP_TITLES: Final[tuple[str, str, str]] = (
    SCORE_STEP_TITLE_DEVIATIONS,
    SCORE_STEP_TITLE_COMBINE,
    SCORE_STEP_TITLE_SCALE,
)

SCORE_CALCULATION_DEVIATION_INTRO: Final[str] = (
    "ここでいう「ずれ」は、両目の間隔を100%としたときの、左右で対応する"
    "点どうしの離れ具合です。両目の間隔が6cmの人なら、ずれ1%はおよそ0.6mmに"
    "あたります。輪郭・目・鼻・口の4か所について、それぞれの値と、次のステップで"
    "使う二乗を並べます。"
)

#: ステップ 2 の平叙文。**なぜ単純平均ではないのかを述べる。** ここが無いと
#: 「二乗して平方根」が意味不明な儀式に見え、初学者が最初に脱落する箇所になる。
SCORE_CALCULATION_COMBINE_INTRO: Final[str] = (
    "4つのずれを、二乗して平均し、平方根をとって1つの数にまとめます"
    "（二乗平均平方根）。単純に足して4で割らないのは、1か所だけ大きくずれている"
    "ときに、その分が残り3部位に薄められてしまうためです。二乗すると大きいずれ"
    "ほど強く残り、最後に平方根をとることで単位が元の「ずれ」に戻ります。"
)

#: ステップ 3 の平叙文。アンカー 2 点を数値として述べる。これが画面に無いと
#: §8.6.3 が要求する可逆性（点から `s_3d` へ戻せること）が成立しない。
SCORE_CALCULATION_SCALE_INTRO: Final[str] = (
    "まとめたずれを、2つの目印にあてはめて点へ置き換えます。ずれが両眼間隔の"
    "1.0%のとき100点、5.0%のとき0点とし、その間を直線でつなぎます。この2つの"
    "目印は実測と文献から決めた固定の値で、母集団の中の位置を表すものでは"
    "ありません。"
)

#: §8.6.7: 目盛りの傾き。数値は `domain.score_scale` の凍結アンカーから導く
#: （`POINTS_PER_DEVIATION_PERCENT`）。UI 側に `25` を直書きすると、アンカーを
#: 差し替えた版で注記だけが黙って嘘になる。
#:
#: `{points}` を含むテンプレートにしてあるのは、f-string で値を埋め込んだ
#: 定数は AST 上 `JoinedStr` になり、`t()` の走査が「同じモジュールの
#: module-level で文字列リテラルへ束縛された名前」として解決できないため
#: （`tests/ui/test_i18n.py`）。`SCORE_CALCULATION_SLOPE_NOTE` は表示に使う
#: 完成形の文字列として、既存の呼び出し側のためにそのまま残す。
SCORE_CALCULATION_SLOPE_NOTE_TEMPLATE: Final[str] = (
    "この目盛りでは、ずれが1ポイント（両眼間隔の1%）動くと点が"
    "{points:.0f}点動きます。"
    "点の細かさは、測定の細かさではありません。"
)
SCORE_CALCULATION_SLOPE_NOTE: Final[str] = SCORE_CALCULATION_SLOPE_NOTE_TEMPLATE.format(
    points=POINTS_PER_DEVIATION_PERCENT
)

#: §8.6.5・不変条件 11: 部位が 1 つでも欠測すると `s_3d` は欠測になる。そのとき
#: 計算過程は数値を出さず、古い値や 0 でも埋めない。
SCORE_CALCULATION_UNAVAILABLE_NOTICE: Final[str] = (
    "部位別の値のいずれかが欠測しているため、今は数値を当てはめられません。"
)

SCORE_CALCULATION_WEIGHT_DIFFERENCE_CAPTION: Final[str] = (
    "全体の対称度は対応点126個を1個ずつ数えるため、点の多い部位（目26組）ほど"
    "強く効きます。全体のスコアは部位を1つずつ数えるため、点の少ない部位"
    "（鼻7組）も同じ重みで入ります。この重みの違いが、全体の対称度と全体の"
    "スコアが一致しない理由です。"
)

#: §8.6.7: なぜ目盛りを引き伸ばしたのかを、元の 1〜2% という事実から説明する。
#: 「98%前後になる」という旧来の注記の役割を、引き伸ばし後の目盛りで置き換える。
SCORE_CALCULATION_HIGH_VALUE_NOTE: Final[str] = (
    "成人の顔の左右のずれは、両眼の間隔の1〜2%ほどです。100からずれを引くだけの"
    "目盛りでは、これが「98%前後」に集まって差が読めません。上の式はその1〜2%の"
    "あたりを広げて読むための目盛りであり、点数そのものは母集団の中の位置を表しません。"
)

#: `docs/symmetry-metric-measurements.md` §12.0（文献の対称/非対称群差）・
#: §12.1（撮り直しの表示値レンジ、take01/take02 3D、被写体 1 名）・§14.4
#: （3 セッションの中央値）を §8.6.7 の目盛りへ換算した値。
SCORE_CALCULATION_MEASURED_RANGE_CAPTION: Final[str] = (
    "同じ人を撮り直した実測（被写体 1 名）では、セッションごとの中央値が81〜88点、"
    "フレーム単位では46〜93点に散りました"
    "（`docs/symmetry-metric-measurements.md` §12.1・§14.4）。文献で「対称」とされた"
    "成人と「非対称」とされた成人の差2.11ポイントは、この目盛りでは52.8点にあたります"
    "（同§12.0）。撮影条件だけで数点動くため、小さな点差を人物の違いとして読めません。"
)


def _bare_percent(value: float) -> str:
    """`format_distance` の数字だけ（`%` を落とす）。

    式の中では単位を 1 行の末尾へ 1 度だけ置くほうが読みやすい。丸め規則を
    3 つ目に増やさないよう、値そのものは共有フォーマッタを通す。
    """
    return format_distance(value).rstrip("%")


def _square_percent(square: float) -> str:
    """距離の二乗を「パーセントポイントの二乗」として整形する。

    ステップ 1 の表とステップ 2 の合計が同じ単位系で並ぶようにするための
    単位換算であり（距離² → %²）、新しい計算ではない。`format_distance` が
    距離へ 100 を掛けるのと同じ位置付けである。
    """
    return f"{square * 10_000.0:.2f}"


def _display_width(text: str) -> int:
    """等幅フォントで占める桁数。全角は 2 桁として数える。

    `str.ljust` / `f"{x:>8}"` はどちらも **文字数** で詰めるため、「輪郭」
    （2 文字 = 4 桁）と「目」（1 文字 = 2 桁）が揃わない。表の桁を合わせないと、
    ステップ 1 が「読者が縦に目で追える表」として機能しない。
    """
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _cell(text: str, cells: int, *, right: bool = False) -> str:
    padding = " " * max(0, cells - _display_width(text))
    return padding + text if right else text + padding


#: ステップ 1 の表の桁。部位ラベル（左寄せ）・ずれ・二乗（どちらも右寄せ）。
_PART_COLUMN_CELLS: Final[int] = 8
_VALUE_COLUMN_CELLS: Final[int] = 9


def _deviation_row(label: str, deviation: str, square: str) -> str:
    return (
        _cell(label, _PART_COLUMN_CELLS)
        + _cell(deviation, _VALUE_COLUMN_CELLS, right=True)
        + _cell(square, _VALUE_COLUMN_CELLS, right=True)
    )


def _step_deviations_block(explanation: ScoreExplanation) -> str:
    """ステップ 1: 部位ごとのずれと二乗を 1 行ずつ並べる。

    見出し語（`t("部位")` 等）・部位ラベル（`part_label`）の英訳は、いずれも
    `_PART_COLUMN_CELLS` / `_VALUE_COLUMN_CELLS` の桁数に収まるよう選んである
    （`_display_width` は East Asian Width を見るため、半角の英語ラベルでも
    行の桁は揃う）。
    """
    return "\n".join(
        [
            _deviation_row(t("部位"), t("ずれ"), t("ずれ²")),
            *(
                _deviation_row(
                    part_label(part.name),
                    format_distance(part.distance),
                    _square_percent(part.square),
                )
                for part in explanation.parts
            ),
        ]
    )


#: 実数値を代入する前の、ステップ 2 の式そのもの。`s_3d` が欠測のときはこれだけ
#: を示す（不変条件 11: 古い値や 0 で埋めない）。
SCORE_FORMULA_COMBINE: Final[str] = (
    "まとめたずれ = √( (輪郭のずれ² + 目のずれ² + 鼻のずれ² + 口のずれ²) ÷ 4 )"
)


#: ステップ 2 の途中経過。実数値は `.format()` で埋め込む（f-string を `t()` へ
#: 渡さないため、テンプレート定数へ切り出す）。
STEP_COMBINE_SUM_TEMPLATE: Final[str] = "  ずれ² の合計 = {squares} = {total}"
STEP_COMBINE_MEAN_TEMPLATE: Final[str] = "  {count} 部位で平均 = {total} ÷ {count} = {mean}"
STEP_COMBINE_SQRT_TEMPLATE: Final[str] = "  平方根をとる = √{mean} = {combined}  ← まとめたずれ"


def _step_combine_block(explanation: ScoreExplanation) -> str:
    """ステップ 2: 二乗和 → 平均 → 平方根の途中経過を、実数値で示す。"""
    squares = " + ".join(_square_percent(part.square) for part in explanation.parts)
    total = _square_percent(explanation.square_sum)
    count = len(explanation.parts)
    mean = _square_percent(explanation.mean_square)
    return "\n".join(
        (
            t(SCORE_FORMULA_COMBINE),
            "",
            t(STEP_COMBINE_SUM_TEMPLATE).format(squares=squares, total=total),
            t(STEP_COMBINE_MEAN_TEMPLATE).format(count=count, total=total, mean=mean),
            t(STEP_COMBINE_SQRT_TEMPLATE).format(
                mean=mean, combined=format_distance(explanation.combined_distance)
            ),
        )
    )


#: ステップ 3 の式そのもの（実数値を代入する前）。`{label}` は `PART_EQUAL_WEIGHT_LABEL`
#: を表示言語に訳した文字列を渡す。
SCORE_FORMULA_SCALE_TEMPLATE: Final[str] = (
    "{label} = 100 × ({zero_anchor} - まとめたずれ) ÷ ({zero_anchor} - {full_anchor})"
)


def _score_formula_scale(*, zero_anchor: float, full_anchor: float) -> str:
    """実数値を代入する前の、ステップ 3 の式そのもの。

    アンカーは `domain.score_scale` の版付き定数から組み立てる（UI 側に写しを
    作らない）。
    """
    return t(SCORE_FORMULA_SCALE_TEMPLATE).format(
        label=t(PART_EQUAL_WEIGHT_LABEL),
        zero_anchor=format_distance(zero_anchor),
        full_anchor=format_distance(full_anchor),
    )


def _step_scale_block(explanation: ScoreExplanation) -> str:
    """ステップ 3: アンカー 2 点を明示したうえで、点へ換算する。"""
    zero_anchor = _bare_percent(explanation.distance_at_zero_score)
    full_anchor = _bare_percent(explanation.distance_at_full_score)
    combined = _bare_percent(explanation.combined_distance)
    formula = _score_formula_scale(
        zero_anchor=explanation.distance_at_zero_score,
        full_anchor=explanation.distance_at_full_score,
    )
    # 続く 2 行の `=` を 1 行目の `=` の真下へ置く。ラベルは全角なので、文字数
    # ではなく表示桁で詰める（`_display_width`）。英語表示ではラベルが半角に
    # なり桁数も変わるため、**訳した後のラベル**の表示幅で詰める
    # （`PART_EQUAL_WEIGHT_LABEL` の生の日本語幅では英語表示のとき揃わない）。
    padding = " " * (_display_width(t(PART_EQUAL_WEIGHT_LABEL)) + 1)
    return "\n".join(
        (
            formula,
            f"{padding}= 100 × ({zero_anchor} - {combined}) ÷ ({zero_anchor} - {full_anchor})",
            f"{padding}= {format_score(explanation.combined_distance)}",
        )
    )


def _resolve_explanation(outcome: AnalysisOutcome) -> ScoreExplanation | None:
    """計算過程に使う `ScoreExplanation` を、詳細タブの主表示と同じ値から作る。

    部位別の値は `resolve_part_distance_3d`（このすぐ上で描画済みの部位別
    メトリクスと同じ、平滑値優先）を通す。完備性の判定は
    `resolve_equal_weight_distance`（その時点の生値、§8.6.5・§8.7）で行う。
    `resolve_part_distance_3d` は平滑値が欠測部位を「保持」しうる（H1 改訂）
    ため、部位別の値そのものではなくこの生値ベースの判定をゲートに使う。
    さもないと、今のフレームでは欠測している部位の古い平滑値を使って数値を
    組み立ててしまう（不変条件 11 違反）。
    """
    if resolve_equal_weight_distance(outcome) is None:
        return None
    return explain_score({name: resolve_part_distance_3d(outcome, name) for name in PART_NAMES})


def _render_score_calculation_detail(outcome: AnalysisOutcome) -> None:
    """§5.3/§8.6.3（H4）: 「全体のスコア」の計算過程を 3 ステップで開示する。

    畳んだ `st.expander` の中に、ステップごとの平叙文と実数値の計算を対にして
    置き、そのあとに 4 つの注記（重みの違い・目盛りを広げた理由・実測レンジ・
    アンカー付きの help）を続ける。ステップの中へ注記を混ぜない——手順と、
    手順の読み方に対する留保は別のものである。

    `s_3d` が欠測のときは式の形だけを示し、数値を当てはめない（不変条件 11）。
    """
    with st.expander(
        t(SCORE_CALCULATION_EXPANDER_TITLE_TEMPLATE).format(label=t(PART_EQUAL_WEIGHT_LABEL))
    ):
        explanation = _resolve_explanation(outcome)

        st.markdown(f"**{t(SCORE_STEP_TITLE_DEVIATIONS)}**")
        st.caption(t(SCORE_CALCULATION_DEVIATION_INTRO))
        if explanation is None:
            st.info(t(SCORE_CALCULATION_UNAVAILABLE_NOTICE))
        else:
            st.code(_step_deviations_block(explanation), language="text")

        st.markdown(f"**{t(SCORE_STEP_TITLE_COMBINE)}**")
        st.caption(t(SCORE_CALCULATION_COMBINE_INTRO))
        st.code(
            t(SCORE_FORMULA_COMBINE) if explanation is None else _step_combine_block(explanation),
            language="text",
        )

        st.markdown(f"**{t(SCORE_STEP_TITLE_SCALE)}**")
        st.caption(t(SCORE_CALCULATION_SCALE_INTRO))
        st.code(
            _score_formula_scale(
                zero_anchor=DISTANCE_AT_ZERO_SCORE,
                full_anchor=DISTANCE_AT_FULL_SCORE,
            )
            if explanation is None
            else _step_scale_block(explanation),
            language="text",
        )
        st.caption(
            t(SCORE_CALCULATION_SLOPE_NOTE_TEMPLATE).format(points=POINTS_PER_DEVIATION_PERCENT),
            # §5.3.1（2026-08-16）: ここには 5 本の caption が並んでいた。手順を
            # 読み終えた直後に注記の壁が続くと、手順そのものまで読まれなくなる。
            # 「採点ではない」「順位ではない」「アンカー 2 点」は §8.6.3 の要求
            # なので**内容は削らず**、`help` へまとめて移した。
            help="\n\n".join(
                (
                    t(PART_EQUAL_WEIGHT_HELP),
                    t(SCORE_CALCULATION_HIGH_VALUE_NOTE),
                    t(SCORE_CALCULATION_MEASURED_RANGE_CAPTION),
                )
            ),
        )

        st.divider()
        # 2 つの総合量が一致しない理由だけは常時出す。数字が 2 つ並んでいる場所で
        # 「なぜ違うのか」が読めないと、どちらかが誤りだと受け取られる。
        st.caption(t(SCORE_CALCULATION_WEIGHT_DIFFERENCE_CAPTION))


# --- 状態と姿勢 ---------------------------------------------------------------


#: 値は文字列リテラルのまま持つ（`t(_STATUS_HEADLINE[status])` という辞書
#: 添字解決のため、`tests/ui/test_i18n.py`）。`Status` の全メンバーを網羅する
#: ので、素の添字（`.get()` ではなく）で安全に引ける。
_STATUS_HEADLINE: Final[dict[Status, str]] = {
    Status.OK: "計測中です。",
    Status.HOLD: "撮影条件を調整してください。",
    Status.NO_FACE: "顔が見つかりません。",
    Status.MULTI_FACE: "2 名以上が写っています。",
    Status.ERROR: "このフレームは処理できませんでした。",
}


def render_guidance(outcome: AnalysisOutcome) -> None:
    """スコアを出す前に何を直せばよいかを示す（スコアより先に案内する）。

    §5.3 の操作可能な文言は `ui.guidance.REASON_GUIDANCE` から得る。ライブの
    バナーが読むのと同じ §6.3 の表なので、詳細パネルとバナーが 1 つの条件を
    二通りに説明することはない。品質ゲートへ理由コードを追加したときに、
    ここで黙って何も表示されなくなることも防げる。
    """
    st.write(t(_STATUS_HEADLINE[outcome.state.status]))
    for code in outcome.state.reason_codes:
        guidance = REASON_GUIDANCE.get(code)
        if guidance:
            st.markdown(f"- {guidance}")


def render_tracking_status(
    status: TrackingStatus,
    messages: Mapping[TrackingStatus, str],
) -> None:
    """追跡メッセージを 1 つ、§5.2 がその状態へ割り当てた severity で表示する。

    Camera と Image は同じ状態を別の言葉で表す（静止画に「動かないで」とは
    言えない）ため、``messages`` は呼び出し側が所有する。ただし状態と色の
    対応はセクションごとの選択ではなくプロダクトの規則なので、ここに置いて
    両セクションが食い違わないようにする。
    """
    message = messages[status]
    if status is TrackingStatus.ERROR:
        st.error(message)
    elif status is TrackingStatus.MULTI_FACE:
        st.warning(message)
    elif status is TrackingStatus.SINGLE_FACE:
        st.success(message)
    else:
        st.info(message)


POSE_UNAVAILABLE_CAPTION: Final[str] = "姿勢: 取得できません"
POSE_CAPTION_TEMPLATE: Final[str] = "姿勢 yaw {yaw:.1f}° / pitch {pitch:.1f}° / roll {roll:.1f}°"
POSE_FACE_AREA_RATIO_TEMPLATE: Final[str] = " · 顔面積比 {ratio:.2f}"


def render_pose(outcome: AnalysisOutcome) -> None:
    pose = outcome.pose
    if pose is None:
        st.caption(t(POSE_UNAVAILABLE_CAPTION))
        return
    caption = t(POSE_CAPTION_TEMPLATE).format(
        yaw=pose.yaw_deg, pitch=pose.pitch_deg, roll=pose.roll_deg
    )
    if outcome.face_area_ratio is not None:
        caption += t(POSE_FACE_AREA_RATIO_TEMPLATE).format(ratio=outcome.face_area_ratio)
    st.caption(caption)


# --- 詳細結果 -----------------------------------------------------------------


def _render_part_breakdown(outcome: AnalysisOutcome, *, show_smoothed: bool) -> None:
    """§8.6.5: ずれがどこにあるかを見るための診断的な内訳。

    これらを足しても全体の対称度にはならない。全体は全対応ペアをまとめて 1 回
    だけ折りたたんで求めるためである。3D 一式が欠測なら 4 部位とも `—`。

    §8.7 改訂（H1）: 部位別も全体値と同じく平滑値を主表示し、生値を併記する
    （`resolve_part_distance_3d` が「平滑値を優先し、無ければ生値」を解決する）。
    片方が生値・片方が平滑値という 2 つの表示規則を作らないためである。
    """
    # §5.3.1: 揺れ幅の注記は `help` へ移した。部位別を見ている人だけが必要とする
    # 情報であり、常時表示だと詳細タブの灰色の文がもう 1 本増える。
    st.caption(
        t("部位別の対称度（全体値の内訳ではなく、ずれの分布を見るための独立した値です）"),
        help=t(PART_LEVEL_VARIATION_CAPTION),
    )
    symmetry_3d = outcome.symmetry_3d
    for column, name in zip(st.columns(len(PART_NAMES)), PART_NAMES, strict=True):
        distance = resolve_part_distance_3d(outcome, name)
        column.metric(
            part_label(name),
            format_symmetry(distance),
            help=t(PART_METRIC_DEVIATION_HELP_TEMPLATE).format(deviation=format_distance(distance)),
        )
        if show_smoothed:
            raw = symmetry_3d.distance(name) if symmetry_3d is not None else None
            column.caption(t(RAW_VALUE_CAPTION_TEMPLATE).format(value=format_symmetry(raw)))


PART_METRIC_DEVIATION_HELP_TEMPLATE: Final[str] = "ずれ {deviation}"
RAW_VALUE_CAPTION_TEMPLATE: Final[str] = "生値 {value}"

SKIN_QUALITY_HEADING: Final[str] = "#### 撮影補正後の肌表面参考値（実験値）"
SKIN_QUALITY_METRIC_LABEL: Final[str] = "肌表面参考値"
SKIN_TONE_UNIFORMITY_LABEL: Final[str] = "色調均一性"
SKIN_TEXTURE_UNIFORMITY_LABEL: Final[str] = "テクスチャ均一性"
SKIN_QUALITY_CAPTION: Final[str] = (
    "ソフトウェアでwhite balanceと明度を補正した頬patchの独立参考値です。"
    "camera、照明、化粧、髭、圧縮の影響を残し、肌質、疾患、健康、美しさを表しません。"
)
SKIN_QUALITY_GAIN_TEMPLATE: Final[str] = "補正gain RGB {r:.2f} / {g:.2f} / {b:.2f}"
SKIN_QUALITY_VERSION_TEMPLATE: Final[str] = "skin metric {version} · 未校正"


def _render_skin_quality(outcome: AnalysisOutcome) -> None:
    """§8.6.1: 幾何とは独立した、撮影補正後の肌表面参考値（実験値）。"""
    skin = outcome.skin_quality
    if skin is None:
        return
    st.divider()
    st.markdown(t(SKIN_QUALITY_HEADING))
    st.metric(t(SKIN_QUALITY_METRIC_LABEL), "—" if skin.score is None else f"{skin.score:.1f}")
    tone_column, texture_column = st.columns(2)
    tone_column.metric(
        t(SKIN_TONE_UNIFORMITY_LABEL),
        "—" if skin.tone_uniformity is None else f"{skin.tone_uniformity:.1f}",
    )
    texture_column.metric(
        t(SKIN_TEXTURE_UNIFORMITY_LABEL),
        "—" if skin.texture_uniformity is None else f"{skin.texture_uniformity:.1f}",
    )
    st.caption(t(SKIN_QUALITY_CAPTION))
    if skin.correction is not None:
        gains = skin.correction.gains
        st.caption(t(SKIN_QUALITY_GAIN_TEMPLATE).format(r=gains[0], g=gains[1], b=gains[2]))
    st.caption(t(SKIN_QUALITY_VERSION_TEMPLATE).format(version=skin.metric_version))


#: §5.3/§8.6.3: ずれを併記する 1 行。`{deviation}` は `format_distance` の出力
#: （既に整形済みの文字列）を差し込む。
OVERALL_DEVIATION_CAPTION_TEMPLATE: Final[str] = (
    "全体の対称度のずれ {deviation}（対称度との和は 100%）"
)
RAW_OVERALL_CAPTION_TEMPLATE: Final[str] = "平滑化前の生値 {symmetry}（ずれ {deviation}）"
METRIC_VERSION_CAPTION_TEMPLATE: Final[str] = (
    "metric {metric_3d} · 2D 診断 {metric_2d} · landmark map {landmark_map}"
)


def render_score_details(
    outcome: AnalysisOutcome,
    *,
    show_smoothed: bool = True,
    show_parts: bool = True,
) -> None:
    """総合値そのものを除く、左右差に関するすべてを描画する。

    全体の対称度・全体のスコアの数値は、映像上の HUD（Camera）と画像上の
    吹き出し（Image）が出す。**見出し結果の区画はどちらの入力にも無い**
    （§5.1、2026-08-12）。こちらは吹き出しが持てないもの——ずれの併記・部位別の
    内訳・計算過程・§5.3 の注記・姿勢・肌表面参考値——を、その下の画面幅いっぱい
    の詳細セクションへ描画する（§5.1, FR-013, AC-LAYOUT-01）。**総合値そのものを
    メトリクスとして重複表示しない。**

    ``show_parts`` は表示専用のパラメータで、部位別 4 つのメトリクスと生値の
    併記を描くかどうかを切り替える（項目26、H3）。**Camera** は映像上の HUD が
    既に部位別 4 つを表示するため ``False`` を渡し、詳細タブでの重複を避ける。
    **Image** は HUD を持たないため既定の ``True`` のまま描画する。``False``
    でも `PART_LEVEL_VARIATION_CAPTION`（§5.3 が義務付ける部位別の揺れ幅の
    注記）と `_render_score_calculation_detail`（H4 の計算過程）は描き続ける。
    HUD には注記を置く場所が無く、この注記の唯一の受け皿が詳細タブになる
    ためである。

    §8.6.3 のとおり、部位別は主指標 `symmetry-3d/1.0.0`
    （`outcome.symmetry_3d`）から作る。`outcome.symmetry`（`symmetry-2d/2.0.0`）
    は §8.6.6 の構造的な欠陥を持つため画面には出さず、エクスポートの
    `distance_*_raw` としてのみ残る。
    """
    st.subheader(t(SCORE_HEADING), help=t(DISTANCE_UNIT_HELP))
    # §5.3.1: 見出し直下の注意は 1 行。「試験中の指標」であることは不変条件 8 が
    # 求めるが、以前の 2 文はその下に並ぶ 4 本の注意と合わせて灰色の壁になっていた。
    st.caption(t("試験中の指標です。") + t(DISTANCE_UNIT_CAPTION))

    # 何か描画できるものがあるか（= 品質ゲートが開いているか）は 2D の有無で
    # 判定する。2D は品質ゲート直後に無条件で算出されるのに対し、3D は `z` の
    # 有無にも依存し独立に欠測しうる（不変条件 12 の失敗分離）。したがって
    # 2D の有無のほうが「解析そのものが成立しているか」の安定した代理になる。
    symmetry = outcome.symmetry
    if symmetry is None or symmetry.overall is None:
        render_guidance(outcome)
        render_pose(outcome)
        return

    # §5.3/§8.6.3: ずれを必ず併記する。数値をどちらの向きからも読めるようにし、
    # 「100%」が達成された目標のように受け取られないようにするためである。
    # 見出し結果の削除（2026-08-12）で吹き出しだけが総合値を出すようになったが、
    # 吹き出しには併記を置く場所が無い。この行がその唯一の受け皿である。
    st.caption(
        t(OVERALL_DEVIATION_CAPTION_TEMPLATE).format(
            deviation=format_distance(resolve_overall_distance(outcome))
        )
    )

    symmetry_3d = outcome.symmetry_3d
    if show_smoothed:
        overall_3d_raw = None if symmetry_3d is None else symmetry_3d.overall_distance
        st.caption(
            t(RAW_OVERALL_CAPTION_TEMPLATE).format(
                symmetry=format_symmetry(overall_3d_raw),
                deviation=format_distance(overall_3d_raw),
            )
        )

    if show_parts:
        _render_part_breakdown(outcome, show_smoothed=show_smoothed)

    _render_score_calculation_detail(outcome)

    render_pose(outcome)
    _render_skin_quality(outcome)

    _render_reading_limits()

    # 版と対応表は来歴であって留保ではない（FR-020）。上の箱へ混ぜず、
    # セクションの締めとして分けて置く。
    metric_3d_label = symmetry_3d.metric_version if symmetry_3d is not None else "—"
    st.caption(
        t(METRIC_VERSION_CAPTION_TEMPLATE).format(
            metric_3d=metric_3d_label,
            metric_2d=outcome.metric_version,
            landmark_map=outcome.landmark_map_version,
        )
    )


def _render_reading_limits() -> None:
    """§5.3.1（2026-08-16）: 留保は 1 行にする。

    以前は見出し付きの箱に 3 本の長文を並べ、その上にさらに 2 本の caption が
    あった。**読まれない注意文は、書いていないのと同じである。** 根拠と全文は
    About が持ち、ここからは `help` で到達させる。
    """
    st.caption(t(READING_LIMITS_NOTE), help=t(SCORE_SCALE_HELP))


# --- 参考年齢・参考性別 -------------------------------------------------------

#: §10.8: 参考年齢・参考性別の区画に必ず添える来歴の 1 行。**Camera と Image で
#: 同じ文字列を使う**（2026-08-26、§5.1）——同じモデルの同じ推論であり、モードで
#: 言い方が変わる理由が無い。
DEMOGRAPHIC_RUNTIME_CAPTION: Final[str] = (
    "runtime: OpenCV DNN CPU · Open Model Zoo 2023.0 FP16由来ONNX / 対象18〜75歳"
)


DEMOGRAPHIC_HEADING: Final[str] = "モデル推定の参考年齢・参考性別"
REFERENCE_AGE_LABEL: Final[str] = "参考年齢"
REFERENCE_AGE_VALUE_TEMPLATE: Final[str] = "{age:.1f}歳"
REFERENCE_GENDER_LABEL: Final[str] = "外見由来の参考性別"
GENDER_CONFIDENCE_CAPTION_TEMPLATE: Final[str] = (
    "性別モデル確率 {confidence:.1%}・対象18〜75歳。"
    "年齢や本人の自己認識・法的性別を断定せず、本人確認や人物評価には使用できません。"
)
GENDER_PROBABILITIES_CAPTION_TEMPLATE: Final[str] = (
    "モデル出力 female {female:.3f} / male {male:.3f}"
)
DEMOGRAPHIC_CACHED_CAPTION: Final[str] = "同一区間内の直近推論を再表示しています。"
DEMOGRAPHIC_MODEL_INFO_EXPANDER_TITLE: Final[str] = "モデル情報"


def render_demographic_estimate(estimate: DemographicEstimate) -> None:
    """§10.8 の年齢・性別参考パネルを描画する（Image と Camera で共有）。

    **2 クラス確率の棒グラフは持たない**（2026-08-26）。以前は Image だけが描き、
    Camera は 5 Hz の再描画で費用が見合わないため切っていた——同じ推定が入力
    モードによって別の見た目になっていた。同じ 2 つの数はすぐ下にテキストで
    出ており、図にすると参考値が左右差の結果より目立つ（§5.2）。
    """
    st.subheader(t(DEMOGRAPHIC_HEADING))
    age_column, gender_column = st.columns(2)
    age_column.metric(
        t(REFERENCE_AGE_LABEL),
        t(REFERENCE_AGE_VALUE_TEMPLATE).format(age=estimate.estimated_age_years),
    )
    gender_column.metric(t(REFERENCE_GENDER_LABEL), gender_label(estimate.gender))
    st.caption(t(GENDER_CONFIDENCE_CAPTION_TEMPLATE).format(confidence=estimate.gender_confidence))
    female, male = estimate.gender_probabilities
    st.caption(t(GENDER_PROBABILITIES_CAPTION_TEMPLATE).format(female=female, male=male))
    if estimate.is_cached:
        st.caption(t(DEMOGRAPHIC_CACHED_CAPTION))
    with st.expander(t(DEMOGRAPHIC_MODEL_INFO_EXPANDER_TITLE)):
        st.code(
            f"version: {estimate.model_version}\n"
            f"sha256: {estimate.model_sha256}\n"
            f"observed_at: {estimate.observed_at.isoformat()}",
            language="text",
        )
