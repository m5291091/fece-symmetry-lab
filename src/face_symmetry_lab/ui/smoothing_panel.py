"""FR-040 / FR-041: リアルタイム表示の落ち着きを選ぶ UI。

設定は 2 つあり、**効くところが違う**。

* 平均する時間（`tau`、§8.7）—— 数値が動く **量** を決める
* 数値を入れ替える間隔（`display_hold_seconds`、ADR-0008）—— 動く **頻度** を決める

`tau` を伸ばしても、平滑値は連続時間のフィルタなので新しいサンプルのたびに必ず
動く。読んでいる最中に数字が止まっていてほしいという要求は前者だけでは満たせない
ため、2 つを同じ区画に並べてある。

**この module は pipeline を知らない。** 受け取るのは「いま有効な `tau`」「いまの
解析周期」「いま有効な保持時間」だけで、選ばれた値を返す。適用（EMA の差し替え・
区間の切り替え・ラッチの設定）は呼び出し側（`ui.camera`）が pipeline のメソッドで
行う。こう分けておくと、この panel は Streamlit だけで単体検証でき、pipeline の
生成を必要としない。

**表示する導出値はすべて `domain.smoothing` から取る。** 追従時間と実効サンプル数
はどちらも §8.7 の式の逆算であり、ここで近似式（`3 × tau` など）を書くと、画面の
説明だけが実装から静かにずれる（ADR-0007 (c)）。

**推奨値を置かない**（§5.3）。この設定に正解は無く、示すのは交換関係だけである。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

import streamlit as st

from face_symmetry_lab.domain.display_hold import (
    DEFAULT_DISPLAY_HOLD_SECONDS,
    DISPLAY_HOLD_CHOICES_SECONDS,
)
from face_symmetry_lab.domain.smoothing import (
    RESPONSE_FRACTION_63,
    RESPONSE_FRACTION_95,
    TAU_CHOICES_SECONDS,
    effective_sample_count,
    nearest_tau_choice,
    response_time_seconds,
)
from face_symmetry_lab.ui.i18n import t

#: 選択値は session state が持つ。`ui.camera` の他の表示設定と同じ命名にそろえる。
SMOOTHING_STATE_KEY: Final[str] = "camera_smoothing_tau_seconds"

#: **区画の見出しはこの panel が持つ。** 呼び出し側（`ui.camera`）が別の見出しを
#: 重ねると、同じ区画に見出しが 2 行並ぶ（実際に一度そうなった）。他の設定区画と
#: 同じ `#####` の高さにそろえてある。
SMOOTHING_HEADING: Final[str] = "##### 数値の落ち着き（移動平均の時間）"

#: 「何が変わらないか」を最初に書く。この設定を見た利用者がまず疑うのは
#: 「数値そのものを自分で動かしているのではないか」であり、そこを曖昧にしたまま
#: 追従時間だけを説明すると、測定の調整だと読まれる（§8.6、不変条件 10）。
#: §5.3.1: 常時表示は 1 行。ここは残す——この設定を見た利用者がまず疑うのは
#: 「数値そのものを自分で動かしているのではないか」であり、そこを曖昧にすると
#: 測定の調整だと読まれる（§8.6、不変条件 10）。
SMOOTHING_SCOPE_NOTE: Final[str] = "変わるのは画面の数値が動く速さだけで、測定値は変わりません。"

SMOOTHING_TRADEOFF_NOTE: Final[str] = (
    "短いほど反応が速くぶれも大きく、長いほど落ち着きますが追随が遅れます。"
)

#: §5.3.1: `help=` へ移した。選び直したときに何が起きるかは、選ぶ人がその場で
#: 開ければよい。
SMOOTHING_RESET_NOTE: Final[str] = (
    "選び直すと、それまでの平均をいったん捨てて新しい時間で測り直します。"
    "エクスポートも新しい区間として記録され、映像上のグラフは描き直しになります。"
)

SMOOTHING_HELP: Final[str] = (
    "表示中の数値は、直近の解析結果を時間で重み付けして平均したものです（時間ベースEMA）。"
    "ここで選ぶのはその平均の時間の長さで、測定そのものの精度や計算式は変わりません。"
)

#: FR-041: 更新間隔の設定。平滑化（量）とは別に、数値が入れ替わる頻度を決める。
HOLD_STATE_KEY: Final[str] = "camera_display_hold_seconds"

#: 2 つ目の設定の widget ラベル。**見出しを別に立てない。** 立てると同じ文字列が
#: 2 行続く（`SMOOTHING_HEADING` で一度やった失敗の繰り返し）。区切りは
#: `st.divider()` が担う。
HOLD_LABEL: Final[str] = "数値を入れ替える間隔"

HOLD_HELP: Final[str] = (
    "画面の数値を、選んだ時間のあいだ入れ替えずに置きます。"
    "測定・記録・エクスポートはこの設定に関わらず今までどおり続きます。"
)

#: 「平滑化を伸ばしたのに数字が止まらない」という行き止まりを説明する。この 1 文が
#: 無いと、2 つの設定が同じことをする 2 通りのやり方に見える。
HOLD_EFFECT_NOTE: Final[str] = (
    "平均する時間を伸ばしても動く回数は変わりません。数字を止めたいときはこちらを長くします。"
)

#: AC-HOLD-02 の挙動を先に伝える。止まっている数値が「いつまで有効か」を利用者が
#: 判断できるようにするための文である。
#: `help=` へ移す。ただし内容は削らない——止まっている数値がいつまで有効かは、
#: この設定を使う人が必ず知る必要がある（AC-HOLD-02 の挙動そのもの）。
HOLD_MISSING_NOTE: Final[str] = (
    "顔が外れたときや計測条件を満たさなくなったときは、この間隔を待たずにすぐ「—」へ戻ります。"
    "止まっている数値が、すでに測れていない状態のまま残ることはありません。"
)

#: `st.select_slider` の options は表示文字列そのものにする。`format_func` を
#: 通さないことで、session state に入る値と画面のラベルが一致し、目盛りの読み
#: 違いが起きない。
#: **FR-043: これらは識別子であり翻訳しない**（ADR-0010 (c)）。言語を切り替えた
#: 瞬間に保存済みの選択が options から外れて設定が失われるためである。表示だけ
#: を `_tau_choice_label` / `_hold_choice_label`（`format_func`）が切り替える。
_CHOICE_LABELS: Final[tuple[str, ...]] = tuple(f"{tau:.1f} 秒" for tau in TAU_CHOICES_SECONDS)
_LABEL_TO_TAU: Final[dict[str, float]] = dict(zip(_CHOICE_LABELS, TAU_CHOICES_SECONDS, strict=True))

#: 0 秒だけは「毎回」と言葉で書く。「0.0 秒ごとに更新」は意味が取れない。
#: この文字列自体は識別子なので翻訳しない（上と同じ理由）。
_HOLD_LABEL_EVERY_FRAME: Final[str] = "毎回"
_HOLD_LABELS: Final[tuple[str, ...]] = tuple(
    _HOLD_LABEL_EVERY_FRAME if hold <= 0.0 else f"{hold:.1f} 秒"
    for hold in DISPLAY_HOLD_CHOICES_SECONDS
)
_LABEL_TO_HOLD: Final[dict[str, float]] = dict(
    zip(_HOLD_LABELS, DISPLAY_HOLD_CHOICES_SECONDS, strict=True)
)

#: FR-043: 表示だけを翻訳するテンプレート（ADR-0010 (a)）。`_CHOICE_LABELS` /
#: `_HOLD_LABELS` の値そのものは上記のとおり識別子のまま変えない。
_SECONDS_VALUE_TEMPLATE: Final[str] = "{value:.1f} 秒"
_SAMPLE_COUNT_VALUE_TEMPLATE: Final[str] = "{value:.1f} 個"
_SAMPLE_COUNT_HELP_TEMPLATE: Final[str] = (
    "いまの解析周期（{interval_ms} ms）での目安です。"
    "解析周期はAIが自動調整するため、同じ設定でもこの本数は動きます。"
)


@dataclass(frozen=True, slots=True)
class StabilityChoice:
    """この panel が返す 2 つの設定。呼び出し側が pipeline へ適用する。"""

    tau_seconds: float
    hold_seconds: float


def _label_for(tau_seconds: float) -> str:
    return f"{tau_seconds:.1f} 秒"


def _hold_label_for(hold_seconds: float) -> str:
    return _HOLD_LABEL_EVERY_FRAME if hold_seconds <= 0.0 else f"{hold_seconds:.1f} 秒"


def nearest_hold_choice(hold_seconds: float | None) -> float:
    """段の外の保持時間を、最も近い段へ寄せる（`nearest_tau_choice` と同じ役割）。"""
    if hold_seconds is None or not isfinite(hold_seconds):
        return DEFAULT_DISPLAY_HOLD_SECONDS
    return min(DISPLAY_HOLD_CHOICES_SECONDS, key=lambda choice: abs(choice - hold_seconds))


def _tau_choice_label(label: str) -> str:
    """`_CHOICE_LABELS` の値（識別子）はそのまま、`st.select_slider` の表示だけを
    翻訳する（ADR-0010 (c)、`ui.camera._background_mode_label` と同じ形）。
    """
    return t(_SECONDS_VALUE_TEMPLATE).format(value=_LABEL_TO_TAU[label])


def _hold_choice_label(label: str) -> str:
    """`_HOLD_LABELS` も同じ理由で値は識別子のまま保つ。「毎回」だけは言葉なので
    翻訳する。
    """
    if label == _HOLD_LABEL_EVERY_FRAME:
        return t(_HOLD_LABEL_EVERY_FRAME)
    return t(_SECONDS_VALUE_TEMPLATE).format(value=_LABEL_TO_HOLD[label])


def render_smoothing_control(
    *,
    tau_seconds: float | None,
    analysis_interval_ms: int,
    hold_seconds: float | None = None,
) -> StabilityChoice:
    """表示の落ち着きに関する 2 つの設定を描画し、選ばれた値を返す。

    ``tau_seconds`` / ``hold_seconds`` は pipeline がいま実際に使っている値
    （`tau_seconds=None` は平滑化を持たない pipeline）。初回だけこの値で選択を
    初期化し、以後は widget が持つ。段の外の値は `nearest_tau_choice` /
    `nearest_hold_choice` で最も近い段へ寄せる——`st.select_slider` は options に
    無い値を渡されると例外になるため、ここを通さないと過去の設定で画面が落ちる。

    **2 つを同じ区画に置くのは、両者が別々のことをするからである**（FR-040 /
    FR-041、ADR-0008）。平滑化は数値が動く**量**を、更新間隔は動く**頻度**を
    決める。片方だけを見せると、「落ち着かせたいのに落ち着かない」——`tau` を
    伸ばしても 5Hz で小数が揺れ続ける——という行き止まりに利用者を置くことになる。
    """
    if SMOOTHING_STATE_KEY not in st.session_state:
        st.session_state[SMOOTHING_STATE_KEY] = _label_for(nearest_tau_choice(tau_seconds))
    if HOLD_STATE_KEY not in st.session_state:
        st.session_state[HOLD_STATE_KEY] = _hold_label_for(nearest_hold_choice(hold_seconds))

    st.markdown(t(SMOOTHING_HEADING))
    selected_label = st.select_slider(
        t("平均する時間"),
        options=_CHOICE_LABELS,
        key=SMOOTHING_STATE_KEY,
        format_func=_tau_choice_label,
        help=t(SMOOTHING_HELP),
    )
    selected = _LABEL_TO_TAU[str(selected_label)]

    # 選択の意味を数値で開示する。3 つとも §8.7 の式から導かれるだけの量で、
    # 新しい定数を持たない（ADR-0007 (c)）。`delta` は使わない——上下の矢印は
    # 設定値に良し悪しがあるかのように読める（§5.2）。
    quick, settled, samples = st.columns(3)
    quick.metric(
        t("63% 追いつくまで"),
        t(_SECONDS_VALUE_TEMPLATE).format(
            value=response_time_seconds(selected, fraction=RESPONSE_FRACTION_63)
        ),
        help=t("表示が新しい水準へ 63% 近づくまでの時間です（この値が時定数そのものです）。"),
    )
    settled.metric(
        t("95% 追いつくまで"),
        t(_SECONDS_VALUE_TEMPLATE).format(
            value=response_time_seconds(selected, fraction=RESPONSE_FRACTION_95)
        ),
        help=t("ほぼ追いついたと言える時間です。時定数のおよそ 3.0 倍になります。"),
    )
    samples.metric(
        t("均している解析サンプル数"),
        t(_SAMPLE_COUNT_VALUE_TEMPLATE).format(
            value=effective_sample_count(selected, interval_ms=analysis_interval_ms)
        ),
        help=t(_SAMPLE_COUNT_HELP_TEMPLATE).format(interval_ms=analysis_interval_ms),
    )

    st.caption(t(SMOOTHING_TRADEOFF_NOTE))

    # FR-041: ここからが「頻度」の設定。平滑化とは効き方が違うので区切りを入れる
    # が、見出しは足さない——widget ラベルと同じ文字列が 2 行続くだけになる。
    # 同じ `select_slider` の形にそろえてあるのは、どちらも「段から選ぶ」設定
    # だからである。
    st.divider()
    selected_hold_label = st.select_slider(
        t(HOLD_LABEL),
        options=_HOLD_LABELS,
        key=HOLD_STATE_KEY,
        format_func=_hold_choice_label,
        help=f"{t(HOLD_HELP)}\n\n{t(HOLD_MISSING_NOTE)}",
    )
    selected_hold = _LABEL_TO_HOLD[str(selected_hold_label)]
    st.caption(t(HOLD_EFFECT_NOTE))

    # §5.3.1（2026-08-16）: 常時出す注意は 1 行。リセットの挙動と「止まった数値が
    # いつまで有効か」は `help` へ移した——内容は削っていない。
    st.caption(t(SMOOTHING_SCOPE_NOTE), help=f"{t(SMOOTHING_RESET_NOTE)}\n\n{t(HOLD_MISSING_NOTE)}")
    return StabilityChoice(tau_seconds=selected, hold_seconds=selected_hold)
