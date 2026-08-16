"""FR-040: 表示平滑化時間（§8.7 の `tau`）を選ぶ UI。

**この module は pipeline を知らない。** 受け取るのは「いま有効な `tau`」と
「いまの解析周期」という 2 つの値だけで、選ばれた `tau` を返す。適用（EMA の
差し替えと区間の切り替え）は呼び出し側（`ui.camera`）が
`RealtimeLandmarkPipeline.set_smoothing_tau_seconds` で行う。こう分けておくと、
この panel は Streamlit だけで単体検証でき、pipeline の生成を必要としない。

**表示する導出値はすべて `domain.smoothing` から取る。** 追従時間と実効サンプル数
はどちらも §8.7 の式の逆算であり、ここで近似式（`3 × tau` など）を書くと、画面の
説明だけが実装から静かにずれる（ADR-0007 (c)）。

**推奨値を置かない**（§5.3）。この設定に正解は無く、示すのは交換関係だけである。
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from face_symmetry_lab.domain.smoothing import (
    RESPONSE_FRACTION_63,
    RESPONSE_FRACTION_95,
    TAU_CHOICES_SECONDS,
    effective_sample_count,
    nearest_tau_choice,
    response_time_seconds,
)

#: 選択値は session state が持つ。`ui.camera` の他の表示設定と同じ命名にそろえる。
SMOOTHING_STATE_KEY: Final[str] = "camera_smoothing_tau_seconds"

#: **区画の見出しはこの panel が持つ。** 呼び出し側（`ui.camera`）が別の見出しを
#: 重ねると、同じ区画に見出しが 2 行並ぶ（実際に一度そうなった）。他の設定区画と
#: 同じ `#####` の高さにそろえてある。
SMOOTHING_HEADING: Final[str] = "##### 数値の落ち着き（移動平均の時間）"

#: 「何が変わらないか」を最初に書く。この設定を見た利用者がまず疑うのは
#: 「数値そのものを自分で動かしているのではないか」であり、そこを曖昧にしたまま
#: 追従時間だけを説明すると、測定の調整だと読まれる（§8.6、不変条件 10）。
SMOOTHING_SCOPE_NOTE: Final[str] = (
    "変わるのは画面の数値が動く速さだけです。左右差の計算式、部位別の生値、"
    "全体のスコア、エクスポートの生値はこの設定では変わりません。"
)

SMOOTHING_TRADEOFF_NOTE: Final[str] = (
    "短くすると動きへの反応が速くなり、そのぶん表示のぶれも大きく見えます。"
    "長くすると表示は落ち着きますが、実際の変化に追いつくまでの遅れが大きくなります。"
)

SMOOTHING_RESET_NOTE: Final[str] = (
    "選び直すと、それまでの平均をいったん捨てて新しい時間で測り直します。"
    "エクスポートも新しい区間として記録され、映像上のグラフは描き直しになります。"
)

SMOOTHING_HELP: Final[str] = (
    "表示中の数値は、直近の解析結果を時間で重み付けして平均したものです（時間ベースEMA）。"
    "ここで選ぶのはその平均の時間の長さで、測定そのものの精度や計算式は変わりません。"
)

#: `st.select_slider` の options は表示文字列そのものにする。`format_func` を
#: 通さないことで、session state に入る値と画面のラベルが一致し、目盛りの読み
#: 違いが起きない。
_CHOICE_LABELS: Final[tuple[str, ...]] = tuple(f"{tau:.1f} 秒" for tau in TAU_CHOICES_SECONDS)
_LABEL_TO_TAU: Final[dict[str, float]] = dict(zip(_CHOICE_LABELS, TAU_CHOICES_SECONDS, strict=True))


def _label_for(tau_seconds: float) -> str:
    return f"{tau_seconds:.1f} 秒"


def render_smoothing_control(
    *,
    tau_seconds: float | None,
    analysis_interval_ms: int,
) -> float:
    """平滑化時間の選択とその意味を描画し、選ばれた `tau`（秒）を返す。

    ``tau_seconds`` は pipeline がいま実際に使っている値（`None` は平滑化を持たない
    pipeline）。初回だけこの値で選択を初期化し、以後は widget が持つ。段の外の値は
    `nearest_tau_choice` で最も近い段へ寄せる——`st.select_slider` は options に無い
    値を渡されると例外になるため、ここを通さないと過去の設定で画面が落ちる。
    """
    if SMOOTHING_STATE_KEY not in st.session_state:
        st.session_state[SMOOTHING_STATE_KEY] = _label_for(nearest_tau_choice(tau_seconds))

    st.markdown(SMOOTHING_HEADING)
    selected_label = st.select_slider(
        "平均する時間",
        options=_CHOICE_LABELS,
        key=SMOOTHING_STATE_KEY,
        help=SMOOTHING_HELP,
    )
    selected = _LABEL_TO_TAU[str(selected_label)]

    # 選択の意味を数値で開示する。3 つとも §8.7 の式から導かれるだけの量で、
    # 新しい定数を持たない（ADR-0007 (c)）。`delta` は使わない——上下の矢印は
    # 設定値に良し悪しがあるかのように読める（§5.2）。
    quick, settled, samples = st.columns(3)
    quick.metric(
        "63% 追いつくまで",
        f"{response_time_seconds(selected, fraction=RESPONSE_FRACTION_63):.1f} 秒",
        help="表示が新しい水準へ 63% 近づくまでの時間です（この値が時定数そのものです）。",
    )
    settled.metric(
        "95% 追いつくまで",
        f"{response_time_seconds(selected, fraction=RESPONSE_FRACTION_95):.1f} 秒",
        help="ほぼ追いついたと言える時間です。時定数のおよそ 3.0 倍になります。",
    )
    samples.metric(
        "均している解析サンプル数",
        f"{effective_sample_count(selected, interval_ms=analysis_interval_ms):.1f} 個",
        help=(
            f"いまの解析周期（{analysis_interval_ms} ms）での目安です。"
            "解析周期はAIが自動調整するため、同じ設定でもこの本数は動きます。"
        ),
    )

    st.caption(SMOOTHING_TRADEOFF_NOTE)
    st.caption(SMOOTHING_SCOPE_NOTE)
    st.caption(SMOOTHING_RESET_NOTE)
    return selected
