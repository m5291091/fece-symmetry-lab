"""映像へ重ねる HUD（§14.3e）の組み立てと配線。

`ui.camera` から分けたのは、HUD が「ブラウザ側の資産」と「その資産へ値を渡す
薄い Python」で構成されており、カメラ画面の合成処理とは変更理由が異なるため
である。**その資産（CSS・実行時 JS）はさらに `ui.hud_assets` へ分けてある。**
ここに残るのは payload の組み立て、隠し要素への直列化、bridge の設置だけで、
幾何計算も推論も持たない。

値の整形は必ず `ui.readings` の共通フォーマッタを通す。HUD がパネルとは別の
表示規則へ分岐すると、同じ測定値が画面上で二通りに見えてしまう。

H1（2026-08-11）で HUD の内容を全体・スコア・部位別 4 つ・参考値へ拡張し、
直近の推移を示す折れ線グラフを追加した。H3（2026-08-12）で映像の隣にあった
「ライブ解析」パネルを削除し、Camera で同じ内容を表示するのはこの HUD だけに
一本化した。グラフの移動平均も `FrameAnalyzer` 側の §8.7 改訂で計算済みの値を
運ぶだけであり、HUD（JS）側では一切平滑化しない。同じ理由——HUD が独自の
2 つ目の表示規則へ分岐しない——による。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

import streamlit as st
import streamlit.components.v1 as components

from face_symmetry_lab.application.realtime import PipelineSnapshot
from face_symmetry_lab.application.reference_baseline import BaselineSummary
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.reference_baseline import (
    OVERALL_METRIC,
    PART_METRICS,
    SCORE_METRIC,
    baseline_reading,
)
from face_symmetry_lab.ui.age_gate_notice import resolve_reference_age_view
from face_symmetry_lab.ui.baseline import resolve_baseline
from face_symmetry_lab.ui.hud_assets import (
    HUD_RUNTIME_JS,
    HUD_STYLESHEET,
    VIDEO_CHROME_CSS,
)
from face_symmetry_lab.ui.i18n import t
from face_symmetry_lab.ui.readings import (
    GENDER_LABELS,
    baseline_note,
    format_baseline_score,
    format_baseline_symmetry,
    format_display_hold,
    format_score,
    format_smoothing_window,
    format_symmetry,
    gate_color,
    gate_guidance,
    gate_label,
    resolve_equal_weight_distance,
    resolve_overall_distance,
    resolve_part_distance_3d,
    score_points_value,
    symmetry_percent,
)

#: §14.3e/FR-032: HUD はパネル配置と同じくプリセットで移動し、ドラッグ位置は
#: 補助的に保持するだけである。"非表示" は何も描画しない。
HUD_HIDDEN: Final[str] = "非表示"
HUD_POSITION_OPTIONS: Final[tuple[str, ...]] = ("左上", "右上", "左下", "右下", HUD_HIDDEN)
HUD_POSITION_STATE_KEY = "camera_hud_position"

#: 各プリセットが HUD を位置指定コンテキスト内のどこへ固定するか。
HUD_PLACEMENTS: Final[dict[str, str]] = {
    "左上": "top:24px; left:24px;",
    "右上": "top:24px; right:24px;",
    "左下": "bottom:24px; left:24px;",
    "右下": "bottom:24px; right:24px;",
}

#: §14.3e: 全画面帯の文言。ページの装飾を畳んでいる間に表示される唯一の文章で
#: あり、現在の状態と戻り方の両方を担う。テストで文言を検証できるよう定数に
#: 切り出している。
FULLSCREEN_BADGE: Final[str] = "全画面表示中 — 右上の「終了」で戻ります"
FULLSCREEN_HINT: Final[str] = (
    "画面下の「START」を押すとカメラが始まります。"
    "ブラウザの F11 と併用すると、タブやアドレスバーも隠れた完全な全画面になります。"
)

#: 5 Hz の fragment が書き込み、bridge script が読み取る隠し要素。値は JSON
#: テキストとして渡す。Streamlit は markdown 内の HTML をサニタイズするため、
#: そのまま生き残ることが保証されるのはテキスト内容だけである。
HUD_DATA_ELEMENT_ID: Final[str] = "fsl-hud-data"

#: bridge 側の DOM だけで完結する更新間隔。パネルと同じ周期だが、サーバーへの
#: 往復が無く、隠し要素を読み直すだけで済む。
HUD_REFRESH_MS: Final[int] = 200

#: H1: 折れ線グラフの既定窓。直近 60 秒・5Hz（`HUD_REFRESH_MS`）で 300 点。
#: page realm の in-memory リングバッファ（`window.__fslHudSeries`）がこの
#: 本数までしか保持しないことで、5Hz の payload が現在値だけを運ぶ設計と
#: 対になる（系列全体を毎回送らない）。
HUD_CHART_WINDOW_SECONDS: Final[int] = 60
HUD_CHART_CAPACITY: Final[int] = HUD_CHART_WINDOW_SECONDS * (1000 // HUD_REFRESH_MS)

#: H1: y 軸の固定窓。実測値は 96〜99% の狭い範囲に集まる
#: （`docs/symmetry-metric-measurements.md` §12.1）。0〜100% に固定すると
#: 5 本が一直線になって何も読めず、自動スケールにするとノイズが誇張されて
#: 「大きく動いた」と誤読される。範囲外は clamp して端に貼り付け、軸の数値
#: （95%・100%）をグラフへ明示することで縦軸の幅の誤解を避ける。
HUD_CHART_Y_MIN: Final[float] = 95.0
HUD_CHART_Y_MAX: Final[float] = 100.0

#: H1: inline SVG の viewBox（px 相当）。骨格を組み立てる JS 側と、`points`
#: 属性を計算する JS 側が同じ値を参照できるよう、bridge の設定 (`CFG`) 経由で
#: 両方へ渡す。
HUD_CHART_WIDTH_PX: Final[int] = 220
HUD_CHART_HEIGHT_PX: Final[int] = 64

#: FR-043: `demo` 行はもともと f-string だった。`t()` は f-string を受け付けない
#: ため（鍵が実行のたび変わり永久に訳せない）、テンプレートへ切り出してから
#: `t()` に通し、その後で `.format()` する。
_HUD_AGE_TEMPLATE: Final[str] = "{age:.0f} 歳"
_HUD_DEMOGRAPHIC_TEMPLATE: Final[str] = "参考年齢 {age} · 参考性別 {gender}"


def build_baseline_payload(
    *,
    baseline: BaselineSummary | None,
    overall: float | None,
    equal_weight: float | None,
    parts: Mapping[str, float | None],
) -> dict[str, object]:
    """FR-039: 各測定値へ添える「平均（差）」の表示文字列。

    純粋関数で、`build_hud_payload` と同じく Camera・Image が共有する。**比べる相手の
    値は、表示に使っているのと同じ変換を通してから引く**（対称度は `symmetry_percent`、
    点は `score_points_value`）。ここで別の変換を挟むと、画面に出ている数と括弧の中の
    差が食い違う。

    `baseline` が `None` のときは全項目が空文字になる。空文字の要素は CSS の `:empty`
    で畳まれるので、併記が無いときに空行が残らない。
    """
    if baseline is None:
        return {
            "symmetry": "",
            "score": "",
            "parts": {name: "" for name in PART_NAMES},
            "note": "",
        }
    metrics = baseline.metrics
    return {
        "symmetry": format_baseline_symmetry(
            baseline_reading(metrics, OVERALL_METRIC, value=symmetry_percent(overall))
        ),
        "score": format_baseline_score(
            baseline_reading(metrics, SCORE_METRIC, value=score_points_value(equal_weight))
        ),
        "parts": {
            name: format_baseline_symmetry(
                baseline_reading(
                    metrics,
                    PART_METRICS[name],
                    value=symmetry_percent(parts.get(name)),
                )
            )
            for name in PART_NAMES
        },
        "note": baseline_note(age=baseline.age, gender=baseline.gender, count=baseline.count),
    }


def build_hud_payload(
    *,
    overall: float | None,
    equal_weight: float | None,
    parts: Mapping[str, float | None],
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
    position: str,
    generation: str,
    notice: str = "",
    baseline: BaselineSummary | None = None,
    state: str = "",
    smoothing: str = "",
    hold: str = "",
    guidance: str = "",
) -> dict[str, object]:
    """§14.3e の HUD が表示する値（H1 で内容を拡張し、H3 で映像隣の
    「ライブ解析」パネルを削除して表示をこの HUD へ一本化した）。

    純粋関数で、Streamlit も markup も副作用も持たない。数値はすべてパネルと
    同じフォーマッタ（`format_symmetry`、`GENDER_LABELS`）を通す。HUD が同じ
    測定値に対する 2 つ目の表示規則へ分岐しないようにするためである。欠測は
    古い値や 0 ではなく `—` と表示する（NFR-013）。

    折れ線グラフ用に、同じ変換を数値のまま返す ``chart`` も持たせる
    （`symmetry_percent` 経由。テキストとグラフが別の変換へ分岐しないように
    する）。``generation`` はリングバッファの世代識別子（``gen``）としてその
    まま運ばれ、page realm 側の JS がこれを比較して古い系列を破棄する
    （§3.4・§8.8）。

    ``demo`` は推定が無いとき、または利用者が年齢・性別パネルを切ったときに
    `—` ではなく空文字にする。空行は「機能が動いていて失敗している」と
    受け取られてしまうためである（FR-024）。

    ``notice``（FR-038）は、``show_demographics`` が偽のときにだけ ``demo`` の
    代わりに使う固定文言（`ui.age_gate_notice`）。年齢ゲートが参考年齢の表示を
    止めた場合と、利用者が単に年齢・性別パネルを切った場合を区別するためで、
    後者では ``notice`` を渡さない（空文字のまま何も出さない）。

    ``state``（§5.2/§5.4）は品質ゲート状態のテキスト、``smoothing``（FR-040）は
    いま有効な平滑化時間、``hold``（FR-041）は表示の更新間隔のテキストで、いずれも
    `ui.readings` の整形を通したものを受け取るだけである。**保持中であることを
    画面から読めるようにするために `hold` を出す**——数値が止まって見えるのが設定
    によるものか測定が止まったのかを、利用者が区別できる必要がある。

    3 つとも**空文字が既定で、`—` ではない**——併記の不在は測定の欠測ではない
    ため（`demo` 行と同じ扱い）、CSS の `:empty` が行ごと畳む。いずれも測定値の
    整形（``symmetry``/``score``/``parts``/``chart``）へは一切混ざらない。

    ``baseline``（FR-039）は同じ参考年齢・参考性別の母集団の要約。**参考年齢・
    参考性別を表示していないときは渡してはならない**（§10.12-3）——母集団の年齢から
    年齢ゲートの内部判断を復元できてしまうためである。渡さなければ ``baseline``
    の各項目は空文字になり、CSS の `:empty` が行ごと畳む。

    ``guidance``（FR-031）は品質ゲートが算出済みの主要因 1 件の行動可能な指示文
    （`ui.readings.gate_guidance` が生成）。状態名チップ（``state``）の下・数値の上
    に出して「何を直せば測定値が出るか」を映像上で示す。空文字なら `:empty` が畳む。
    測定値の整形（``symmetry``/``score``/``parts``/``chart``）へは混ざらない。
    """
    visible = position != HUD_HIDDEN and position in HUD_PLACEMENTS
    demo = ""
    if show_demographics and estimate is not None:
        age = t(_HUD_AGE_TEMPLATE).format(age=estimate.estimated_age_years)
        # `DemographicEstimate` は `gender` が `gender_probabilities` の argmax
        # であることを検証済みなので、ここで導出し直さずラベルへ変換する。
        gender = GENDER_LABELS.get(estimate.gender, "—")
        demo = t(_HUD_DEMOGRAPHIC_TEMPLATE).format(age=age, gender=gender)
    elif notice:
        demo = notice
    part_readings = {name: format_symmetry(parts.get(name)) for name in PART_NAMES}
    chart: dict[str, float | None] = {"overall": symmetry_percent(overall)}
    for name in PART_NAMES:
        chart[name] = symmetry_percent(parts.get(name))
    return {
        "visible": visible,
        "symmetry": format_symmetry(overall),
        "label": t("対称度（実験値）"),
        # §8.6.7: 「全体のスコア」だけは点の目盛りである。`format_symmetry` を
        # 使うと HUD だけが 98% 前後の別の目盛りを出し、見出しと食い違う。
        "score": format_score(equal_weight),
        "parts": part_readings,
        "demo": demo,
        "gate": gate,
        "state": state,
        "smoothing": smoothing,
        "hold": hold,
        "guidance": guidance,
        "gen": generation,
        "chart": chart,
        "baseline": build_baseline_payload(
            baseline=baseline,
            overall=overall,
            equal_weight=equal_weight,
            parts=parts,
        ),
    }


def hud_html(
    *,
    overall: float | None,
    equal_weight: float | None,
    parts: Mapping[str, float | None],
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
    position: str,
    generation: str,
    notice: str = "",
    baseline: BaselineSummary | None = None,
    state: str = "",
    smoothing: str = "",
    hold: str = "",
    guidance: str = "",
) -> str:
    """payload を bridge が読む隠し要素へ直列化する。

    意図的に小さく、装飾も持たせない。5 Hz で通信を越えるのはこれだけであり、
    以前の版——装飾済みのブロックと `<style>` タグ——が映像のカクつきの原因
    だった。再描画のたびにブラウザが全文書のスタイル再計算を強いられていた。
    折れ線グラフを追加した H1 でも、運ぶのは現在値と `chart` の数値だけで、
    系列の履歴やマークアップは運ばない。
    """
    payload = build_hud_payload(
        overall=overall,
        equal_weight=equal_weight,
        parts=parts,
        estimate=estimate,
        show_demographics=show_demographics,
        gate=gate,
        position=position,
        generation=generation,
        notice=notice,
        baseline=baseline,
        state=state,
        smoothing=smoothing,
        hold=hold,
        guidance=guidance,
    )
    encoded = json.dumps(payload, ensure_ascii=False)
    return f'<div id="{HUD_DATA_ELEMENT_ID}" style="display:none">{encoded}</div>'


def render_hud_payload(
    snapshot: PipelineSnapshot,
    *,
    show_demographics: bool,
    smoothing_tau_seconds: float | None = None,
    display_hold_seconds: float | None = None,
) -> None:
    """1 つの snapshot 分の HUD 値を publish する（5 Hz の見出し fragment から呼ぶ）。

    隠し要素であり id で参照されるため、DOM 上のどこへ出ても構わない。これが
    3 つ目の fragment を新設せずに既存の fragment へ相乗りできる理由である。

    §8.6.3/§8.7: 全体値・全体のスコア・部位別はいずれも主指標 `symmetry-3d/1.0.0`
    から作り（`ui.readings.resolve_overall_distance`/`resolve_equal_weight_distance`/
    `resolve_part_distance_3d`）、パネルの見出しと同じ解決を通す。映像上の HUD と
    パネルが同じ測定値に対して二通りの表示規則へ分岐しないようにするためである。

    H1: 世代識別子には `PipelineSnapshot.segment_id`（`SessionRecorder.segment_id`
    の読み戻し）をそのまま流用する。セッションの Reset・入力ソースの切替・
    指標に影響する設定変更のいずれでも更新されるため、折れ線グラフのリング
    バッファをこれらを跨いで正しく空にできる（§3.4・§8.8）。

    FR-038: `snapshot.age_gate_outcome` が参考年齢の表示を許さない場合
    （`may_show_reference_age` が偽）は、数値の代わりに `ui.age_gate_notice` の
    固定文言を出す。ゲートを未接続の pipeline（`age_gate_outcome` が `None`）は
    従来どおり `show_demographics` だけで決める。

    FR-039: 参考年齢・参考性別を実際に表示しているときに限り、同じ参考年齢・参考性別の
    参照平均（§9.5）を解決して併記する。参照データが読めない・件数がしきい値未満・
    年齢が 18〜75 の外のいずれでも `None` が返るだけで、左右差の表示は変わらない
    （不変条件 12、AC-BASELINE-03）。

    FR-040/§5.4: `smoothing_tau_seconds`（pipeline が報告する現在の時定数）と
    品質ゲート状態を、枠色とは別のテキストとして併せて publish する。**snapshot
    ではなく引数で受けるのは、設定を変えた瞬間に表示へ反映させるためである。**
    `PipelineSnapshot` は解析フレームごとにしか作り直されないので、そこへ載せる
    と選び直してから次の解析まで（最大 `analysis_interval_ms`）古い値が残る。
    """
    analysis = snapshot.analysis
    overall = resolve_overall_distance(analysis) if analysis is not None else None
    equal_weight = resolve_equal_weight_distance(analysis) if analysis is not None else None
    parts: dict[str, float | None] = (
        {name: resolve_part_distance_3d(analysis, name) for name in PART_NAMES}
        if analysis is not None
        else {}
    )

    # §5.1（2026-08-26）: 表示可否は `ui.age_gate_notice` の 1 つの解決に任せる。
    # HUD・専用タブ・吹き出し・レポートが同じ関数を通ることが、「ある区画だけが
    # ゲートの判断を漏らす」壊れ方を構造的に無くす手段である（§10.12-3）。
    view = resolve_reference_age_view(
        estimate=snapshot.demographic_estimate,
        gate_outcome=snapshot.age_gate_outcome,
        show_demographics=show_demographics,
    )

    # FR-039/§10.12-3: 参考年齢・参考性別を実際に出しているときだけ母集団を解決する。
    # `view.estimate` は既にゲートを通した後なので、ここで `snapshot` を読み直さない
    # ——読み直すと、年齢ゲートが隠した判断が「20 歳・男性の平均」という母集団の
    # 表示から復元できてしまう。
    baseline = resolve_baseline(view.estimate)

    st.markdown(
        hud_html(
            overall=overall,
            equal_weight=equal_weight,
            parts=parts,
            estimate=view.estimate,
            show_demographics=view.is_shown,
            notice=view.notice,
            baseline=baseline,
            gate=gate_color(analysis.state if analysis is not None else None),
            state=gate_label(analysis.state if analysis is not None else None),
            smoothing=format_smoothing_window(smoothing_tau_seconds),
            hold=format_display_hold(display_hold_seconds),
            guidance=gate_guidance(analysis.state if analysis is not None else None),
            position=st.session_state.get(HUD_POSITION_STATE_KEY, HUD_POSITION_OPTIONS[0]),
            generation=snapshot.segment_id or "",
        ),
        unsafe_allow_html=True,
    )


def hud_bridge_script(*, fullscreen: bool, position: str) -> str:
    """`HUD_RUNTIME_JS` をページへ設置するローダー。

    `components.v1.html` の iframe 内で動く。この iframe はページと same-origin
    なので `window.parent` へ到達できる。そこで行う仕事は最小限に留める。設定を
    publish し、実行時コード本体は `<script>` として親の realm へ渡すだけである。
    iframe 内で定義したものは、次の再実行で Streamlit がそれを差し替えた瞬間に
    動かなくなるためである。

    HUD をページの `<body>` へ付けることが、全画面表示で生き残る条件でもある。
    `<body>` 直下なら Streamlit のどのコンテナにもクリップされず、包含ブロック
    にもされないため、`position: fixed` が期待どおりに振る舞う。

    FR-043: `ui.hud_assets.HUD_RUNTIME_JS` は骨格を 1 度だけ作る静的な文字列
    なので、画面に出る日本語（部位ラベル・凡例・「全体のスコア」・grip の文言）
    を直接焼き込むと言語切替が効かない。ここで `t()` を通した文字列を CFG へ
    載せ、runtime 側は骨格を作り直さず `textContent` を CFG の値で書き換える
    （bridge が再実行されるたびに、そのときの表示言語を反映する）。
    """
    config = json.dumps(
        {
            "fullscreen": fullscreen,
            "position": position,
            "anchor": HUD_PLACEMENTS.get(position, ""),
            "dataId": HUD_DATA_ELEMENT_ID,
            "css": HUD_STYLESHEET,
            "videoCss": VIDEO_CHROME_CSS,
            "intervalMs": HUD_REFRESH_MS,
            "badgeText": t(FULLSCREEN_BADGE),
            "hintText": t(FULLSCREEN_HINT),
            "scoreLabel": t("全体のスコア"),
            "partLabels": {
                "contour": t("輪郭"),
                "eyes": t("目"),
                "nose": t("鼻"),
                "mouth": t("口"),
            },
            "legendOverallLabel": t("全体"),
            "gripText": t("ドラッグで移動"),
            "chartWidth": HUD_CHART_WIDTH_PX,
            "chartHeight": HUD_CHART_HEIGHT_PX,
            "chartCapacity": HUD_CHART_CAPACITY,
            "chartYMin": HUD_CHART_Y_MIN,
            "chartYMax": HUD_CHART_Y_MAX,
        },
        ensure_ascii=False,
    )
    runtime = json.dumps(HUD_RUNTIME_JS)
    return f"""
<script>
(function () {{
  var win = window.parent;
  var doc = win && win.document;
  if (!doc || !doc.body) return;
  win.__fslHudConfig = {config};
  var previous = doc.getElementById('fsl-hud-runtime');
  if (previous) previous.remove();
  var runtime = doc.createElement('script');
  runtime.id = 'fsl-hud-runtime';
  runtime.textContent = {runtime};
  doc.head.appendChild(runtime);
}})();
</script>
"""


def render_hud_bridge(*, fullscreen: bool, position: str) -> None:
    """HUD bridge を 1 回の実行につき 1 度だけ設置する（5 Hz の fragment 内では呼ばない）。"""
    components.html(hud_bridge_script(fullscreen=fullscreen, position=position), height=0)
