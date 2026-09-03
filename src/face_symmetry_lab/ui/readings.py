"""画面に出る 1 つの数値を「どこから読み、どう整形するか」だけを決める層。

**この module は何も描画しない。** `AnalysisOutcome` から表示値を取り出す関数と、
値を文字列にする関数、そして状態→色・部位→ラベルの対応表だけを持つ。表示言語
（FR-043）に依存する語（部位名・性別・状態名など）だけは
`face_symmetry_lab.ui.i18n.t` を通す——`st.write` 等の描画 API は一切呼ばない。

**分離した理由。** これを読む側は 4 つある——詳細パネル（`ui.scores`）、映像上の
HUD（`ui.hud`）、静止画の吹き出し（`ui.still_overlay`）、Image の組み立て
（`ui.image_section`）。同じ測定値に対して 2 つ目の表示規則が生まれると、画面上
の別々の場所が別々の数字を出す。以前これらが `ui.scores`（描画 module）に同居
していたため、HUD が値を整形するためだけに描画 module へ依存していた。**訳を
ここへ集約する理由も同じ**——`t()` をこの module の関数の中で 1 度だけ通せば、
HUD・吹き出し・詳細パネルのすべてが同じ訳を受け取る（FR-043、AC-I18N-01）。

規則は 1 つだけである。

* **欠測は `—`。** 0 でも前回値でもない（不変条件 11、NFR-013）。
"""

from __future__ import annotations

from typing import Final

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import AnalysisState, GenderLabel, Status
from face_symmetry_lab.domain.symmetry_display import symmetry_percent as _symmetry_percent
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.i18n import t

#: §8.6.3 の部位ラベル。値は文字列リテラルのまま持つ——`t(PART_LABELS[name])`
#: という形で、この module 自身だけでなく他の module（`ui.still_overlay` /
#: `ui.report` 等）からも辞書の添字を経由して訳せるようにするため
#: （`tests/ui/test_i18n.py` の辞書添字解決）。
PART_LABELS: Final[dict[str, str]] = {
    "contour": "輪郭",
    "eyes": "目",
    "nose": "鼻",
    "mouth": "口",
}

#: `PART_LABELS` と同じ理由で、値は文字列リテラルのまま持つ。
GENDER_LABELS: Final[dict[GenderLabel, str]] = {
    GenderLabel.FEMALE: "女性",
    GenderLabel.MALE: "男性",
}


def part_label(name: str) -> str:
    """`PART_LABELS[name]` を表示言語（FR-043）に合わせて訳す。"""
    if name not in PART_LABELS:
        return name
    return t(PART_LABELS[name])


def gender_label(gender: GenderLabel) -> str:
    """`GENDER_LABELS[gender]` を表示言語（FR-043）に合わせて訳す。"""
    if gender not in GENDER_LABELS:
        return "—"
    return t(GENDER_LABELS[gender])


#: §9.6.1（2026-08-31）: 観測された性別確率を「女性 X% ・ 男性 Y%」形式で整形する。
GENDER_PROBABILITY_TEMPLATE: Final[str] = "{female} {female_pct}% ・ {male} {male_pct}%"


def format_gender_probabilities(probabilities: tuple[float, float]) -> str:
    """観測された性別確率を『女性 X% ・ 男性 Y%』へ整形する（和は 100% に丸める）。"""
    female_pct = round(probabilities[0] * 100)
    male_pct = 100 - female_pct
    return t(GENDER_PROBABILITY_TEMPLATE).format(
        female=gender_label(GenderLabel.FEMALE),
        female_pct=female_pct,
        male=gender_label(GenderLabel.MALE),
        male_pct=male_pct,
    )


# --- 整形 --------------------------------------------------------------------


def format_distance(value: float | None) -> str:
    """正規化距離を百分率で表示する。欠測時は `—` を返す。"""
    return "—" if value is None else f"{value * 100.0:.2f}%"


def symmetry_percent(value: float | None) -> float | None:
    """§8.6.3 の対称度変換を、文字列ではなく数値のまま返す。

    式そのものは `domain.symmetry_display.symmetry_percent` が持ち、ここは
    委譲するだけである。domain へ置いてあるのは、FR-021 の要約
    （`domain.session_summary`）も同じ変換を必要とするためで、ui へ置いた
    ままだと domain が ui へ依存することになる。

    `format_symmetry`・`ui.hud` の折れ線グラフ・要約・FR-042 のレポートが
    すべてこの 1 つの式を共有する。2 箇所に書くと、画面上の別々の場所が同じ
    測定値から別の数を出す。
    """
    return _symmetry_percent(value)


def format_symmetry(value: float | None) -> str:
    """正規化距離を §8.6.3 の対称度パーセントとして表示する。

    `100% - ずれ%` は線形かつ可逆で、係数を持たない。したがって同じ測定値の
    読む向きを変えるだけであり、採点ではない。指標そのもの、エクスポートの
    列、`metric_version` はいずれもこれによって変わらない。
    """
    percent = symmetry_percent(value)
    return "—" if percent is None else f"{percent:.2f}%"


# --- 値の解決 ----------------------------------------------------------------


def resolve_overall_distance(outcome: AnalysisOutcome) -> float | None:
    """§8.6.3/§8.7: 「全体の対称度」として表示する 3D の総合距離を 1 か所で決める。

    平滑値（`overall_distance_3d_smooth`）を優先し、無ければ生値
    （`symmetry_3d.overall_distance`）へ落ちる。静止画は §8.7 のとおり平滑値を
    持たない（`application.still_image.analyze_still_image` が明示的に欠測へ
    落とす）ため、この関数は自然に生値へ落ちる——呼び出し側が入力モードを
    分岐する必要はない。3D 一式が丸ごと欠測なら `None`（`—`）を返し、`symmetry`
    （2D）へは決してフォールバックしない。
    """
    if outcome.overall_distance_3d_smooth is not None:
        return outcome.overall_distance_3d_smooth
    symmetry_3d = outcome.symmetry_3d
    return None if symmetry_3d is None else symmetry_3d.overall_distance


def resolve_part_distance_3d(outcome: AnalysisOutcome, name: str) -> float | None:
    """§8.6.3/§8.7 改訂(H1): 部位別 `d_3d,p` の平滑値を 1 か所で決める。

    `resolve_overall_distance` と同じ「平滑値を優先し、無ければ生値」という
    解決を部位別にも適用する。`outcome.part_distances_3d_smooth` がコンテナ
    ごと欠測(静止画・3D 一式欠測)、またはその部位のキーだけ欠測(不変条件 11、
    部位が個別に欠測したフレーム)のいずれでも自然に生値へ落ちる。詳細タブと
    HUD のどちらの呼び出し側もこの解決を個別に書かないよう、ここへ集約する
    (表示の一貫性: 片方が生値・片方が平滑値という 2 つの表示規則を作らない)。
    """
    smoothed = outcome.part_distances_3d_smooth
    if smoothed is not None:
        value = smoothed.get(name)
        if value is not None:
            return value
    symmetry_3d = outcome.symmetry_3d
    if symmetry_3d is None:
        return None
    return symmetry_3d.distance(name)


# --- 品質ゲートの色 -----------------------------------------------------------

#: §14.3e: 品質ゲートの状態ごとの色。映像の枠線と HUD の輪郭に使う。これらは
#: **状態の識別子であって評価ではない**。緑は「計測条件を満たしている」という
#: 意味であって、「良い顔である」ではない。§5.4 は色だけで意味を伝えることを
#: 禁じているため、ゲートが閉じている間は詳細結果タブの `render_guidance` が
#: 状態文言と理由を示し、ゲートが閉じると HUD・詳細結果の数値が `—` へ変わる
#: こと（値の有無）と合わせて色以外の手掛かりを保つ（項目26）。
GATE_COLORS: Final[dict[Status, str]] = {
    Status.OK: "#2E9E5B",
    Status.HOLD: "#D9A21B",
    Status.NO_FACE: "#B4433B",
    Status.MULTI_FACE: "#B4433B",
    Status.ERROR: "#6B6B6B",
}

#: 解析がまだ 1 つも無いとき、および表が取りこぼした状態のときに使う。判定を
#: 匂わせる色ではなく、`Status.ERROR` の中立的な灰色を選ぶ。
_UNKNOWN_GATE_COLOR: Final[str] = GATE_COLORS[Status.ERROR]


def gate_color(state: AnalysisState | None) -> str:
    """ゲート状態 1 つに対する §14.3e の表示色。

    ``None``（このセッションでまだ解析が無い）は中立的な灰色として読む。
    待機中の表示が「条件を満たしている」色を借りることがないようにする。
    """
    if state is None:
        return _UNKNOWN_GATE_COLOR
    return GATE_COLORS.get(state.status, _UNKNOWN_GATE_COLOR)


#: §5.2/§5.4: 枠色と同じ情報をテキストでも持つ。**述べるのは「何が起きているか」
#: だけで、良し悪しではない**（「良好」「不良」を使わない）。詳細結果タブの
#: `render_guidance` が出す行動可能な指示文（§6.3）とは役割が違う——こちらは
#: HUD の 1 行に収まる状態名であり、指示文の置き換えではない。値は文字列
#: リテラルのまま持つ（`t(GATE_LABELS[status])` という辞書添字解決のため）。
GATE_LABELS: Final[dict[Status, str]] = {
    Status.OK: "計測中",
    Status.HOLD: "調整中",
    Status.NO_FACE: "顔が見つかりません",
    Status.MULTI_FACE: "顔が複数あります",
    Status.ERROR: "処理に失敗しました",
}

#: 解析がまだ 1 つも無いとき。`gate_color` の中立色と対になる文言で、`Status` の
#: どの値とも別物である（「まだ何も測っていない」は失敗ではない）。
_UNKNOWN_GATE_LABEL: Final[str] = "待機中"


def gate_label(state: AnalysisState | None) -> str:
    """ゲート状態 1 つに対する §5.4 のテキスト表示（表示言語に従って訳す）。"""
    if state is None or state.status not in GATE_LABELS:
        return t(_UNKNOWN_GATE_LABEL)
    return t(GATE_LABELS[state.status])


def gate_guidance(state: AnalysisState | None) -> str:
    """FR-031: ライブ映像上 HUD へ併記する、主要因 1 件の行動可能な指示文。

    `gate_label` が状態名(「調整中」等)を返すのに対し、こちらは「何を直せば
    よいか」を返す。詳細結果タブ(`ui.scores.render_guidance`)が全阻害理由を
    一覧で示す正本で、ここはその主要因 1 件だけを HUD の 1 行へ出す。判断は
    一切足さず、品質ゲートが算出済みの `reason`(無ければ `reason_codes[0]`)を
    §6.3 の同じ表(`ui.guidance.REASON_GUIDANCE`)へ通すだけである(FR-031)。

    ゲートを満たしている間(`Status.OK`)と解析がまだ無い間(`None`)は空文字を
    返す——CSS の `:empty` が行ごと畳み、代わりに測定値が出る。空文字であって
    `—` ではない(状態の不在は測定の欠測ではない。`gate_label`/`format_*` と
    同じ扱い)。
    """
    if state is None or state.status is Status.OK:
        return ""
    code = state.reason or (state.reason_codes[0] if state.reason_codes else None)
    if code is None or code not in REASON_GUIDANCE:
        return ""
    return t(REASON_GUIDANCE[code])


def gate_warning(state: AnalysisState | None) -> str:
    """FR-031/§6.2: ライブ映像上 HUD へ併記する、非ブロッキングの助言（照度など）。

    `gate_guidance` が「測定値を出すために直す阻害理由」を返すのに対し、こちらは
    分析を止めない `warning_codes`（§6.2、例: `LOW_LIGHT`/`OVEREXPOSED`）を、同じ
    §6.3 の表（`ui.guidance.REASON_GUIDANCE`）へ通して助言として返す。警告が無ければ
    空文字（`—` ではない。CSS の `:empty` が畳む）。**品質ゲートの阻害理由とは別物**で、
    警告があっても測定値の表示は止まらない。複数あるときは先頭 1 件だけを出す。
    """
    if state is None or not state.warning_codes:
        return ""
    code = state.warning_codes[0]
    if code not in REASON_GUIDANCE:
        return ""
    return t(REASON_GUIDANCE[code])


def display_signature(outcome: AnalysisOutcome | None) -> tuple[object, ...]:
    """FR-041: 表示ラッチ（`domain.display_hold`）へ渡す「値が届いているかの型」。

    **値そのものは含めない。** 含めると毎フレーム鍵が変わり、保持がまったく効か
    なくなる（保持したいのは、まさにその値の細かい揺れである）。含めるのは
    「品質ゲートの状態」と「画面に出る 5 つの数値が欠測かどうか」だけで、これが
    変わった瞬間にラッチは保持を打ち切る——顔を見失った後に古い数値が残らない
    のはこのためである（不変条件 11、AC-HOLD-02）。

    有無の判定には、実際に画面へ出すのと**同じ解決関数**を使う。別の条件で判定
    すると、「表示は `—` なのにラッチは値があると思っている」状態が生まれる。
    """
    if outcome is None:
        return (None,)
    return (
        outcome.state.status,
        resolve_overall_distance(outcome) is not None,
        tuple(resolve_part_distance_3d(outcome, name) is not None for name in PART_NAMES),
    )


DISPLAY_HOLD_TEMPLATE: Final[str] = "更新 {seconds:.1f} 秒ごと"


def format_display_hold(hold_seconds: float | None) -> str:
    """FR-041: いま有効な表示の更新間隔を、HUD と設定 UI が共有する形で表示する。

    **0 秒（毎回更新）は空文字にする。** 既定の挙動をわざわざ画面へ書くと、
    HUD に常時 1 行増えるだけで何も伝わらない。`format_smoothing_window` と同じ
    く、`—` は使わない（設定の不在は測定の欠測ではない）。
    """
    if hold_seconds is None or hold_seconds <= 0.0:
        return ""
    return t(DISPLAY_HOLD_TEMPLATE).format(seconds=hold_seconds)


SMOOTHING_WINDOW_TEMPLATE: Final[str] = "平滑化 {seconds:.1f} 秒"


def format_smoothing_window(tau_seconds: float | None) -> str:
    """FR-040: いま有効な平滑化の時定数を、HUD と設定 UI が共有する形で表示する。

    **欠測（`None`）は空文字であり `—` ではない。** `—` は「あるはずの測定値が
    欠測している」の記号であり（不変条件 11）、設定が読めないこと（analyzer 未
    接続）は測定の欠測ではない。空文字は CSS の `:empty` で行ごと畳まれる。
    """
    if tau_seconds is None:
        return ""
    return t(SMOOTHING_WINDOW_TEMPLATE).format(seconds=tau_seconds)
