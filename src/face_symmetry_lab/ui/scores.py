"""Camera と静止画の両セクションが共有する詳細結果の描画。

文言は §5.3 に従う。指標は実験的な左右差 *距離* として導入し、`HOLD` は次に
取るべき操作へ言い換え、値の大小に価値判断を結び付けない。§8.6 で 0〜100 の
変換を廃止したため、ここに出る対称度はすべて両眼間隔に対する正規化距離を
百分率で示したものである。小さいという事実は対応点どうしが近いことだけを
意味し、決して「良い」を意味しない。

**この module は「描画」と「文言」だけを持つ。** 表示値の解決と整形は
`ui.readings`、対称化パネルは `ui.symmetrization_panel` にある。§8.6.3（H4）で
この 2 つを外へ出したのは、以前ここが「文言・整形・値の解決・描画」を 1 つの
module に抱えており、HUD や吹き出しが数値を整形するためだけにこの描画
module へ依存していたためである。

**「全体のスコア」の計算過程開示は撤回した**（2026-09-03、§0 項目 34）。§8.6.7
の表示スケールと、それを 3 ステップで開示していた H4 の機能一式（旧
`domain.score_explanation`）を機能ごと削除した。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import streamlit as st

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate, Status
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.i18n import t
from face_symmetry_lab.ui.readings import (
    format_distance,
    format_symmetry,
    gender_label,
    part_label,
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


#: §5.3 が義務付ける「この数値をどう読むか」の留保。**畳まない。**
#: 開かないと見えない場所へ置くと数字だけが独り歩きする。以前はこれらが
#: 詳細結果の末尾に無名の `st.caption` として 3 本
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
    阻害理由（`reason_codes`）は `st.markdown` で箇条書き、非ブロッキック警告
    （`warning_codes`）は `st.caption` で視覚的に区別する（§6.2/§8.2）。
    """
    st.write(t(_STATUS_HEADLINE[outcome.state.status]))
    for code in outcome.state.reason_codes:
        if code in REASON_GUIDANCE:
            st.markdown(f"- {t(REASON_GUIDANCE[code])}")
    for code in outcome.state.warning_codes:
        if code in REASON_GUIDANCE:
            st.caption(f"⚠ {t(REASON_GUIDANCE[code])}")


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

    全体の対称度の数値は、映像上の HUD（Camera）と画像上の吹き出し（Image）が
    出す。**見出し結果の区画はどちらの入力にも無い**（§5.1、2026-08-12）。
    こちらは吹き出しが持てないもの——ずれの併記・部位別の内訳・§5.3 の注記・
    姿勢——を、その下の画面幅いっぱいの詳細セクションへ描画する（§5.1, FR-013,
    AC-LAYOUT-01）。**総合値そのものをメトリクスとして重複表示しない。**

    ``show_parts`` は表示専用のパラメータで、部位別 4 つのメトリクスと生値の
    併記を描くかどうかを切り替える（項目26、H3）。**Camera** は映像上の HUD が
    既に部位別 4 つを表示するため ``False`` を渡し、詳細タブでの重複を避ける。
    **Image** は HUD を持たないため既定の ``True`` のまま描画する。``False``
    でも `PART_LEVEL_VARIATION_CAPTION`（§5.3 が義務付ける部位別の揺れ幅の
    注記）は描き続ける。HUD には注記を置く場所が無く、この注記の唯一の受け皿が
    詳細タブになるためである。

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

    render_pose(outcome)

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
    About が持つ。
    """
    st.caption(t(READING_LIMITS_NOTE))


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
