"""Camera と静止画の両セクションが共有する結果表示。

文言は §5.3 に従う。指標は実験的な左右差 *距離* として導入し、`HOLD` は次に
取るべき操作へ言い換え、値の大小に価値判断を結び付けない。§8.6 で 0〜100 の
変換を廃止したため、ここに出る値はすべて両眼間隔に対する正規化距離を百分率で
示したものである。小さいという事実は対応点どうしが近いことだけを意味し、
決して「良い」を意味しない。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import streamlit as st

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.symmetrization import SymmetrizationView
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import (
    AnalysisState,
    DemographicEstimate,
    GenderLabel,
    Status,
)
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE

SCORE_HEADING = "対称度（実験値・両眼距離比）"

#: §5.2: 距離は両眼間隔に対する比なので、小数 2 桁の百分率が、精度を偽らずに
#: 読める形になる。§5.3/§12.1（`docs/symmetry-metric-measurements.md`）: 実写の
#: 表示値はテイク内でも数パーセントポイントの狭い範囲に集まるため、「100% は
#: 目標値ではない」という既存の注意の延長として、その狭い範囲の中の違いを
#: 対称・非対称の判定として読ませない注意文を添える。
DISTANCE_UNIT_CAPTION = (
    "対称度は、両眼の間隔を 100% としたときの左右対応点のずれを 100% から引いた値です。"
    "100% は対応点距離がすべて 0 になる数学上の上限で、目標値ではありません。"
    "高い・低いは良し悪しを表しません。"
    "実写では表示値が数パーセントポイントの狭い範囲に集まりやすく、"
    "その中の違いを対称・非対称の判定として読まないでください。"
)


#: §8.6.3/§8.6.5: 見出しに出す `s_3d` のラベル。**ユーザー指定により
#: 「部位平均の対称度」ではなく「全体のスコア」とする**（2026-08-11、
#: W5-fix）。実体は §8.6.3 と同じ線形・無パラメータの変換で読んだ対称度であり、
#: `s_3d` の重みは全部位について 1 に固定され調整の余地を持たないため、§8.6 が
#: 禁じる較正されていない自由パラメータには当たらない（§8.6.3「表示語として
#: のスコア」・§8.6.5）。採点・評価ではないことは help（`PART_EQUAL_WEIGHT_HELP`）
#: で明示する。
PART_EQUAL_WEIGHT_LABEL: Final[str] = "全体のスコア"

#: §8.6.3/§8.6.5: 「全体のスコア」の help。全体の対称度（対応点を 1 票ずつ数える
#: `d_3d`）とは重みだけが異なる並列の総合量であることと、採点ではないことの
#: 両方を明示する。
PART_EQUAL_WEIGHT_HELP: Final[str] = (
    "部位を 1 票ずつ数えた対称度です。全体の対称度（対応点を 1 票ずつ）とは重みだけが"
    "違い、鼻のようにペア数の少ない部位も、目のように多い部位と同じ重みで入ります。"
    "較正された感度係数や部位の重み付けを持たない値であり、採点ではありません。"
    "高い・低いは良し悪しを表しません。"
)

#: §5.3・§14.7（同 docs）: 部位別の値は同一条件で撮り直すだけでも動く幅がある。
#: この注記は 4 部位すべてに等しく掛かる形にし、特定の部位（鼻など）だけを
#: 名指しで「使えない」と述べない。測定は被写体 1 名によるため、部位ごとの
#: 個人差は未検証である（OD-09）。
PART_LEVEL_VARIATION_CAPTION: Final[str] = (
    "部位別の値は、同じ条件で撮り直すだけでも動きます。"
    "撮影のたびに動く幅より小さい変化は、部位別の値だけでは左右差の変化として読み取れません。"
    "この幅は部位ごとに異なり、個人差は未検証です。"
)


#: §8.6.2 の点ごとのずれは §8.5 のペア距離の半分であり、§8.6.3 が向きを変える
#: 量とは別物である。そのため対称度ではなく「ズレ」として述べ、専用の注記を
#: 添える。
DEVIATION_UNIT_CAPTION = (
    "ズレは、両眼の間隔を 100% としたときの、実測点と左右対称にそろえた位置との距離です。"
    "小さいほど両者が近いことだけを意味し、良し悪しを表しません。"
)


#: §5.3/§10.7: FR-035 の変形画像を利用者へ提示するときの枠付け。文言を直接
#: 検証できるよう定数に切り出している。これは「補正済みの写真」と最も
#: 取り違えられやすい出力である。
SYMMETRIZED_IMAGE_CAPTION = "左右対称にそろえた画像（ランドマークを基準に画素を変形した加工画像）"
SYMMETRIZED_IMAGE_NOTICE = (
    "対応点を左右対称の位置へ移動させ、その間の画素を三角形ごとに変形した合成画像です。"
    "実際の顔写真ではなく、より正しい・望ましい顔でもありません。"
    "加工画像であることを明示せずに提示・共有しないでください。"
)
SYMMETRIZED_IMAGE_LIMITS = (
    "ランドマークが覆わない髪・耳・首・背景は、画像の外周を固定したまま"
    "顔の変形に緩やかに追従するため、境界付近に不自然さが残ることがあります。"
)


def format_distance(value: float | None) -> str:
    """正規化距離を百分率で表示する。欠測時は `—` を返す。"""
    return "—" if value is None else f"{value * 100.0:.2f}%"


def format_symmetry(value: float | None) -> str:
    """正規化距離を §8.6.3 の対称度パーセントとして表示する。

    `100% - ずれ%` は線形かつ可逆で、係数を持たない。したがって同じ測定値の
    読む向きを変えるだけであり、採点ではない。指標そのもの、エクスポートの
    列、`metric_version` はいずれもこれによって変わらない。

    ずれが両眼間隔そのものを超える退化ケースのために 0% で下限を設ける。
    それ以外の場所では `d` は上限のない値のままである（§8.6.3）。
    """
    if value is None:
        return "—"
    return f"{min(100.0, max(0.0, (1.0 - value) * 100.0)):.2f}%"


def resolve_overall_distance(outcome: AnalysisOutcome) -> float | None:
    """§8.6.3/§8.7: 「全体の対称度」として表示する 3D の総合距離を 1 か所で決める。

    平滑値（`overall_distance_3d_smooth`）を優先し、無ければ生値
    （`symmetry_3d.overall.distance`）へ落ちる。静止画は §8.7 のとおり平滑値を
    持たない（`application.still_image.analyze_still_image` が明示的に欠測へ
    落とす）ため、この関数は自然に生値へ落ちる——呼び出し側が入力モードを
    分岐する必要はない。3D 一式が丸ごと欠測なら `None`（`—`）を返し、`symmetry`
    （2D）へは決してフォールバックしない。

    Camera（見出し・HUD）と Image のどの呼び出し側もこの解決を個別に書かない
    よう、ここへ集約している。
    """
    if outcome.overall_distance_3d_smooth is not None:
        return outcome.overall_distance_3d_smooth
    symmetry_3d = outcome.symmetry_3d
    if symmetry_3d is not None and symmetry_3d.overall is not None:
        return symmetry_3d.overall.distance
    return None


def resolve_equal_weight_distance(outcome: AnalysisOutcome) -> float | None:
    """§8.6.3/§8.6.5: 見出しの「全体のスコア」として表示する `s_3d` を 1 か所で決める。

    `s_3d` は §8.6.5・§8.7 のとおり EMA の対象に含めないため、常に生値である
    （`resolve_overall_distance` と異なり、平滑値へのフォールバックを持たない
    ——そもそも平滑値が存在しない）。3D 一式が欠測、または部位が 1 つでも
    欠測していれば `None`（`—`）を返し、0 や前回値へ置換しない（不変条件 11）。
    `Symmetry3DResult.equal_weight_distance` が「部位が 1 つでも欠測なら欠測に
    する。使える部位だけで平均しない」を実装しているので、ここではそれを
    そのまま読むだけでよい。

    Camera（見出し）と Image のどちらの呼び出し側もこの解決を個別に書かないよう、
    `resolve_overall_distance` と同じくここへ集約している。
    """
    symmetry_3d = outcome.symmetry_3d
    if symmetry_3d is None:
        return None
    return symmetry_3d.equal_weight_distance


_PART_LABELS: Final[dict[str, str]] = {
    "contour": "輪郭",
    "eyes": "目",
    "nose": "鼻",
    "mouth": "口",
}

GENDER_LABELS: Final[dict[GenderLabel, str]] = {
    GenderLabel.FEMALE: "女性",
    GenderLabel.MALE: "男性",
}

#: §14.3e: 品質ゲートの状態ごとの色。映像の枠線と HUD の輪郭に使う。これらは
#: **状態の識別子であって評価ではない**。緑は「計測条件を満たしている」という
#: 意味であって、「良い顔である」ではない。§5.4 は色だけで意味を伝えることを
#: 禁じているため、どの呼び出し側も `render_tracking_status` /
#: `guidance_message` のテキストを併記し続ける。
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
    st.write(_STATUS_HEADLINE.get(outcome.state.status, ""))
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


def render_pose(outcome: AnalysisOutcome) -> None:
    pose = outcome.pose
    if pose is None:
        st.caption("姿勢: 取得できません")
        return
    st.caption(
        f"姿勢 yaw {pose.yaw_deg:.1f}° / pitch {pose.pitch_deg:.1f}° / roll {pose.roll_deg:.1f}°"
        + (
            f" · 顔面積比 {outcome.face_area_ratio:.2f}"
            if outcome.face_area_ratio is not None
            else ""
        )
    )


def render_headline_metrics(
    *,
    overall: float | None,
    equal_weight: float | None,
    estimate: DemographicEstimate | None,
) -> None:
    """FR-013/FR-014/FR-032: 見出しを 2 行に分けて映像や写真のすぐ隣に出す。

    Camera と Image が共有し、両者が同じ見出し配置を使う（§5.1）。**見出しは
    2 行構成である。** 1 行目は左右差の 2 値（全体の対称度・全体のスコア）、
    2 行目は参考値の 2 値（参考性別・参考年齢）とする。不変条件 5（§1.3.1）は
    参考年齢・参考性別と左右差を別々の結果として扱い、相関・因果を示唆しない
    と定めており、1 行に 3〜4 項目を並置するより行を分けるほうがこの不変条件
    に忠実である。

    ``overall``（§8.6.3 の `d_3d`）と ``equal_weight``（同 `s_3d`）は、それぞれ
    `resolve_overall_distance` / `resolve_equal_weight_distance` で解決した
    値を渡すこと。どちらも重みだけが異なる並列の総合量で、通常は一致しない
    （§8.6.5）。

    ``estimate`` は性別・年齢へ分解済みの引数ではなく、まとめて受け取る。
    §10.8 は 2 つの値を 1 つの観測として扱っており、どちらの呼び出し側も
    同じ形で展開していたためである。欠測は古い値や 0 ではなく `—` を出す
    （NFR-013）。
    """
    symmetry_column, score_column = st.columns(2)
    symmetry_column.metric(
        "全体の対称度",
        format_symmetry(overall),
        # §5.3: ずれを必ず併記する。数値をどちらの向きからも読めるようにし、
        # 「100%」が達成された目標のように受け取られないようにするためである。
        help=(
            "100% から、左右対応点のずれ（両眼間隔比）を引いた値です。"
            "100% は数学上の上限で目標値ではありません。"
        ),
    )
    score_column.metric(
        PART_EQUAL_WEIGHT_LABEL,
        format_symmetry(equal_weight),
        help=PART_EQUAL_WEIGHT_HELP,
    )
    st.caption(f"ずれ {format_distance(overall)}（対称度との和は 100%）")

    gender_column, age_column = st.columns(2)
    gender_column.metric(
        "参考性別",
        "—" if estimate is None else GENDER_LABELS[estimate.gender],
    )
    age_column.metric(
        "参考年齢",
        "—" if estimate is None else f"{estimate.estimated_age_years:.0f}歳",
    )


def render_score_details(outcome: AnalysisOutcome, *, show_smoothed: bool = True) -> None:
    """見出しの総合値を除く、左右差に関するすべてを描画する。

    見出し（`render_headline_metrics`）は全体の対称度・全体のスコアの両方を
    映像や写真のすぐ隣に出す。こちらは部位別の内訳・姿勢・肌表面参考値を
    描画する。いずれもその下の画面幅いっぱいの詳細セクションに属する
    （§5.1, FR-013, AC-LAYOUT-01）。**全体のスコア（`s_3d`）は見出しへ表示する
    ため、ここでは重複して描画しない。**

    §8.6.3 のとおり、部位別は主指標 `symmetry-3d/1.0.0`
    （`outcome.symmetry_3d`）から作る。`outcome.symmetry`（`symmetry-2d/2.0.0`）
    は §8.6.6 の構造的な欠陥を持つため画面には出さず、エクスポートの
    `distance_*_raw` としてのみ残る。
    """
    st.subheader(SCORE_HEADING)
    st.info(
        "試験中の指標です。撮影条件で値が動くため、数値の絶対値ではなく"
        "同一人物・同一環境での相対変化だけを参照してください。"
    )

    # 何か描画できるものがあるか（= 品質ゲートが開いているか）は 2D の有無で
    # 判定する。2D は品質ゲート直後に無条件で算出されるのに対し、3D は `z` の
    # 有無にも依存し独立に欠測しうる（不変条件 12 の失敗分離）。したがって
    # 2D の有無のほうが「解析そのものが成立しているか」の安定した代理になる。
    symmetry = outcome.symmetry
    if symmetry is None or symmetry.overall is None:
        render_guidance(outcome)
        render_pose(outcome)
        return

    symmetry_3d = outcome.symmetry_3d
    overall_3d_raw = (
        symmetry_3d.overall.distance
        if symmetry_3d is not None and symmetry_3d.overall is not None
        else None
    )
    if show_smoothed:
        st.caption(
            f"平滑化前の生値 {format_symmetry(overall_3d_raw)}"
            f"（ずれ {format_distance(overall_3d_raw)}）"
        )

    # §8.6.5: ずれがどこにあるかを見るための診断的な内訳。これらを足しても
    # 見出しの値にはならない。見出しは全対応ペアをまとめて 1 回だけ
    # 折りたたんで求めるためである。3D 一式が欠測なら 4 部位とも `—`。
    st.caption("部位別の対称度（全体値の内訳ではなく、ずれの分布を見るための独立した値です）")
    columns = st.columns(len(PART_NAMES))
    for column, name in zip(columns, PART_NAMES, strict=True):
        distance = symmetry_3d.distance(name) if symmetry_3d is not None else None
        column.metric(
            _PART_LABELS[name],
            format_symmetry(distance),
            help=f"ずれ {format_distance(distance)}",
        )
    st.caption(PART_LEVEL_VARIATION_CAPTION)

    render_pose(outcome)
    if outcome.skin_quality is not None:
        skin = outcome.skin_quality
        st.divider()
        st.markdown("#### 撮影補正後の肌表面参考値（実験値）")
        st.metric(
            "肌表面参考値",
            "—" if skin.score is None else f"{skin.score:.1f}",
        )
        tone_column, texture_column = st.columns(2)
        tone_column.metric(
            "色調均一性",
            "—" if skin.tone_uniformity is None else f"{skin.tone_uniformity:.1f}",
        )
        texture_column.metric(
            "テクスチャ均一性",
            "—" if skin.texture_uniformity is None else f"{skin.texture_uniformity:.1f}",
        )
        st.caption(
            "ソフトウェアでwhite balanceと明度を補正した頬patchの独立参考値です。"
            "camera、照明、化粧、髭、圧縮の影響を残し、肌質、疾患、健康、美しさを表しません。"
        )
        if skin.correction is not None:
            gains = skin.correction.gains
            st.caption(f"補正gain RGB {gains[0]:.2f} / {gains[1]:.2f} / {gains[2]:.2f}")
        st.caption(f"skin metric {skin.metric_version} · 未校正")
    st.caption(DISTANCE_UNIT_CAPTION)
    st.caption(
        "人物の状態や価値を示す尺度ではなく、実写で 100% になることも保証しません。"
        "部位別の値を平均しても全体の対称度にはなりません（全体の対称度は全対応ペアを"
        "まとめて 1 回だけ折りたたんで求めるためです）。見出しの「全体のスコア」は"
        "部位別の値から明示的に組み立てた別の値で、全体の対称度とは重みだけが異なります。"
    )
    st.caption(
        "同一人物・同一環境内の相対的な傾向を見るための値です。"
        "他人との比較、医学的評価、魅力度の判断には使用できません。"
    )
    metric_3d_label = symmetry_3d.metric_version if symmetry_3d is not None else "—"
    st.caption(
        f"metric {metric_3d_label} · 2D 診断 {outcome.metric_version} · "
        f"landmark map {outcome.landmark_map_version}"
    )


def render_symmetrized_image(view: SymmetrizationView, *, requested: bool) -> None:
    """FR-035: 変形画像を、加工であるという枠付けのまま表示する。

    Camera と Image が共有する。この文言が効く唯一の場所を、両者の間で
    食い違わせないためである。§5.3/§10.7: 対称化した写真は「補正済み」や
    「より良い」顔と最も取り違えられやすい出力なので、幾何変形であることを
    明示し、before/after としては決して見せない。

    fail-soft である。変形を要求されたが変位ガードや画像化の段階が拒否した
    場合は理由を示し、上の点オーバーレイだけで成立させる（AC-SYMM-02）。
    """
    if not requested:
        return
    if view.warped is None:
        st.info(view.warp_unavailable_reason or "対称化した画像は生成されていません。")
        return
    st.image(view.warped, caption=SYMMETRIZED_IMAGE_CAPTION, width="stretch")
    st.caption(SYMMETRIZED_IMAGE_NOTICE)
    st.caption(SYMMETRIZED_IMAGE_LIMITS)


def render_demographic_estimate(
    estimate: DemographicEstimate,
    *,
    show_distribution: bool = True,
) -> None:
    """§10.8 の年齢・性別参考パネルを描画する（Image と Camera で共有）。

    `show_distribution` は 2 クラスの確率グラフを描く。ライブのカメラパネルは
    これを切る。パネルは 5 Hz で再描画されるのに対し、この値は最大でも 1 秒に
    1 度しか変わらず、グラフの費用がここの他の要素すべての合計より大きい
    ためである。どちらの場合も同じ数値はテキストとして参照できる。
    """
    st.subheader("モデル推定の参考年齢・参考性別")
    age_column, gender_column = st.columns(2)
    age_column.metric("参考年齢", f"{estimate.estimated_age_years:.1f}歳")
    gender_column.metric("外見由来の参考性別", GENDER_LABELS[estimate.gender])
    st.caption(
        f"性別モデル確率 {estimate.gender_confidence:.1%}・対象18〜75歳。"
        "年齢や本人の自己認識・法的性別を断定せず、本人確認や人物評価には使用できません。"
    )
    if show_distribution:
        st.bar_chart(
            {
                "モデルラベル": ["female", "male"],
                "モデル出力": list(estimate.gender_probabilities),
            },
            x="モデルラベル",
            y="モデル出力",
        )
    else:
        female, male = estimate.gender_probabilities
        st.caption(f"モデル出力 female {female:.3f} / male {male:.3f}")
    if estimate.is_cached:
        st.caption("同一区間内の直近推論を再表示しています。")
    with st.expander("モデル情報"):
        st.code(
            f"version: {estimate.model_version}\n"
            f"sha256: {estimate.model_sha256}\n"
            f"observed_at: {estimate.observed_at.isoformat()}",
            language="text",
        )
