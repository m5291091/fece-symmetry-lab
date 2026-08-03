"""Result presentation shared by the camera and still-image sections.

Wording follows §5.3: the metric is introduced as an experimental left/right
difference *distance*, `HOLD` becomes an actionable instruction, and no value
judgement is attached to a large or small number. §8.6 removed the 0-100
transform, so every value here is a normalized distance shown as a percentage
of the interocular distance — smaller simply means the paired points sit
closer together, never "better".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import streamlit as st

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.symmetrization import SymmetrizationView
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel, Status
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE

SCORE_HEADING = "対称度（実験値・両眼距離比）"

#: §5.2: distances are ratios of the interocular distance, so a percentage
#: with two decimals is the readable form without inventing precision.
DISTANCE_UNIT_CAPTION = (
    "対称度は、両眼の間隔を 100% としたときの左右対応点のずれを 100% から引いた値です。"
    "100% は対応点距離がすべて 0 になる数学上の上限で、目標値ではありません。"
    "高い・低いは良し悪しを表しません。"
)


#: §8.6.2's per-point offsets are half the §8.5 pair distance, so they are not
#: the quantity §8.6.3 reorients. They stay stated as deviations, with their
#: own caption rather than the symmetry one.
DEVIATION_UNIT_CAPTION = (
    "ズレは、両眼の間隔を 100% としたときの、実測点と左右対称にそろえた位置との距離です。"
    "小さいほど両者が近いことだけを意味し、良し悪しを表しません。"
)


#: §5.3/§10.7: the user-visible framing of FR-035's warped image. Kept as
#: constants so the wording can be asserted on directly — this is the output
#: most easily mistaken for a "corrected" photograph.
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
    """Render a normalized distance as a percentage, or `—` when missing."""
    return "—" if value is None else f"{value * 100.0:.2f}%"


def format_symmetry(value: float | None) -> str:
    """Render a normalized distance as §8.6.3's symmetry percentage.

    `100% - ずれ%`: linear, invertible, and carrying no coefficient, so it
    reorients the same measurement rather than scoring it. The metric, the
    export columns and `metric_version` are all untouched by this — only the
    direction the number is read in.

    Clamped at 0% for the degenerate case where the deviation exceeds the
    interocular distance itself; `d` keeps its unbounded value everywhere
    else (§8.6.3).
    """
    if value is None:
        return "—"
    return f"{min(100.0, max(0.0, (1.0 - value) * 100.0)):.2f}%"


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

_STATUS_HEADLINE: Final[dict[Status, str]] = {
    Status.OK: "計測中です。",
    Status.HOLD: "撮影条件を調整してください。",
    Status.NO_FACE: "顔が見つかりません。",
    Status.MULTI_FACE: "2 名以上が写っています。",
    Status.ERROR: "このフレームは処理できませんでした。",
}


def render_guidance(outcome: AnalysisOutcome) -> None:
    """Show what to fix before a score can be produced (Guide before score).

    §5.3's actionable wording comes from `ui.guidance.REASON_GUIDANCE`, the
    same §6.3 table the live banner reads, so the detail panel and the banner
    never describe one condition two ways — and a reason code added to the
    quality gate cannot silently render as nothing here.
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
    """Show one tracking message at the severity §5.2 assigns to its state.

    Camera and Image word the same states differently (a still photo cannot
    be asked to "hold still"), so the caller owns ``messages`` — but the
    colour each state maps to is a product rule, not a per-section choice,
    and lives here so the two sections cannot drift apart.
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
    estimate: DemographicEstimate | None,
) -> None:
    """FR-013/FR-014: overall distance, gender and age, right beside the media.

    Shared by Camera and Image so both use the same headline layout (§5.1).
    ``estimate`` is taken whole rather than as pre-split gender/age fields:
    §10.8 keeps the two demographic values one observation, and both call
    sites had otherwise been unpacking it identically. Missing values show
    `—` rather than a stale or zeroed number (NFR-013).
    """
    score_column, gender_column, age_column = st.columns(3)
    score_column.metric(
        "全体の対称度",
        format_symmetry(overall),
        # §5.3: the deviation is always shown alongside, so the number can be
        # read in either direction and "100%" cannot pass as a target reached.
        help=(
            "100% から、左右対応点のずれ（両眼間隔比）を引いた値です。"
            "100% は数学上の上限で目標値ではありません。"
        ),
    )
    gender_column.metric(
        "参考性別",
        "—" if estimate is None else GENDER_LABELS[estimate.gender],
    )
    age_column.metric(
        "参考年齢",
        "—" if estimate is None else f"{estimate.estimated_age_years:.0f}歳",
    )
    st.caption(f"ずれ {format_distance(overall)}（対称度との和は 100%）")


def render_score_details(outcome: AnalysisOutcome, *, show_smoothed: bool = True) -> None:
    """Render everything about the 左右差 besides the headline total.

    The headline total is `render_headline_metrics`, shown beside the media;
    this renders the part breakdown, pose, and skin quality that belong in
    the full-width detail section below it (§5.1, FR-013, AC-LAYOUT-01).
    """
    st.subheader(SCORE_HEADING)
    st.info(
        "試験中の指標です。撮影条件で値が動くため、数値の絶対値ではなく"
        "同一人物・同一環境での相対変化だけを参照してください。"
    )

    symmetry = outcome.symmetry
    if symmetry is None or symmetry.overall is None:
        render_guidance(outcome)
        render_pose(outcome)
        return

    if show_smoothed:
        st.caption(
            f"平滑化前の生値 {format_symmetry(symmetry.overall)}"
            f"（ずれ {format_distance(symmetry.overall)}）"
        )

    # §8.6: a diagnostic breakdown of where the deviation sits. These do not
    # add up to the headline value, which is folded once over the pooled pairs.
    st.caption("部位別の対称度（全体値の内訳ではなく、ずれの分布を見るための独立した値です）")
    columns = st.columns(len(PART_NAMES))
    for column, name in zip(columns, PART_NAMES, strict=True):
        distance = symmetry.distance(name)
        column.metric(
            _PART_LABELS[name],
            format_symmetry(distance),
            help=f"ずれ {format_distance(distance)}",
        )

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
        "部位別の値を平均しても全体の値にはなりません（全体は全対応ペアをまとめて 1 回だけ"
        "折りたたんで求めるためです）。"
    )
    st.caption(
        "同一人物・同一環境内の相対的な傾向を見るための値です。"
        "他人との比較、医学的評価、魅力度の判断には使用できません。"
    )
    st.caption(f"metric {outcome.metric_version} · landmark map {outcome.landmark_map_version}")


def render_symmetrized_image(view: SymmetrizationView, *, requested: bool) -> None:
    """FR-035: the warped image, framed as the manipulation it is.

    Shared by Camera and Image so the one place this wording matters cannot
    drift between them. §5.3/§10.7: a symmetrized photo is the output most
    likely to be mistaken for a "corrected" or "better" face, so it is
    labelled as a geometric deformation and never shown as a before/after.

    Fails soft: when the warp was requested but the displacement guard or the
    imaging step refused, the reason is shown and the point overlay above
    stands on its own (AC-SYMM-02).
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
    """Render the §10.8 age/gender reference panel (Image and Camera share this).

    `show_distribution` draws the two-class probability chart. The live camera
    panel leaves it off: it repaints at 5 Hz while this value changes at most
    once a second, and the chart costs far more than every other element here
    combined. The same numbers stay available as text either way.
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
