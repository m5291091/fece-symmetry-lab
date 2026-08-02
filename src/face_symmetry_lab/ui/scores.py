"""Score presentation shared by the camera and still-image sections.

Wording follows §5.3: the metric is introduced as an experimental left/right
difference score, `HOLD` becomes an actionable instruction, and no value
judgement is attached to a high or low number.
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel, ReasonCode, Status

SCORE_HEADING = "左右差スコア（実験値）"

_PART_LABELS: Final[dict[str, str]] = {
    "contour": "輪郭",
    "eyes": "目",
    "nose": "鼻",
    "mouth": "口",
}

#: §5.3 requires actionable guidance instead of raw reason codes.
_REASON_GUIDANCE: Final[dict[ReasonCode, str]] = {
    ReasonCode.NO_FACE: "顔がフレームに入るようにしてください。",
    ReasonCode.MULTIPLE_FACES: "1 名だけが写るようにしてください。",
    ReasonCode.OFF_AXIS_YAW: "顔をカメラの正面へ向けてください。",
    ReasonCode.OFF_AXIS_PITCH: "顎を少し上げる、または下げてください。",
    ReasonCode.OFF_AXIS_ROLL: "頭を水平に戻してください。",
    ReasonCode.TOO_CLOSE: "カメラから少し離れてください。",
    ReasonCode.TOO_FAR: "カメラに少し近づいてください。",
    ReasonCode.UNSTABLE: "そのまま静止してください。",
    ReasonCode.LOW_CONFIDENCE: "明るさと顔の位置を確認してください。",
    ReasonCode.LOW_QUALITY: "この画像では計算できませんでした。",
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
    """Show what to fix before a score can be produced (Guide before score)."""
    st.write(_STATUS_HEADLINE.get(outcome.state.status, ""))
    for code in outcome.state.reason_codes:
        guidance = _REASON_GUIDANCE.get(code)
        if guidance:
            st.markdown(f"- {guidance}")


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
    gender: GenderLabel | None,
    age_years: float | None,
) -> None:
    """FR-013/FR-014: total score, gender and age, right beside the media.

    Shared by Camera and Image so both use the same headline layout (§5.1).
    Missing values show `—` rather than a stale or zeroed number (NFR-013).
    """
    score_column, gender_column, age_column = st.columns(3)
    score_column.metric("全体スコア", "—" if overall is None else f"{overall:.1f}")
    gender_column.metric("参考性別", "—" if gender is None else GENDER_LABELS[gender])
    age_column.metric("参考年齢", "—" if age_years is None else f"{age_years:.0f}歳")


def render_score_details(outcome: AnalysisOutcome, *, show_smoothed: bool = True) -> None:
    """Render everything about the 左右差スコア besides the headline total.

    The headline total is `render_headline_metrics`, shown beside the media;
    this renders the part breakdown, pose, and skin quality that belong in
    the full-width detail section below it (§5.1, FR-013, AC-LAYOUT-01).
    """
    st.subheader(SCORE_HEADING)
    if not outcome.sensitivity_frozen:
        st.info(
            "試験中の指標です。感度係数は固定評価セットで未凍結のため、"
            "数値の絶対値ではなく同一条件での相対変化だけを参照してください。"
        )

    symmetry = outcome.symmetry
    if symmetry is None or symmetry.overall is None:
        render_guidance(outcome)
        render_pose(outcome)
        return

    if show_smoothed:
        st.caption(f"平滑化前の生値 {symmetry.overall:.1f}")

    columns = st.columns(len(PART_NAMES))
    for column, name in zip(columns, PART_NAMES, strict=True):
        value = symmetry.part(name)
        column.metric(_PART_LABELS[name], "—" if value is None else f"{value:.1f}")

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
    st.caption(
        "100 は、正規化後の対応点距離がすべて 0 になった場合の数学上の上限です。"
        "人物の状態や価値を示す尺度ではなく、実写で 100 になることも保証しません。"
    )
    st.caption(
        "同一人物・同一環境内の相対的な傾向を見るための値です。"
        "他人との比較、医学的評価、魅力度の判断には使用できません。"
    )
    st.caption(f"metric {outcome.metric_version} · landmark map {outcome.landmark_map_version}")


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
