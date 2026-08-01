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
from face_symmetry_lab.domain.models import ReasonCode, Status

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


def render_scores(outcome: AnalysisOutcome, *, show_smoothed: bool = True) -> None:
    """Render the overall and per-part values, or say why they are missing."""
    st.subheader(SCORE_HEADING)
    if not outcome.sensitivity_frozen:
        st.info(
            "試験中の指標です。感度係数は固定評価セットで未凍結のため、"
            "数値の絶対値ではなく同一条件での相対変化だけを参照してください。"
        )

    symmetry = outcome.symmetry
    if symmetry is None or symmetry.overall is None:
        st.metric("全体", "—")
        render_guidance(outcome)
        render_pose(outcome)
        return

    displayed = outcome.overall_smooth if show_smoothed else symmetry.overall
    st.metric("全体", f"{(displayed or symmetry.overall):.1f}")
    if show_smoothed:
        st.caption(f"平滑化前の生値 {symmetry.overall:.1f}")

    columns = st.columns(len(PART_NAMES))
    for column, name in zip(columns, PART_NAMES, strict=True):
        value = symmetry.part(name)
        column.metric(_PART_LABELS[name], "—" if value is None else f"{value:.1f}")

    render_pose(outcome)
    st.caption(
        "同一人物・同一環境内の相対的な傾向を見るための値です。"
        "他人との比較、医学的評価、魅力度の判断には使用できません。"
    )
    st.caption(f"metric {outcome.metric_version} · landmark map {outcome.landmark_map_version}")
