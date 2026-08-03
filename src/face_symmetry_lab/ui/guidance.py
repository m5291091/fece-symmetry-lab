"""Turn a quality-gate reason into one actionable Japanese instruction.

`REASON_GUIDANCE` is §6.3's 既定ガイド column, and the only place the app
answers "what do we tell the user about this reason code": the live "ライブ
解析" banner (FR-031) reads the primary reason through `guidance_message`,
and `ui.scores.render_guidance` walks every blocking reason through the same
table, so the two can never word the same condition differently.

Nothing here is ever drawn onto the video *frame*: FR-013 keeps the camera
preview a pass-through, and Japanese text could not be painted onto it anyway
(PIL's default bitmap font carries no CJK glyphs). That reasoning is about
pixels, and it still holds.

It does not extend to the DOM layer. §5.2's overlay exception lets
`ui.camera` composite a HUD and a gate-coloured border *over* the video with
CSS (§14.3e), which the browser draws and the frame path never sees — so it
adds nothing to `processing_ms` and has no font constraint. This guidance
text stays in the side panel regardless: §5.4 requires the state to be
readable without relying on the overlay or on colour.
"""

from __future__ import annotations

from typing import Final

from face_symmetry_lab.domain.models import AnalysisState, ReasonCode

#: Every `ReasonCode` maps to one short, actionable instruction. No new
#: judgement is made here — this only translates a reason the quality gate
#: (`application.quality.QualityGate`) already computed (§6.2, §6.3).
REASON_GUIDANCE: Final[dict[ReasonCode, str]] = {
    ReasonCode.NO_FACE: "顔がカメラに映るようにしてください",
    ReasonCode.MULTIPLE_FACES: "1人だけがカメラに映るようにしてください",
    ReasonCode.POSTURE: "顔を正面に向けてください",
    ReasonCode.DISTANCE: "カメラとの距離を調整してください",
    ReasonCode.LOW_QUALITY: "撮影条件を確認してください",
    ReasonCode.OFF_AXIS_YAW: "顔を正面（左右方向）に向けてください",
    ReasonCode.OFF_AXIS_PITCH: "顔を正面（上下方向）に向けてください",
    ReasonCode.OFF_AXIS_ROLL: "頭の傾きをまっすぐにしてください",
    ReasonCode.TOO_CLOSE: "もう少しカメラから離れてください",
    ReasonCode.TOO_FAR: "もう少しカメラに近づいてください",
    ReasonCode.UNSTABLE: "そのままの位置で少し静止してください",
    ReasonCode.LOW_CONFIDENCE: "顔がはっきり映るようにしてください",
    ReasonCode.LOW_LIGHT: "明るい場所へ移動するか、照明を調整してください",
    ReasonCode.OVEREXPOSED: "光が強すぎます。照明を弱めてください",
    ReasonCode.BLUR: "カメラを固定し、ピントを合わせてください",
    ReasonCode.STALE_RESULT: "処理が遅延しています。しばらくお待ちください",
}


def guidance_message(state: AnalysisState) -> str | None:
    """A short Japanese instruction for `state.reason`, or None when satisfied.

    §14.3e: reuses the quality gate's own primary reason (already the first
    of possibly several concurrent `reason_codes`) rather than deriving a new
    priority order, so this never disagrees with the status shown elsewhere.
    """
    if state.reason is None:
        return None
    return REASON_GUIDANCE.get(state.reason)
