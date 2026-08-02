"""High-contrast numeric presentation painted into the realtime preview."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.models import AnalysisState, DemographicEstimate, ReasonCode


class OverlayPosition(StrEnum):
    """§15.1, FR-032: where the movable score panel is drawn. Display-only."""

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"


#: FR-031: every `ReasonCode` maps to one short, actionable Japanese
#: instruction. No new judgement is made here — this only translates a
#: reason the quality gate (`application.quality.QualityGate`) already
#: computed (§6.2, §6.3) into guidance text.
_REASON_GUIDANCE: Final[dict[ReasonCode, str]] = {
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
    return _REASON_GUIDANCE.get(state.reason)


@lru_cache(maxsize=12)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


#: Index of the reference age/gender line within `metric_overlay_lines`'
#: output when a `demographic` estimate is supplied — `draw_camera_metrics`
#: uses this to give that one line its own, more prominent font tier.
DEMOGRAPHIC_LINE_INDEX: Final[int] = 3


def metric_overlay_lines(
    outcome: AnalysisOutcome,
    *,
    current_sample: bool = True,
    demographic: DemographicEstimate | None = None,
) -> tuple[str, ...]:
    """Format the score/reference data shown in the movable video panel (§5.4).

    Ordered so the items the panel is meant to make clear at a glance —
    overall score, part breakdown, then reference age/gender (FR-033, right
    after the scores so it isn't buried under pose/skin detail) — come
    before the lower-priority skin/pose readout, which stays available for
    completeness but is drawn smaller (`draw_camera_metrics`). `demographic`
    is never used to judge or identify anyone (§10.7).
    """
    symmetry = outcome.symmetry
    overall = outcome.overall_smooth
    if overall is None and symmetry is not None:
        overall = symmetry.overall
    parts = {
        name: symmetry.part(name) if symmetry is not None else None
        for name in ("contour", "eyes", "nose", "mouth")
    }
    pose = outcome.pose
    skin_quality = outcome.skin_quality
    lines = [
        f"{'LIVE SAMPLE' if current_sample else 'LAST SAMPLE'} | {outcome.state.status.value}",
        f"OVERALL {_number(overall)}",
        " ".join(
            (
                f"C {_number(parts['contour'])}",
                f"E {_number(parts['eyes'])}",
                f"N {_number(parts['nose'])}",
                f"M {_number(parts['mouth'])}",
            )
        ),
    ]
    if demographic is not None:
        assert len(lines) == DEMOGRAPHIC_LINE_INDEX
        lines.append(
            f"REF AGE {demographic.estimated_age_years:.0f} · "
            f"REF GENDER {demographic.gender.value.upper()}"
        )
    lines.append(
        f"SKIN REF {_number(skin_quality.score if skin_quality else None)} | CAPTURE-DEPENDENT"
    )
    lines.append(
        " ".join(
            (
                f"YAW {_number(pose.yaw_deg if pose else None, signed=True)}",
                f"PITCH {_number(pose.pitch_deg if pose else None, signed=True)}",
                f"ROLL {_number(pose.roll_deg if pose else None, signed=True)}",
            )
        )
    )
    return tuple(lines)


def _panel_origin(
    position: OverlayPosition,
    *,
    width: int,
    height: int,
    panel_width: int,
    panel_height: int,
) -> tuple[int, int]:
    at_right = position in (OverlayPosition.TOP_RIGHT, OverlayPosition.BOTTOM_RIGHT)
    at_bottom = position in (OverlayPosition.BOTTOM_LEFT, OverlayPosition.BOTTOM_RIGHT)
    x = width - panel_width if at_right else 0
    y = height - panel_height if at_bottom else 0
    return x, y


def draw_camera_metrics(
    rgb_image: NDArray[np.uint8],
    outcome: AnalysisOutcome | None,
    *,
    current_sample: bool = True,
    demographic: DemographicEstimate | None = None,
    position: OverlayPosition = OverlayPosition.TOP_LEFT,
) -> NDArray[np.uint8]:
    """Draw score/reference values over a copy of the camera image.

    Position guidance (§14.3e, FR-031) is shown only in the "ライブ解析" side
    panel, never here — PIL's default bitmap font (`_font`) has no Japanese
    glyphs, so Japanese text drawn onto the frame renders as tofu boxes.
    """
    painted = rgb_image.copy()
    if outcome is None or painted.ndim != 3 or painted.shape[2] != 3:
        return painted

    height, width = painted.shape[:2]
    if height < 1 or width < 1:
        return painted
    lines = metric_overlay_lines(outcome, current_sample=current_sample, demographic=demographic)
    scale = max(0.65, min(1.0, width / 1280, height / 720))
    status_size = max(16, round(22 * scale))
    overall_size = max(30, round(50 * scale))
    #: Between `overall_size` and `detail_size`: reference age/gender (FR-033)
    #: reads clearly at a glance without competing with the overall score.
    emphasis_size = max(22, round(34 * scale))
    detail_size = max(13, round(18 * scale))
    status_font = _font(status_size)
    overall_font = _font(overall_size)
    emphasis_font = _font(emphasis_size)
    detail_font = _font(detail_size)
    has_demographic_line = demographic is not None
    fonts = [status_font, overall_font] + [
        emphasis_font if has_demographic_line and i == DEMOGRAPHIC_LINE_INDEX else detail_font
        for i in range(2, len(lines))
    ]
    padding = max(12, round(18 * scale))
    gap_after_status = max(5, round(8 * scale))
    gap_after_overall = max(7, round(11 * scale))
    gap_uniform = max(4, round(6 * scale))
    gaps = [gap_after_status, gap_after_overall] + [gap_uniform] * (len(lines) - 3)

    image = Image.fromarray(painted).convert("RGBA")
    measuring = ImageDraw.Draw(image)
    boxes = [
        measuring.textbbox((0, 0), line, font=font) for line, font in zip(lines, fonts, strict=True)
    ]
    line_heights = [box[3] - box[1] for box in boxes]
    content_height = sum(line_heights) + sum(gaps) + padding * 2
    panel_height = min(height, max(round(258 * scale), content_height))
    panel_width = min(
        width,
        max(round(540 * scale), max(box[2] - box[0] for box in boxes) + padding * 2),
    )
    origin_x, origin_y = _panel_origin(
        position, width=width, height=height, panel_width=panel_width, panel_height=panel_height
    )

    accent = {
        "OK": (56, 189, 248, 255),
        "HOLD": (251, 191, 36, 255),
        "ERROR": (248, 113, 113, 255),
    }.get(outcome.state.status.value, (148, 163, 184, 255))
    panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
    panel_drawing = ImageDraw.Draw(panel)
    panel_drawing.rounded_rectangle(
        (origin_x, origin_y, origin_x + panel_width - 1, origin_y + panel_height - 1),
        radius=max(10, round(18 * scale)),
        fill=(5, 12, 24, 232),
        outline=accent,
        width=max(2, round(3 * scale)),
    )
    image = Image.alpha_composite(image, panel)
    drawing = ImageDraw.Draw(image)
    top = origin_y + padding
    for index, (line, font, line_height) in enumerate(zip(lines, fonts, line_heights, strict=True)):
        if top >= height:
            break
        fill = accent if index == 0 else (248, 250, 252, 255)
        drawing.text((origin_x + padding, top), line, font=font, fill=fill)
        if index < len(gaps):
            top += line_height + gaps[index]

    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
