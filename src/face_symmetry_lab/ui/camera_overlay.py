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


def metric_overlay_lines(
    outcome: AnalysisOutcome,
    *,
    current_sample: bool = True,
    demographic: DemographicEstimate | None = None,
) -> tuple[str, ...]:
    """Format the same score and pose data exposed by the text panel (§5.4).

    `demographic` appends one reference age/gender line (FR-033) so the same
    information a browser's native video fullscreen would otherwise hide
    stays visible — it is never used to judge or identify anyone (§10.7).
    """
    symmetry = outcome.symmetry
    overall = outcome.overall_smooth
    raw = outcome.overall_raw
    if overall is None and symmetry is not None:
        overall = symmetry.overall
    parts = {
        name: symmetry.part(name) if symmetry is not None else None
        for name in ("contour", "eyes", "nose", "mouth")
    }
    pose = outcome.pose
    skin_quality = outcome.skin_quality
    #: Surfaces the EMA gap (§8.7) live so a jump between a photo and the
    #: camera view can be told apart from smoothing lag instead of guessed at.
    overall_line = f"OVERALL {_number(overall)}"
    if raw is not None:
        overall_line += f" · RAW {_number(raw)}"
    lines = [
        f"{'LIVE SAMPLE' if current_sample else 'LAST SAMPLE'} | {outcome.state.status.value}",
        overall_line,
        " ".join(
            (
                f"C {_number(parts['contour'])}",
                f"E {_number(parts['eyes'])}",
                f"N {_number(parts['nose'])}",
                f"M {_number(parts['mouth'])}",
            )
        ),
        f"SKIN REF {_number(skin_quality.score if skin_quality else None)} | CAPTURE-DEPENDENT",
        " ".join(
            (
                f"YAW {_number(pose.yaw_deg if pose else None, signed=True)}",
                f"PITCH {_number(pose.pitch_deg if pose else None, signed=True)}",
                f"ROLL {_number(pose.roll_deg if pose else None, signed=True)}",
            )
        ),
    ]
    if demographic is not None:
        lines.append(
            f"REF AGE {demographic.estimated_age_years:.0f} · "
            f"REF GENDER {demographic.gender.value.upper()}"
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


def _draw_guidance_banner(
    image: Image.Image,
    message: str,
    *,
    width: int,
    height: int,
    scale: float,
    panel_position: OverlayPosition,
) -> Image.Image:
    """Draw a short instruction banner opposite the score panel (§14.3e).

    Anchored to the top when the panel is at the bottom, and vice versa, so
    the two never overlap regardless of which corner the panel was moved to.
    """
    font = _font(max(16, round(24 * scale)))
    drawing_probe = ImageDraw.Draw(image)
    box = drawing_probe.textbbox((0, 0), message, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    padding_x = max(16, round(24 * scale))
    padding_y = max(10, round(14 * scale))
    banner_width = min(width, text_width + padding_x * 2)
    banner_height = text_height + padding_y * 2
    origin_x = max(0, (width - banner_width) // 2)
    at_top = panel_position in (OverlayPosition.BOTTOM_LEFT, OverlayPosition.BOTTOM_RIGHT)
    margin = max(6, round(6 * scale))
    origin_y = margin if at_top else height - banner_height - margin

    banner = Image.new("RGBA", image.size, (0, 0, 0, 0))
    banner_drawing = ImageDraw.Draw(banner)
    banner_drawing.rounded_rectangle(
        (origin_x, origin_y, origin_x + banner_width - 1, origin_y + banner_height - 1),
        radius=max(8, round(14 * scale)),
        fill=(120, 53, 15, 235),
        outline=(251, 191, 36, 255),
        width=max(2, round(3 * scale)),
    )
    composited = Image.alpha_composite(image, banner)
    drawing = ImageDraw.Draw(composited)
    drawing.text(
        (origin_x + padding_x, origin_y + padding_y),
        message,
        font=font,
        fill=(255, 251, 235, 255),
    )
    return composited


def draw_camera_metrics(
    rgb_image: NDArray[np.uint8],
    outcome: AnalysisOutcome | None,
    *,
    current_sample: bool = True,
    demographic: DemographicEstimate | None = None,
    position: OverlayPosition = OverlayPosition.TOP_LEFT,
) -> NDArray[np.uint8]:
    """Draw score/guidance/reference values over a copy of the camera image."""
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
    detail_size = max(15, round(21 * scale))
    status_font = _font(status_size)
    overall_font = _font(overall_size)
    detail_font = _font(detail_size)
    fonts = [status_font, overall_font] + [detail_font] * (len(lines) - 2)
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

    message = guidance_message(outcome.state)
    if message is not None:
        image = _draw_guidance_banner(
            image, message, width=width, height=height, scale=scale, panel_position=position
        )
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
