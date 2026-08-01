"""High-contrast numeric presentation painted into the realtime preview."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from face_symmetry_lab.application.analyzer import AnalysisOutcome


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
) -> tuple[str, ...]:
    """Format the same score and pose data exposed by the text panel (§5.4)."""
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
    return (
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
    )


def draw_camera_metrics(
    rgb_image: NDArray[np.uint8],
    outcome: AnalysisOutcome | None,
    *,
    current_sample: bool = True,
) -> NDArray[np.uint8]:
    """Draw readable numeric values over a copy of the camera image."""
    painted = rgb_image.copy()
    if outcome is None or painted.ndim != 3 or painted.shape[2] != 3:
        return painted

    height, width = painted.shape[:2]
    if height < 1 or width < 1:
        return painted
    lines = metric_overlay_lines(outcome, current_sample=current_sample)
    scale = max(0.65, min(1.0, width / 1280, height / 720))
    status_size = max(16, round(22 * scale))
    overall_size = max(30, round(50 * scale))
    detail_size = max(15, round(21 * scale))
    status_font = _font(status_size)
    overall_font = _font(overall_size)
    detail_font = _font(detail_size)
    fonts = (status_font, overall_font, detail_font, detail_font, detail_font)
    padding = max(12, round(18 * scale))
    gaps = (
        max(5, round(8 * scale)),
        max(7, round(11 * scale)),
        max(4, round(6 * scale)),
        max(4, round(6 * scale)),
    )

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

    accent = {
        "OK": (56, 189, 248, 255),
        "HOLD": (251, 191, 36, 255),
        "ERROR": (248, 113, 113, 255),
    }.get(outcome.state.status.value, (148, 163, 184, 255))
    panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
    panel_drawing = ImageDraw.Draw(panel)
    panel_drawing.rounded_rectangle(
        (0, 0, panel_width - 1, panel_height - 1),
        radius=max(10, round(18 * scale)),
        fill=(5, 12, 24, 232),
        outline=accent,
        width=max(2, round(3 * scale)),
    )
    image = Image.alpha_composite(image, panel)
    drawing = ImageDraw.Draw(image)
    top = padding
    for index, (line, font, line_height) in enumerate(zip(lines, fonts, line_heights, strict=True)):
        if top >= height:
            break
        fill = accent if index == 0 else (248, 250, 252, 255)
        drawing.text((padding, top), line, font=font, fill=fill)
        if index < len(gaps):
            top += line_height + gaps[index]
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
