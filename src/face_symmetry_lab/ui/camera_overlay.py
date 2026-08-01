"""High-contrast numeric presentation painted into the realtime preview."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from face_symmetry_lab.application.analyzer import AnalysisOutcome


@lru_cache(maxsize=4)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def metric_overlay_lines(outcome: AnalysisOutcome) -> tuple[str, ...]:
    """Format the same score and pose data exposed by the text panel (§5.4)."""
    symmetry = outcome.symmetry
    overall = outcome.overall_smooth
    if overall is None and symmetry is not None:
        overall = symmetry.overall
    parts = {
        name: symmetry.part(name) if symmetry is not None else None
        for name in ("contour", "eyes", "nose", "mouth")
    }
    pose = outcome.pose
    return (
        f"STATUS {outcome.state.status.value}",
        f"OVERALL {_number(overall)}",
        " ".join(
            (
                f"C {_number(parts['contour'])}",
                f"E {_number(parts['eyes'])}",
                f"N {_number(parts['nose'])}",
                f"M {_number(parts['mouth'])}",
            )
        ),
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
) -> NDArray[np.uint8]:
    """Draw readable numeric values over a copy of the camera image."""
    painted = rgb_image.copy()
    if outcome is None or painted.ndim != 3 or painted.shape[2] != 3:
        return painted

    height, width = painted.shape[:2]
    if height < 1 or width < 1:
        return painted
    lines = metric_overlay_lines(outcome)
    font_size = max(14, min(24, int(width / 1280 * 24)))
    font = _font(font_size)
    line_height = font_size + 10
    padding = 12
    image = Image.fromarray(painted)
    drawing = ImageDraw.Draw(image)
    boxes = [drawing.textbbox((0, 0), line, font=font) for line in lines]
    panel_width = min(width, max(box[2] - box[0] for box in boxes) + padding * 2)
    panel_height = min(height, line_height * len(lines) + padding * 2)
    drawing.rectangle((0, 0, panel_width - 1, panel_height - 1), fill=(8, 8, 8))
    for index, line in enumerate(lines):
        top = padding + line_height * index
        if top >= height:
            break
        drawing.text((padding, top), line, font=font, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8).copy()
