"""Pillow drawing helpers shared by the review, evaluation, and correction-review tools.

Two styles live here:

* a dark HUD panel below the camera frame (debug and review videos);
* the minimal showcase overlay drawn on top of the frame: a route-condition
  badge and the executed steering and throttle (used for the README media).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# HUD palette
TEXT = (242, 242, 242)
MUTED = (178, 178, 178)
HUD_BG = (18, 18, 18)
LINE = (72, 72, 72)
BLUE = (80, 160, 255)  # human / ground truth
ORANGE = (255, 170, 50)  # model
GREEN = (92, 215, 132)  # executed / target / ok
YELLOW = (245, 196, 72)  # warning
RED = (240, 88, 88)  # error / stop

_BOLD_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "DejaVuSans-Bold.ttf",
)


@cache
def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Return DejaVu Sans (bold: DejaVu or Arial Bold), or Pillow's default font."""
    candidates = _BOLD_FONTS if bold else ("DejaVuSans.ttf",)
    for candidate in candidates:
        if "/" in candidate and not Path(candidate).is_file():
            continue
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale ``image`` to fit ``width`` x ``height`` without distortion, centered on black."""
    scale = min(width / image.width, height / image.height)
    resized_w = max(1, round(image.width * scale))
    resized_h = max(1, round(image.height * scale))
    resized = image.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - resized_w) // 2, (height - resized_h) // 2))
    return canvas


def hud_canvas(
    image: Image.Image, width: int, image_height: int, hud_height: int
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """The camera image fitted into the top part and a dark panel below it, split by a line."""
    canvas = Image.new("RGB", (width, image_height + hud_height), HUD_BG)
    canvas.paste(fit_image(image, width, image_height), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line((0, image_height, width, image_height), fill=LINE, width=1)
    return canvas, draw


def draw_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float | None,
    color: tuple[int, int, int],
    low: float = -1.0,
    high: float = 1.0,
) -> None:
    """Horizontal bar filled from zero to ``value``; nothing is filled for ``None``."""
    draw.rectangle((x, y, x + width, y + height), outline=LINE, fill=(32, 32, 32))
    zero = x + round((0.0 - low) / (high - low) * width)
    draw.line((zero, y, zero, y + height), fill=(120, 120, 120))
    if value is None:
        return
    position = x + round((max(low, min(high, value)) - low) / (high - low) * width)
    draw.rectangle((min(zero, position), y + 2, max(zero, position), y + height - 2), fill=color)


def save_frame(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)
    return path


# ---------------------------------------------------------------------------
# Showcase overlay
# ---------------------------------------------------------------------------

_PANEL = (12, 12, 14, 170)
_OVERLAY_TEXT = (245, 245, 245, 255)
_OVERLAY_MUTED = (190, 190, 195, 255)
_TRACK = (255, 255, 255, 60)
_FORWARD = (92, 215, 132, 255)
_TURN = (255, 176, 46, 255)


def _arrow(draw: ImageDraw.ImageDraw, condition: str, cx: float, cy: float, r: float, fill) -> None:
    """Solid arrow for FORWARD / LEFT / RIGHT centered on (cx, cy)."""
    shaft = r * 0.36
    if condition == "FORWARD":
        points = [
            (cx, cy - r),
            (cx + r, cy),
            (cx + shaft, cy),
            (cx + shaft, cy + r),
            (cx - shaft, cy + r),
            (cx - shaft, cy),
            (cx - r, cy),
        ]
    else:
        sign = 1 if condition == "RIGHT" else -1
        points = [
            (cx + sign * r, cy),
            (cx, cy - r),
            (cx, cy - shaft),
            (cx - sign * r, cy - shaft),
            (cx - sign * r, cy + shaft),
            (cx, cy + shaft),
            (cx, cy + r),
        ]
    draw.polygon(points, fill=fill)


def draw_condition_badge(image: Image.Image, condition: str, *, scale: float = 1.0) -> Image.Image:
    """Draw a rounded "CONDITION" badge with an arrow in the top-left corner."""
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    label_font = load_font(int(11 * scale), bold=True)
    value_font = load_font(int(22 * scale), bold=True)
    color = _FORWARD if condition == "FORWARD" else _TURN

    pad = 10 * scale
    x0, y0 = 12 * scale, 12 * scale
    icon = 22 * scale
    width = pad + icon + 8 * scale + draw.textlength(condition, font=value_font) + pad
    height = pad * 0.7 + 13 * scale + 26 * scale + pad * 0.7
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=10 * scale, fill=_PANEL)
    draw.text((x0 + pad, y0 + pad * 0.7), "CONDITION", font=label_font, fill=_OVERLAY_MUTED)
    cy = y0 + pad * 0.7 + 26 * scale
    _arrow(draw, condition, x0 + pad + icon / 2, cy, icon / 2, color)
    draw.text((x0 + pad + icon + 8 * scale, cy), condition, font=value_font, fill=color, anchor="lm")
    return Image.alpha_composite(base, layer)


def draw_command_bars(
    image: Image.Image, *, throttle: float, steering: float, scale: float = 1.0
) -> Image.Image:
    """Draw steering (centered, -1..1) and throttle (0..1) bars along the bottom edge."""
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = base.size
    label_font = load_font(int(12 * scale), bold=True)
    value_font = load_font(int(14 * scale), bold=True)

    margin = 12 * scale
    row_h = 22 * scale
    panel_h = 2 * row_h + 18 * scale
    px0, py0, px1, py1 = margin, height - margin - panel_h, width - margin, height - margin
    draw.rounded_rectangle((px0, py0, px1, py1), radius=10 * scale, fill=_PANEL)

    bar_x0 = px0 + 10 * scale + 74 * scale
    bar_x1 = px1 - 10 * scale - 52 * scale
    bar_h = 8 * scale

    def row(y: float, label: str, value: float, *, centered: bool, color, fmt: str) -> None:
        draw.text((px0 + 10 * scale, y), label, font=label_font, fill=_OVERLAY_MUTED, anchor="lm")
        draw.rounded_rectangle((bar_x0, y - bar_h / 2, bar_x1, y + bar_h / 2), radius=bar_h / 2, fill=_TRACK)
        if centered:
            mid = (bar_x0 + bar_x1) / 2
            end = mid + max(-1.0, min(1.0, value)) * (bar_x1 - bar_x0) / 2
            lo, hi = sorted((mid, end))
            draw.rounded_rectangle(
                (lo, y - bar_h / 2, max(hi, lo + 1), y + bar_h / 2), radius=bar_h / 2, fill=color
            )
            draw.line((mid, y - bar_h, mid, y + bar_h), fill=_OVERLAY_TEXT, width=max(1, int(2 * scale)))
        else:
            end = bar_x0 + max(0.0, min(1.0, value)) * (bar_x1 - bar_x0)
            draw.rounded_rectangle(
                (bar_x0, y - bar_h / 2, max(end, bar_x0 + 1), y + bar_h / 2), radius=bar_h / 2, fill=color
            )
        draw.text((px1 - 10 * scale, y), format(value, fmt), font=value_font, fill=_OVERLAY_TEXT, anchor="rm")

    row(py0 + 9 * scale + row_h / 2, "STEERING", steering, centered=True, color=_TURN, fmt="+.2f")
    row(py0 + 9 * scale + row_h * 1.5, "THROTTLE", throttle, centered=False, color=_FORWARD, fmt=".2f")
    return Image.alpha_composite(base, layer)
