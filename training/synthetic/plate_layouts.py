"""Synthetic USA license-plate layout definitions and graphics rendering.

Defines realistic USA-compatible license-plate styles, dimensions, typography,
color palettes, headers, registration decals, and alphanumeric pattern generators.
All layouts are clearly designated as synthetic/compatible templates.
"""
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import os
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class PlateStyle:
    """Color palette and visual elements for a synthetic plate style."""

    bg_color_top: Tuple[int, int, int]
    bg_color_bottom: Tuple[int, int, int]
    text_color: Tuple[int, int, int]
    header_color: Tuple[int, int, int]
    border_color: Tuple[int, int, int]
    bevel_color: Tuple[int, int, int]
    has_gradient: bool = False
    has_header_banner: bool = False
    banner_color: Optional[Tuple[int, int, int]] = None
    has_bottom_stripe: bool = False
    bottom_stripe_color: Optional[Tuple[int, int, int]] = None
    sticker_colors: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(180, 40, 40), (40, 100, 180)]
    )


@dataclass
class PlateLayout:
    """Complete specification for a synthetic license plate layout."""

    layout_id: str
    state_name: str
    format_pattern: str  # D = Digit (0-9), L = Letter (A-Z)
    style: PlateStyle
    motto_text: str = ""
    separator_char: str = ""  # e.g. "-", " ", "."
    font_name_hint: str = "bold"


# --- Predefined Synthetic USA-Compatible Layouts ---
SYNTHETIC_LAYOUTS: Dict[str, PlateLayout] = {
    "SYN_CALIFORNIA_COMPAT": PlateLayout(
        layout_id="SYN_CALIFORNIA_COMPAT",
        state_name="CALIFORNIA",
        format_pattern="DLLLDDD",
        separator_char="",
        motto_text="dmv.ca.gov",
        style=PlateStyle(
            bg_color_top=(252, 252, 252),
            bg_color_bottom=(248, 248, 248),
            text_color=(15, 30, 80),
            header_color=(190, 25, 25),
            border_color=(40, 40, 40),
            bevel_color=(210, 210, 210),
            has_gradient=False,
            sticker_colors=[(200, 40, 40), (40, 120, 200)],
        ),
    ),
    "SYN_TEXAS_COMPAT": PlateLayout(
        layout_id="SYN_TEXAS_COMPAT",
        state_name="TEXAS",
        format_pattern="LLLDDDD",
        separator_char="-",
        motto_text="THE LONE STAR STATE",
        style=PlateStyle(
            bg_color_top=(245, 245, 245),
            bg_color_bottom=(240, 240, 240),
            text_color=(25, 25, 25),
            header_color=(15, 30, 70),
            border_color=(35, 35, 35),
            bevel_color=(200, 200, 200),
            has_gradient=False,
            sticker_colors=[(40, 140, 60), (210, 160, 30)],
        ),
    ),
    "SYN_NEWYORK_COMPAT": PlateLayout(
        layout_id="SYN_NEWYORK_COMPAT",
        state_name="NEW YORK",
        format_pattern="LLLDDDD",
        separator_char="-",
        motto_text="EXCELSIOR",
        style=PlateStyle(
            bg_color_top=(245, 195, 35),
            bg_color_bottom=(240, 185, 25),
            text_color=(10, 25, 70),
            header_color=(10, 25, 70),
            border_color=(10, 25, 70),
            bevel_color=(255, 220, 80),
            has_gradient=True,
            has_header_banner=True,
            banner_color=(15, 35, 90),
            sticker_colors=[(220, 50, 50), (40, 40, 40)],
        ),
    ),
    "SYN_FLORIDA_COMPAT": PlateLayout(
        layout_id="SYN_FLORIDA_COMPAT",
        state_name="FLORIDA",
        format_pattern="LLLDDD",
        separator_char=" ",
        motto_text="SUNSHINE STATE",
        style=PlateStyle(
            bg_color_top=(252, 252, 252),
            bg_color_bottom=(248, 248, 248),
            text_color=(20, 90, 45),
            header_color=(220, 100, 20),
            border_color=(35, 35, 35),
            bevel_color=(220, 220, 220),
            has_gradient=False,
            sticker_colors=[(220, 120, 30), (30, 130, 60)],
        ),
    ),
    "SYN_PENNSYLVANIA_COMPAT": PlateLayout(
        layout_id="SYN_PENNSYLVANIA_COMPAT",
        state_name="PENNSYLVANIA",
        format_pattern="LLLDDDD",
        separator_char="-",
        motto_text="visitPA.com",
        style=PlateStyle(
            bg_color_top=(250, 250, 250),
            bg_color_bottom=(245, 245, 245),
            text_color=(15, 30, 75),
            header_color=(255, 255, 255),
            border_color=(35, 35, 35),
            bevel_color=(215, 215, 215),
            has_header_banner=True,
            banner_color=(20, 45, 110),
            has_bottom_stripe=True,
            bottom_stripe_color=(240, 190, 30),
            sticker_colors=[(180, 30, 30), (30, 80, 160)],
        ),
    ),
    "SYN_PACIFICA_COMPAT": PlateLayout(
        layout_id="SYN_PACIFICA_COMPAT",
        state_name="PACIFICA",
        format_pattern="DLLDDDD",
        separator_char="-",
        motto_text="THE OCEAN STATE",
        style=PlateStyle(
            bg_color_top=(200, 225, 245),
            bg_color_bottom=(250, 250, 250),
            text_color=(10, 35, 80),
            header_color=(10, 35, 80),
            border_color=(30, 30, 30),
            bevel_color=(230, 240, 255),
            has_gradient=True,
            sticker_colors=[(40, 150, 210), (210, 50, 50)],
        ),
    ),
    "SYN_COLUMBIA_COMPAT": PlateLayout(
        layout_id="SYN_COLUMBIA_COMPAT",
        state_name="COLUMBIA",
        format_pattern="LLLDDD",
        separator_char="-",
        motto_text="EVERGREEN STATE",
        style=PlateStyle(
            bg_color_top=(245, 250, 245),
            bg_color_bottom=(240, 245, 240),
            text_color=(15, 65, 30),
            header_color=(15, 65, 30),
            border_color=(15, 65, 30),
            bevel_color=(210, 235, 215),
            has_gradient=False,
            sticker_colors=[(30, 130, 50), (200, 140, 30)],
        ),
    ),
    "SYN_RETRO_BLUE_COMPAT": PlateLayout(
        layout_id="SYN_RETRO_BLUE_COMPAT",
        state_name="CALIFORNIA",
        format_pattern="DLLLDDD",
        separator_char="",
        motto_text="",
        style=PlateStyle(
            bg_color_top=(25, 45, 95),
            bg_color_bottom=(20, 35, 80),
            text_color=(240, 195, 35),
            header_color=(240, 195, 35),
            border_color=(240, 195, 35),
            bevel_color=(50, 75, 130),
            has_gradient=False,
            sticker_colors=[(230, 200, 40), (220, 50, 50)],
        ),
    ),
}


def _get_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    """Safely retrieves a clean TrueType font or falls back gracefully to default PIL font."""
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue

    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def generate_valid_plate_text(
    pattern: str, rng: np.random.RandomState, separator: str = ""
) -> Tuple[str, str]:
    """Generates authentic-looking alphanumeric plate text adhering to a pattern.

    Args:
        pattern: String containing 'D' (digits 0-9) and 'L' (letters A-Z).
        rng: Deterministic numpy RandomState instance.
        separator: Infix separator character (e.g. '-', ' ').

    Returns:
        Tuple of (display_text_with_separator, normalized_clean_text).
    """
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # Omitting easily confused I/O by convention on real plates
    digits = string.digits

    char_list = []
    for char_type in pattern:
        if char_type == "D":
            char_list.append(rng.choice(list(digits)))
        elif char_type == "L":
            char_list.append(rng.choice(list(letters)))
        else:
            char_list.append(char_type)

    clean_text = "".join(char_list)

    if separator and len(char_list) >= 6:
        midpoint = 3 if len(char_list) in (6, 7) else len(char_list) // 2
        display_text = "".join(char_list[:midpoint]) + separator + "".join(char_list[midpoint:])
    else:
        display_text = clean_text

    return display_text, clean_text


def render_synthetic_plate_image(
    layout: PlateLayout,
    display_text: str,
    rng: np.random.RandomState,
    plate_width: int = 400,
    plate_height: int = 200,
) -> Image.Image:
    """Renders a standalone, realistic synthetic license plate image.

    Standard US aspect ratio is 2:1 (12 x 6 inches).
    Renders embossed stamped-metal bevels, bolt holes, header text,
    alphanumeric center characters, stickers, and motto text.
    """
    style = layout.style
    plate_img = Image.new("RGB", (plate_width, plate_height), style.bg_color_top)
    draw = ImageDraw.Draw(plate_img)

    # 1. Gradient Background (if configured)
    if style.has_gradient:
        for y in range(plate_height):
            ratio = y / float(plate_height)
            r = int(style.bg_color_top[0] * (1.0 - ratio) + style.bg_color_bottom[0] * ratio)
            g = int(style.bg_color_top[1] * (1.0 - ratio) + style.bg_color_bottom[1] * ratio)
            b = int(style.bg_color_top[2] * (1.0 - ratio) + style.bg_color_bottom[2] * ratio)
            draw.line([(0, y), (plate_width, y)], fill=(r, g, b))

    # 2. Header Banner / Top Stripe
    if style.has_header_banner and style.banner_color:
        banner_h = int(plate_height * 0.22)
        draw.rectangle([(0, 0), (plate_width, banner_h)], fill=style.banner_color)

    # 3. Bottom Stripe (e.g. PA style)
    if style.has_bottom_stripe and style.bottom_stripe_color:
        stripe_h = int(plate_height * 0.12)
        draw.rectangle([(0, plate_height - stripe_h), (plate_width, plate_height)], fill=style.bottom_stripe_color)

    # 4. Outer Embossed Rim (Stamped metal effect)
    margin = 7
    radius = 16
    draw.rounded_rectangle(
        [margin, margin, plate_width - margin, plate_height - margin],
        radius=radius,
        outline=style.border_color,
        width=4,
    )
    draw.rounded_rectangle(
        [margin + 4, margin + 4, plate_width - margin - 4, plate_height - margin - 4],
        radius=radius - 4,
        outline=style.bevel_color,
        width=2,
    )

    # 5. Bolt Mounting Holes (Standard 4 corners)
    bolt_radius = 6
    bolt_offsets = [
        (margin + 26, margin + 22),
        (plate_width - margin - 26, margin + 22),
        (margin + 26, plate_height - margin - 22),
        (plate_width - margin - 26, plate_height - margin - 22),
    ]
    for bx, by in bolt_offsets:
        draw.ellipse([bx - bolt_radius, by - bolt_radius, bx + bolt_radius, by + bolt_radius], fill=(80, 80, 80))
        draw.ellipse([bx - bolt_radius + 2, by - bolt_radius + 2, bx + bolt_radius - 2, by + bolt_radius - 2], fill=(45, 45, 45))

    # 6. Registration Stickers / Decals (Top corners)
    if len(style.sticker_colors) >= 2 and rng.rand() > 0.15:
        # Left sticker
        draw.rectangle([(margin + 52, margin + 12), (margin + 84, margin + 34)], fill=style.sticker_colors[0], outline=(40, 40, 40))
        # Right sticker
        draw.rectangle([(plate_width - margin - 84, margin + 12), (plate_width - margin - 52, margin + 34)], fill=style.sticker_colors[1], outline=(40, 40, 40))

    # 7. State Header Text
    font_header = _get_font(size=int(plate_height * 0.15), bold=True)
    header_text = layout.state_name
    header_bbox = draw.textbbox((0, 0), header_text, font=font_header)
    hw = header_bbox[2] - header_bbox[0]
    hh = header_bbox[3] - header_bbox[1]
    hx = (plate_width - hw) // 2
    hy = int(plate_height * 0.08)
    draw.text((hx, hy), header_text, fill=style.header_color, font=font_header)

    # 8. Main Alphanumeric Plate Text (Embossed stamp effect)
    font_main = _get_font(size=int(plate_height * 0.42), bold=True)
    text_bbox = draw.textbbox((0, 0), display_text, font=font_main)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    tx = (plate_width - tw) // 2
    ty = int(plate_height * 0.32)

    # Embossed shadow (1px offset)
    shadow_color = (max(0, style.text_color[0] - 25), max(0, style.text_color[1] - 25), max(0, style.text_color[2] - 25))
    highlight_color = (min(255, style.text_color[0] + 50), min(255, style.text_color[1] + 50), min(255, style.text_color[2] + 50))
    draw.text((tx + 2, ty + 2), display_text, fill=shadow_color, font=font_main)
    draw.text((tx - 1, ty - 1), display_text, fill=highlight_color, font=font_main)
    draw.text((tx, ty), display_text, fill=style.text_color, font=font_main)

    # 9. Slogan / Motto Text at Bottom
    if layout.motto_text:
        font_motto = _get_font(size=int(plate_height * 0.085), bold=True)
        motto_bbox = draw.textbbox((0, 0), layout.motto_text, font=font_motto)
        mw = motto_bbox[2] - motto_bbox[0]
        mx = (plate_width - mw) // 2
        my = int(plate_height * 0.81)
        motto_color = style.header_color if not style.has_bottom_stripe else (15, 30, 80)
        draw.text((mx, my), layout.motto_text, fill=motto_color, font=font_motto)

    return plate_img
