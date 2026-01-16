"""Color conversion and validation utilities for EntropyPilot."""

import colorsys


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """
    Convert hex color to RGB tuple (0-1 range) for matplotlib.

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        Tuple of (r, g, b) values in 0-1 range

    Example:
        >>> hex_to_rgb("#FF0000")
        (1.0, 0.0, 0.0)
    """
    hex_color = hex_color.lstrip("#")
    try:
        r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b)
    except (ValueError, IndexError):
        return (0.5, 0.5, 0.5)  # Gray fallback


def hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """
    Convert hex color to HSL (Hue, Saturation, Lightness).

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        Tuple of (h, s, l) where:
        - h: Hue in degrees (0-360)
        - s: Saturation (0-1)
        - l: Lightness (0-1)

    Example:
        >>> hex_to_hsl("#FF0000")
        (0.0, 1.0, 0.5)
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360, s, l)  # Hue in degrees, S and L as 0-1


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """
    Convert RGB values (0-1 range) to hex color string.

    Args:
        r: Red component (0-1)
        g: Green component (0-1)
        b: Blue component (0-1)

    Returns:
        Hex color string with # prefix

    Example:
        >>> rgb_to_hex(1.0, 0.0, 0.0)
        '#ff0000'
    """
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def is_red_or_orange(hex_color: str) -> bool:
    """
    Detect if a color is a shade of red or orange.

    Uses HSL color space to check hue ranges:
    - Red: Hue roughly 0-30 or 330-360
    - Orange: Hue roughly 15-45
    - Combined range: 10-40 or 330-360

    Args:
        hex_color: Hex color string to validate

    Returns:
        True if color is red or orange, False otherwise

    Example:
        >>> is_red_or_orange("#FF0000")
        True
        >>> is_red_or_orange("#0000FF")
        False
    """
    try:
        h, s, l = hex_to_hsl(hex_color)
        # Need some saturation to be considered a color (not gray)
        if s < 0.15:
            return False
        # Red/orange hue ranges
        return (10 <= h <= 40) or (330 <= h <= 360)
    except (ValueError, IndexError):
        return False


def is_cool_blue_aqua_teal(hex_color: str) -> bool:
    """
    Detect if a color is a shade of cool blue, aqua, or teal.

    Uses HSL color space to check hue ranges:
    - Cool blues: Hue roughly 180-260
    - Aquas/Teals: Hue roughly 160-200
    - Combined range: 160-260

    Args:
        hex_color: Hex color string to validate

    Returns:
        True if color is cool blue/aqua/teal, False otherwise

    Example:
        >>> is_cool_blue_aqua_teal("#00FFFF")
        True
        >>> is_cool_blue_aqua_teal("#FF0000")
        False
    """
    try:
        h, s, l = hex_to_hsl(hex_color)
        # Very low saturation = gray (acceptable as neutral)
        # Very high/low lightness = white/black (acceptable as neutral)
        if s < 0.1 and (l < 0.15 or l > 0.85):
            return True  # Allow near-black, near-white, and grays
        # Must have some saturation to be a "color"
        if s < 0.1:
            return True  # Grays are acceptable
        # Check if in the cool blue/aqua/teal hue range
        return 160 <= h <= 260
    except (ValueError, IndexError):
        return False
