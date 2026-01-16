"""Visualization utilities for EntropyPilot."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def visualize_palette(colors: list[str], title: str, figsize: tuple[int, int] = (10, 2)) -> None:
    """
    Display a color palette as visual swatches in a new figure.

    Args:
        colors: List of hex color codes
        title: Title for the palette
        figsize: Figure size as (width, height) in inches

    Example:
        >>> visualize_palette(["#FF0000", "#00FF00", "#0000FF"], "RGB Colors")
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, pad=20)
    for i, color in enumerate(colors):
        rect = patches.Rectangle(
            (i, 0), 1, 1, linewidth=1, edgecolor="none", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(i + 0.5, -0.2, color, ha="center", va="center", fontsize=10)
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.show()


def draw_palette_on_axis(ax: plt.Axes, colors: list[str], title: str, subtitle: str) -> None:
    """
    Draw color swatches onto a specific matplotlib Axis.

    Useful for creating multi-panel visualizations with multiple palettes.

    Args:
        ax: Matplotlib Axes object to draw on
        colors: List of hex color codes
        title: Main title for the palette
        subtitle: Subtitle or description text

    Example:
        >>> fig, (ax1, ax2) = plt.subplots(1, 2)
        >>> draw_palette_on_axis(ax1, ["#FF0000"], "Red", "Pure red palette")
        >>> draw_palette_on_axis(ax2, ["#0000FF"], "Blue", "Pure blue palette")
    """
    # Main Title
    ax.text(0, 1.3, title, fontsize=12, fontweight="bold", transform=ax.transAxes)
    # Subtitle explaining the mechanism
    ax.text(
        0,
        1.1,
        subtitle,
        fontsize=10,
        fontstyle="italic",
        color="#555555",
        transform=ax.transAxes,
    )

    # Draw the swatches
    for i, color in enumerate(colors):
        # Draw the colored rectangle
        # Coordinates are (x, y), width, height
        rect = patches.Rectangle(
            (i, 0), 1, 1, linewidth=1, edgecolor="#e0e0e0", facecolor=color
        )
        ax.add_patch(rect)

        # Add the hex code text below
        # Use try/except in case the LLM generates invalid hex colors that break matplotlib
        try:
            ax.text(
                i + 0.5,
                -0.2,
                color,
                ha="center",
                va="center",
                fontsize=9,
                family="monospace",
            )
        except Exception:
            ax.text(
                i + 0.5,
                -0.2,
                "INVALID",
                ha="center",
                va="center",
                fontsize=8,
                color="red",
            )

    # Clean up the axis view
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")  # Hide X/Y axes and ticks
