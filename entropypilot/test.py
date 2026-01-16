import matplotlib.pyplot as plt

from entropypilot.utils import draw_palette_on_axis, get_colors_from_llm_sync

# 1. SETUP
# Using a slightly older/smaller model (like gpt-3.5 or gpt-4-turbo)
# often highlights these architectural issues better than the newest flagship.
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.9


def get_colors_from_llm(prompt):
    """Wrapper around sync LLM interface with logging."""
    print(f"Asking LLM: '{prompt}'...")
    return get_colors_from_llm_sync(prompt, model=MODEL, temperature=TEMPERATURE)


# =========================================
# MAIN EXECUTION
# =========================================
if __name__ == "__main__":
    # Define the two opposing prompts

    # TEST A: High Entropy (Negative Constraint)
    # The model has to consider every color in existence and try to filter out red.
    neg_prompt = "Generate a palette of 6 distinct hex codes. CONSTRAINT: The palette must NOT contain any shade of red or orange."

    # TEST B: Low Entropy (Affirmative Constraint)
    # The model only has to look in one specific corner of color space.
    aff_prompt = "Generate a palette of 6 distinct hex codes. CONSTRAINT: The palette must ONLY contain shades of cool blues, aquas, and teals."

    # Fetch data
    print("--- Fetching palettes from LLM (this might take a few seconds) ---")
    neg_colors = get_colors_from_llm(neg_prompt)
    aff_colors = get_colors_from_llm(aff_prompt)
    print("--- Data fetched. Rendering GUI. ---")

    # Setup the GUI Window (Figure and Axes)
    # nrows=2, ncols=1 means stacked vertically
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7))
    fig.canvas.manager.set_window_title("Architecture vs. Prompting Demo") # type: ignore

    # 1. Draw Negative Results on Top Axis (ax1)
    draw_palette_on_axis(
        ax=ax1,
        colors=neg_colors,
        title="TEST A: Negative Constraint (High Entropy)",
        subtitle=f'Prompt: "{neg_prompt}"\nResult: Often muddy or destructed. The model spends its energy trying to *exclude* data.',
    )

    # 2. Draw Affirmative Results on Bottom Axis (ax2)
    draw_palette_on_axis(
        ax=ax2,
        colors=aff_colors,
        title="TEST B: Affirmative Constraint (Low Entropy)",
        subtitle=f'Prompt: "{aff_prompt}"\nResult: Vibrant and cohesive. The model\'s search space was architecturally narrowed.',
    )

    # Adjust layout to prevent overlaps and show the window
    plt.subplots_adjust(hspace=0.6, top=0.9, bottom=0.1)
    print("Displaying results...")
    plt.show()
