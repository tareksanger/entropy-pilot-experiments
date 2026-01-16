"""LLM interface utilities for EntropyPilot."""

from openai import AsyncOpenAI, OpenAI

from entropypilot.config import config
from entropypilot.utils.models import Colors

# Initialize OpenAI clients
async_client = AsyncOpenAI(api_key=config.openai_api_key)
sync_client = OpenAI(api_key=config.openai_api_key)


async def get_colors_from_llm_async(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    seed: int | None = None,
) -> Colors:
    """
    Async LLM call for color generation using structured outputs.

    Args:
        prompt: The prompt to send to the LLM
        model: OpenAI model to use (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0 for deterministic)
        seed: Optional seed for deterministic sampling. Use unique seeds to bypass caching.

    Returns:
        Colors model with palette of hex codes

    Example:
        >>> colors = await get_colors_from_llm_async("Generate 6 blue colors")
        >>> print(colors.palette)
        ['#0000FF', ...]
    """
    try:
        kwargs = {
            "model": model,
            "temperature": temperature,
            "response_format": Colors,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a color palette generator. Output only raw JSON lists of 6 hex codes under the key 'palette'.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = await async_client.chat.completions.parse(**kwargs)
        return response.choices[0].message.parsed or Colors(palette=["#cccccc"] * 6)
    except Exception as e:
        print(f"Error: {e}")
        return Colors(palette=["#cccccc"] * 6)


def get_colors_from_llm_sync(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    seed: int | None = None,
):
    """
    Sync LLM call for color generation using structured outputs.

    Args:
        prompt: The prompt to send to the LLM
        model: OpenAI model to use (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0.9 for variety)
        seed: Optional seed for deterministic sampling. Use unique seeds to bypass caching.

    Returns:
        Colors model with palette of hex codes

    Example:
        >>> colors = get_colors_from_llm_sync("Generate 6 blue colors")
        >>> print(colors.palette)
        ['#0000FF', ...]
    """
    try:
        kwargs = {
            "model": model,
            "temperature": temperature,
            "response_format": Colors,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a color palette generator. Output only raw JSON lists of 6 hex codes under the key 'palette'.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = sync_client.chat.completions.parse(**kwargs)
        return response.choices[0].message.parsed or Colors(palette=["#cccccc"] * 6)
    except Exception as e:
        print(f"Error generating colors: {e}")
        # Return a fallback gray palette if LLM fails entirely
        return Colors(palette=["#cccccc"] * 6)


# Convenience alias - default to async version
get_colors_from_llm = get_colors_from_llm_async
