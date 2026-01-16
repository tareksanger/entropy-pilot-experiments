"""Pydantic models for EntropyPilot."""

from pydantic import BaseModel, Field, field_validator


class Colors(BaseModel):
    """Color palette response model for LLM structured outputs."""

    palette: list[str] = Field(
        ..., description="The list of colors in the palette using hex codes."
    )
    
    @field_validator("palette")
    @classmethod
    def validate_palette(cls, v):
        if isinstance(v, list):
            return [
                f"#{color.strip()}" if not color.strip().startswith("#") else color.strip()
                for color in v
            ]