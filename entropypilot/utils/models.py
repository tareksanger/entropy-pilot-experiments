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
            if all(color.strip().startswith("#") for color in v):
                return v
            return ['#' + color.strip() for color in v if not color.strip().startswith("#")]
    
