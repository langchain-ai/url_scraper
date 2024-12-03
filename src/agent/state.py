"""Define the state structures for the agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

from agent.validators import validate_json_schema


class ExtractRequest(BaseModel):
    """Request body for the extract endpoint."""

    url: str = Field(
        default=None,
        description="The URL to scrape.",
        examples=["https://science.nasa.gov/exoplanet-catalog/toi-5005-b/"],
    )
    json_schema: Dict[str, Any] = Field(
        ...,
        description="JSON schema that describes what content should be extracted "
        "from the text.",
        examples=[
            {
                "title": "Exoplanet Information Extractor",
                "description": (
                    "Schema for extracting information about exoplanets from text."
                ),
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "title": "Name",
                        "description": "The name of the exoplanet.",
                    },
                    "discovery_method": {
                        "type": "string",
                        "title": "Discovery Method",
                        "description": "The method used to discover the exoplanet.",
                    },
                    "discovery_year": {
                        "type": "integer",
                        "title": "Discovery Year",
                        "description": "The year in which the exoplanet was discovered.",
                    },
                    "mass": {
                        "type": "number",
                        "title": "Mass",
                        "description": "The mass of the exoplanet in Earth masses.",
                    },
                    "radius": {
                        "type": "number",
                        "title": "Radius",
                        "description": "The radius of the exoplanet in Earth radii.",
                    },
                    "orbital_period": {
                        "type": "number",
                        "title": "Orbital Period",
                        "description": "The orbital period of the exoplanet in days.",
                    },
                },
            }
        ],
    )

    @field_validator("json_schema")
    def validate_schema(cls, v: Any) -> Dict[str, Any]:
        """Validate the schema."""
        validate_json_schema(v)
        return v


class ExtractResponse(BaseModel):
    """Response body for the extract endpoint."""

    data: List[Any] = Field(
        ...,
        description="The extracted data. A list of extracted objects matching "
        "the requested schema.",
    )


class State(TypedDict):
    """The shared state."""

    json_schema: Dict[str, Any]
    url: Optional[str]
    data: List[Any]
