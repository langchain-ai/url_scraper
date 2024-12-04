"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Literal, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

# Add any other models you want users to use here
PREDEFINED_MODELS = Literal["openai/gpt-4o"]


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    model: Annotated[PREDEFINED_MODELS, {"__template_metadata__": {"kind": "llm"}}] = (
        field(
            default="openai/gpt-4o",
            metadata={
                "description": "The name of the language model to use for the agent's "
                "main interactions."
                "Should be in the form: provider/model-name."
            },
        )
    )
    proxy: Optional[str] = field(
        default=None,
        metadata={
            "description": "The proxies to use when making requests to the internet. "
            "Passed to httpx.AsyncClient as the proxy parameter.",
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
