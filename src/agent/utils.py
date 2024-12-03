from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.utils.json_schema import dereference_refs


def _rm_titles(kv: dict) -> dict:
    """Remove titles from a dictionary."""
    new_kv = {}
    for k, v in kv.items():
        if k == "title":
            continue
        elif isinstance(v, dict):
            new_kv[k] = _rm_titles(v)
        else:
            new_kv[k] = v
    return new_kv


# PUBLIC API


def update_json_schema(
    schema: dict,
    *,
    multi: bool = True,
) -> dict:
    """Add missing fields to JSON schema and add support for multiple records.

    This makes it such that a schema always gets interpreted as a multi record schema.

    This allows a chat model to extract multiple records from a single document OR
    extract no records if nothing in the document matches the schema.
    """
    if multi:
        # Wrap the schema in an object called "Root" with a property called: "data"
        # which will be a json array of the original schema.
        schema_ = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": dereference_refs(schema),
                },
            },
            "required": ["data"],
        }
    else:
        raise NotImplementedError("Only multi is supported for now.")

    schema_["title"] = "extractor"
    schema_["description"] = "Extract information matching the given schema."
    return schema_


# The rate limiter is used to limit the number of requests made to the API
# to avoid hitting the rate limit.
# The rate limiter will wait until it is allowed to make a request
# based on the rate limit configuration.
RATE_LIMITER = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.

    Returns:
        BaseChatModel: The chat model.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)

    return init_chat_model(
        model,
        model_provider=provider,
        temperature=0,
        rate_limiter=RATE_LIMITER,
    )
