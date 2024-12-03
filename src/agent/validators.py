from typing import Any, Dict

from jsonschema import exceptions
from jsonschema.validators import Draft202012Validator


def validate_json_schema(schema: Dict[str, Any]) -> None:
    """Validate a JSON schema."""
    try:
        Draft202012Validator.check_schema(schema)
    except exceptions.ValidationError as e:
        # No way to throw a client side error yet
        raise e
