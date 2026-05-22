"""Lightweight JSON schema validation for LLM structured output.

No external dependencies — pure recursive dict walking.
"""


def validate_against_schema(data: object, schema: dict) -> list[str]:
    """Validate `data` against a JSON-Schema-like dict.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    if "type" in schema:
        expected = schema["type"]
        if not _check_type(data, expected):
            errors.append(f"expected type '{expected}', got '{_type_name(data)}'")
            return errors  # wrong type makes further checks meaningless

    if isinstance(data, dict):
        errors.extend(_validate_object(data, schema))
    elif isinstance(data, list):
        errors.extend(_validate_array(data, schema))

    return errors


def _validate_object(data: dict, schema: dict) -> list[str]:
    errors: list[str] = []

    if "required" in schema:
        for key in schema["required"]:
            if key not in data:
                errors.append(f"missing required key: '{key}'")

    if "properties" in schema:
        for key, prop_schema in schema["properties"].items():
            if key in data:
                sub = validate_against_schema(data[key], prop_schema)
                for e in sub:
                    errors.append(f"'{key}': {e}")

    return errors


def _validate_array(data: list, schema: dict) -> list[str]:
    errors: list[str] = []

    if "minItems" in schema and len(data) < schema["minItems"]:
        errors.append(f"expected at least {schema['minItems']} items, got {len(data)}")

    if "maxItems" in schema and len(data) > schema["maxItems"]:
        errors.append(f"expected at most {schema['maxItems']} items, got {len(data)}")

    if "items" in schema:
        for i, item in enumerate(data):
            sub = validate_against_schema(item, schema["items"])
            for e in sub:
                errors.append(f"[{i}]: {e}")

    return errors


def _check_type(data: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(data, dict)
    if expected == "array":
        return isinstance(data, list)
    if expected == "string":
        return isinstance(data, str)
    if expected == "number":
        return isinstance(data, (int, float)) and not isinstance(data, bool)
    if expected == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if expected == "boolean":
        return isinstance(data, bool)
    if expected == "null":
        return data is None
    return True  # unknown type — pass


def _type_name(data: object) -> str:
    if data is None:
        return "null"
    if isinstance(data, bool):
        return "boolean"
    if isinstance(data, int):
        return "integer"
    if isinstance(data, float):
        return "number"
    if isinstance(data, str):
        return "string"
    if isinstance(data, list):
        return "array"
    if isinstance(data, dict):
        return "object"
    return type(data).__name__
