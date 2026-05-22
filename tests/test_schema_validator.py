"""Tests for schema_validator — validate_against_schema."""

import pytest
from sigma.schema_validator import validate_against_schema


class TestTypeChecking:
    def test_object_type_valid(self):
        assert validate_against_schema({"a": 1}, {"type": "object"}) == []

    def test_object_type_invalid(self):
        errors = validate_against_schema([1, 2], {"type": "object"})
        assert any("object" in e for e in errors)

    def test_array_type_valid(self):
        assert validate_against_schema([1, 2], {"type": "array"}) == []

    def test_string_type_valid(self):
        assert validate_against_schema("hello", {"type": "string"}) == []

    def test_number_type(self):
        assert validate_against_schema(3.14, {"type": "number"}) == []
        assert validate_against_schema(42, {"type": "number"}) == []

    def test_integer_type(self):
        assert validate_against_schema(42, {"type": "integer"}) == []
        errors = validate_against_schema(3.14, {"type": "integer"})
        assert len(errors) > 0

    def test_boolean_type(self):
        assert validate_against_schema(True, {"type": "boolean"}) == []
        assert validate_against_schema(False, {"type": "boolean"}) == []

    def test_null_type(self):
        assert validate_against_schema(None, {"type": "null"}) == []


class TestRequired:
    def test_all_required_present(self):
        schema = {"type": "object", "required": ["x", "y"]}
        assert validate_against_schema({"x": 1, "y": 2}, schema) == []

    def test_missing_required(self):
        schema = {"type": "object", "required": ["x", "y"]}
        errors = validate_against_schema({"x": 1}, schema)
        assert any("y" in e for e in errors)

    def test_no_required_key_in_schema(self):
        assert validate_against_schema({"x": 1}, {"type": "object"}) == []


class TestProperties:
    def test_nested_property_valid(self):
        schema = {
            "type": "object",
            "properties": {"value": {"type": "number"}},
        }
        assert validate_against_schema({"value": 42}, schema) == []

    def test_nested_property_wrong_type(self):
        schema = {
            "type": "object",
            "properties": {"value": {"type": "number"}},
        }
        errors = validate_against_schema({"value": "abc"}, schema)
        assert len(errors) > 0

    def test_extra_properties_ignored(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert validate_against_schema({"a": "x", "b": 99}, schema) == []

    def test_deeply_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "properties": {"mass": {"type": "number"}},
                    "required": ["mass"],
                }
            },
        }
        assert validate_against_schema({"params": {"mass": 5.0}}, schema) == []
        errors = validate_against_schema({"params": {"x": 1}}, schema)
        assert len(errors) > 0


class TestItems:
    def test_items_valid(self):
        schema = {"type": "array", "items": {"type": "number"}}
        assert validate_against_schema([1, 2.5, 3], schema) == []

    def test_items_invalid_element(self):
        schema = {"type": "array", "items": {"type": "number"}}
        errors = validate_against_schema([1, "bad", 3], schema)
        assert len(errors) > 0

    def test_min_items(self):
        schema = {"type": "array", "minItems": 2}
        assert validate_against_schema([1, 2], schema) == []
        errors = validate_against_schema([1], schema)
        assert len(errors) > 0

    def test_max_items(self):
        schema = {"type": "array", "maxItems": 2}
        assert validate_against_schema([1, 2], schema) == []
        errors = validate_against_schema([1, 2, 3], schema)
        assert len(errors) > 0


class TestEmptySchema:
    def test_empty_schema_always_valid(self):
        assert validate_against_schema(42, {}) == []
        assert validate_against_schema("hi", {}) == []
        assert validate_against_schema({"a": [1, 2, 3]}, {}) == []


class TestRealWorldSchemas:
    def test_estimate_schema(self):
        """Schema used for consensus estimation output."""
        schema = {
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": {"type": "number"},
                "min": {"type": "number"},
                "max": {"type": "number"},
                "confidence": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
        }
        assert validate_against_schema(
            {"value": 150.0, "min": 140, "max": 160, "confidence": 4, "reasoning": "test"},
            schema,
        ) == []
        errors = validate_against_schema(
            {"value": "not a number"}, schema,
        )
        assert len(errors) > 0

    def test_tool_result_schema(self):
        schema = {
            "type": "object",
            "required": ["success", "data"],
            "properties": {
                "success": {"type": "boolean"},
                "data": {"type": "object"},
            },
        }
        assert validate_against_schema(
            {"success": True, "data": {"mass_kg": 5.0}}, schema,
        ) == []
