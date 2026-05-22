"""Tests for Pydantic output parsing integration."""

import pytest
from pydantic import BaseModel

from sigma.pydantic_validator import (
    PydanticOutputParser,
    PydanticValidationError,
    validate_output,
)


# ── Test Models ────────────────────────────────────────────────────

class RocketSpec(BaseModel):
    mass_kg: float
    thrust_n: float
    material: str


class AgentAnalysis(BaseModel):
    confidence: str
    value: float
    reasoning: str = ""


class MinimalModel(BaseModel):
    name: str


class NestedModel(BaseModel):
    title: str
    spec: RocketSpec


# ── PydanticOutputParser ───────────────────────────────────────────

class TestPydanticOutputParser:

    def test_parse_valid_json(self):
        parser = PydanticOutputParser(RocketSpec)
        result = parser.parse('{"mass_kg": 5.0, "thrust_n": 500, "material": "aluminum"}')
        assert isinstance(result, RocketSpec)
        assert result.mass_kg == 5.0
        assert result.thrust_n == 500
        assert result.material == "aluminum"

    def test_parse_json_with_markdown_fence(self):
        parser = PydanticOutputParser(RocketSpec)
        text = '```json\n{"mass_kg": 3.0, "thrust_n": 300, "material": "steel"}\n```'
        result = parser.parse(text)
        assert result.mass_kg == 3.0
        assert result.material == "steel"

    def test_parse_json_with_extra_text(self):
        parser = PydanticOutputParser(RocketSpec)
        text = 'Here is the result: {"mass_kg": 10.0, "thrust_n": 1000, "material": "carbon"} end'
        result = parser.parse(text)
        assert result.mass_kg == 10.0

    def test_parse_invalid_json_raises(self):
        parser = PydanticOutputParser(RocketSpec)
        with pytest.raises(PydanticValidationError):
            parser.parse("not json at all")

    def test_parse_empty_string_raises(self):
        parser = PydanticOutputParser(RocketSpec)
        with pytest.raises(PydanticValidationError):
            parser.parse("")

    def test_parse_none_text_raises(self):
        parser = PydanticOutputParser(RocketSpec)
        with pytest.raises(PydanticValidationError):
            parser.parse(None)

    def test_parse_missing_required_fields_raises(self):
        parser = PydanticOutputParser(RocketSpec)
        with pytest.raises(PydanticValidationError) as exc:
            parser.parse('{"mass_kg": 5.0}')
        assert "validation error" in str(exc.value).lower() or "验证" in str(exc.value)

    def test_parse_minimal_model(self):
        parser = PydanticOutputParser(MinimalModel)
        result = parser.parse('{"name": "test"}')
        assert result.name == "test"

    def test_parse_nested_model(self):
        parser = PydanticOutputParser(NestedModel)
        result = parser.parse(
            '{"title": "design", "spec": {"mass_kg": 1.0, "thrust_n": 100, "material": "alu"}}'
        )
        assert result.title == "design"
        assert result.spec.mass_kg == 1.0

    def test_parse_with_default_value(self):
        parser = PydanticOutputParser(AgentAnalysis)
        result = parser.parse('{"confidence": "HIGH", "value": 42.0}')
        assert result.confidence == "HIGH"
        assert result.value == 42.0
        assert result.reasoning == ""  # default


# ── parse_with_retry ───────────────────────────────────────────────

class TestParseWithRetry:

    def test_first_attempt_succeeds_no_retry(self):
        parser = PydanticOutputParser(RocketSpec)
        call_count = [0]

        def mock_llm(system, user):
            call_count[0] += 1
            return '{"mass_kg": 1.0, "thrust_n": 100, "material": "test"}'

        result = parser.parse_with_retry(
            '{"mass_kg": 1.0, "thrust_n": 100, "material": "test"}',
            mock_llm, system="sys", user_prefix="prefix",
        )
        assert result.mass_kg == 1.0
        assert call_count[0] == 0  # No retry needed

    def test_retry_on_validation_failure(self):
        parser = PydanticOutputParser(RocketSpec, max_retries=1)
        calls = []

        def mock_llm(system, user):
            calls.append(user)
            return "still not valid json"  # always returns bad data

        # First text is bad, retry also fails
        with pytest.raises(PydanticValidationError):
            parser.parse_with_retry(
                "bad text", mock_llm, system="sys", user_prefix="fix",
            )
        assert len(calls) >= 1  # At least one retry attempt was made

    def test_exhausted_retries(self):
        parser = PydanticOutputParser(RocketSpec, max_retries=1)

        def mock_llm(system, user):
            return '{"mass_kg": "not_a_number"}'

        with pytest.raises(PydanticValidationError):
            parser.parse_with_retry(
                '{"mass_kg": "bad"}', mock_llm, system="sys", user_prefix="fix",
            )

    def test_zero_max_retries(self):
        parser = PydanticOutputParser(RocketSpec, max_retries=0)
        with pytest.raises(PydanticValidationError):
            parser.parse_with_retry(
                "invalid", lambda s, u: '{"mass_kg": 1, "thrust_n": 2, "material": "m"}',
                system="s", user_prefix="p",
            )


# ── validate_output convenience ────────────────────────────────────

class TestValidateOutput:

    def test_validate_dict(self):
        result = validate_output(RocketSpec, {"mass_kg": 2.0, "thrust_n": 200, "material": "alu"})
        assert isinstance(result, RocketSpec)
        assert result.mass_kg == 2.0

    def test_validate_json_string(self):
        result = validate_output(RocketSpec, '{"mass_kg": 3.0, "thrust_n": 300, "material": "fe"}')
        assert result.mass_kg == 3.0

    def test_validate_invalid_dict_raises(self):
        with pytest.raises(PydanticValidationError):
            validate_output(RocketSpec, {"mass_kg": 1.0})


# ── PydanticValidationError ────────────────────────────────────────

class TestPydanticValidationError:

    def test_error_stores_raw_text(self):
        err = PydanticValidationError("msg", raw_text="raw", errors=["e1"])
        assert err.raw_text == "raw"
        assert err.errors == ["e1"]

    def test_error_default_empty(self):
        err = PydanticValidationError("msg")
        assert err.raw_text == ""
        assert err.errors == []
