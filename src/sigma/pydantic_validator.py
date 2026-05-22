"""Pydantic output parsing — structured output validation for LLM responses.

Integrates Pydantic models into the Sigma/Tau pipeline:
  - Agents/tasks declare an output_model (Pydantic BaseModel subclass)
  - LLM text response is parsed as JSON and validated against the model
  - On validation failure, the LLM is retried with the error feedback

Usage:
    from pydantic import BaseModel
    from sigma.pydantic_validator import PydanticOutputParser

    class RocketSpec(BaseModel):
        mass_kg: float
        thrust_n: float
        material: str

    parser = PydanticOutputParser(RocketSpec)
    spec = parser.parse(llm_response)  # RocketSpec instance
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from sigma.log import get_logger

_log = get_logger("sigma.pydantic")

T = TypeVar("T")


class PydanticValidationError(Exception):
    """Raised when LLM output fails Pydantic validation after all retries."""

    def __init__(self, message: str, raw_text: str = "", errors: list[str] | None = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.errors = errors or []


class PydanticOutputParser:
    """Parse LLM text output into a validated Pydantic model instance.

    Handles common LLM output quirks: markdown fences, trailing commas,
    and partial JSON. On validation failure, can retry the LLM with
    the validation errors as feedback.
    """

    def __init__(self, model: type, max_retries: int = 2):
        """
        Args:
            model: A Pydantic v2 BaseModel subclass.
            max_retries: Max LLM retries on validation failure (default 2).
        """
        self.model = model
        self.max_retries = max_retries

    def parse(self, text: str | None) -> Any:
        """Parse a single LLM response into the model.

        Returns a validated model instance.
        Raises PydanticValidationError if parsing fails.
        """
        if not text:
            raise PydanticValidationError("输入为空，无法解析")
        data = self._extract_json(text)
        if data is None:
            raise PydanticValidationError(
                f"无法从响应中提取 JSON: {text[:200]}...",
                raw_text=text,
            )
        try:
            return self.model.model_validate(data)
        except Exception as e:
            raise PydanticValidationError(
                f"Pydantic 验证失败: {e}",
                raw_text=text,
                errors=[str(e)],
            )

    def parse_with_retry(
        self, text: str, llm_call, system: str = "", user_prefix: str = ""
    ) -> Any:
        """Parse LLM output, retrying with error feedback on failure.

        Args:
            text: Initial LLM response text.
            llm_call: callable(system_prompt, user_prompt) -> str
            system: System prompt for retry calls.
            user_prefix: Prepend to retry user prompt.

        Returns:
            Validated model instance (raises PydanticValidationError if all retries fail).
        """
        last_text = text
        for attempt in range(self.max_retries + 1):
            try:
                return self.parse(last_text)
            except PydanticValidationError as e:
                if attempt >= self.max_retries:
                    raise
                schema_hint = self._schema_description()
                retry_user = (
                    f"{user_prefix}\n\n"
                    f"前次输出格式有误：{e}\n\n"
                    f"期望的 JSON 结构（Pydantic 模型 {self.model.__name__}）：\n{schema_hint}\n\n"
                    f"请修正后重新输出，只输出符合格式的 JSON，不要其他文字。"
                )
                try:
                    last_text = llm_call(system, retry_user)
                except Exception as llm_err:
                    raise PydanticValidationError(
                        f"重试 LLM 调用失败: {llm_err}",
                        raw_text=last_text,
                        errors=e.errors,
                    )

    async def parse_async_with_retry(
        self, text: str, async_llm_call,
        system: str = "", user_prefix: str = ""
    ) -> Any:
        """Async variant of parse_with_retry."""
        import asyncio

        last_text = text
        for attempt in range(self.max_retries + 1):
            try:
                return self.parse(last_text)
            except PydanticValidationError as e:
                if attempt >= self.max_retries:
                    raise
                schema_hint = self._schema_description()
                retry_user = (
                    f"{user_prefix}\n\n"
                    f"前次输出格式有误：{e}\n\n"
                    f"期望的 JSON 结构（Pydantic 模型 {self.model.__name__}）：\n{schema_hint}\n\n"
                    f"请修正后重新输出，只输出符合格式的 JSON，不要其他文字。"
                )
                try:
                    last_text = await async_llm_call(system, retry_user)
                except Exception as llm_err:
                    raise PydanticValidationError(
                        f"重试 LLM 调用失败: {llm_err}",
                        raw_text=last_text,
                        errors=e.errors,
                    )

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON dict from LLM text output.

        Handles markdown fences, stray characters, and common LLM quirks.
        """
        if not text:
            return None
        # Try direct parse first
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # Strip markdown fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) > 2 and lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1])
            elif len(lines) > 1:
                cleaned = "\n".join(lines[1:])
        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        return None

    def _schema_description(self) -> str:
        """Generate a human-readable description of the expected schema."""
        fields = []
        for name, field_info in self.model.model_fields.items():
            annotation = field_info.annotation
            type_name = getattr(annotation, "__name__", str(annotation))
            default = field_info.default
            desc = f"  {name}: {type_name}"
            if default is not None:
                desc += f" (default={default})"
            if field_info.description:
                desc += f"  # {field_info.description}"
            fields.append(desc)
        return "\n".join(fields)


def validate_output(model: type, data: dict | str) -> Any:
    """Convenience: validate a dict or JSON string against a Pydantic model.

    Returns a validated model instance.
    Raises PydanticValidationError on failure.
    """
    if isinstance(data, str):
        parser = PydanticOutputParser(model)
        return parser.parse(data)
    try:
        return model.model_validate(data)
    except Exception as e:
        raise PydanticValidationError(str(e), raw_text=str(data), errors=[str(e)])
