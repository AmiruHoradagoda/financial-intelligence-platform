"""Parse external JSON and provide the legacy boolean validation helper."""

import json

from pydantic import ValidationError

from financial_intelligence.infrastructure.llm_client import InvalidLLMResponse
from financial_intelligence.schemas.qa import QuestionBatch

__all__ = ["InvalidLLMResponse", "parse_llm_json", "validate_generated_questions"]


def parse_llm_json(content: str) -> dict:
    """Accept a JSON object, optionally enclosed in a complete Markdown fence."""
    if not isinstance(content, str) or not content.strip():
        raise InvalidLLMResponse("LLM returned an empty response")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (
            len(lines) < 3
            or lines[0].strip().lower() not in ("```", "```json")
            or lines[-1].strip() != "```"
        ):
            raise InvalidLLMResponse("LLM returned an invalid Markdown JSON fence")
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidLLMResponse(f"LLM returned invalid JSON: {error.msg}") from error
    if not isinstance(data, dict):
        raise InvalidLLMResponse("LLM response must be a JSON object")
    return data


def validate_generated_questions(questions: list[dict]) -> bool:
    """Keep the old bool API while applying the complete batch contract."""
    try:
        QuestionBatch.model_validate({"questions": questions})
    except ValidationError:
        return False
    return True
