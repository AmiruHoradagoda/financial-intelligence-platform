"""Grounded question generation with an injected text completion dependency."""

from pydantic import ValidationError

from financial_intelligence.config import load_config
from financial_intelligence.data_factory.prompts import build_question_prompt
from financial_intelligence.data_factory.validation import (
    InsufficientInformation,
    InvalidLLMResponse,
    parse_llm_json,
    validate_generated_questions,
)
from financial_intelligence.infrastructure.llm_client import LiteLLMClient, LLMClient
from financial_intelligence.schemas.qa import QuestionBatch

__all__ = [
    "InsufficientInformation",
    "QuestionGenerator",
    "generate_questions",
    "parse_llm_json",
    "validate_generated_questions",
    "InvalidLLMResponse",
]


class QuestionGenerator:
    def __init__(self, llm_client: LLMClient, model: str) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "Question generation model must be configured in "
                "config/data.yaml under data_generation.model"
            )
        self.llm_client = llm_client
        self.model = model

    def generate(self, chunk: dict) -> list[dict]:
        if not chunk["text"].strip():
            raise InsufficientInformation(
                "The excerpt contains no nonblank source text"
            )
        content = self.llm_client.complete(
            model=self.model, prompt=build_question_prompt(chunk["text"])
        )
        data = parse_llm_json(content)
        if "unprocessable_reason" in data:
            reason = data["unprocessable_reason"]
            if (
                set(data) != {"unprocessable_reason"}
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise InvalidLLMResponse("Invalid unprocessable excerpt response")
            raise InsufficientInformation(reason.strip())
        try:
            batch = QuestionBatch.model_validate(data)
        except ValidationError as error:
            raise InvalidLLMResponse(f"Invalid question batch: {error}") from error
        return [item.model_dump() for item in batch.questions]


def generate_questions(chunk: dict) -> list[dict]:
    """Generate questions using the model selected in config/data.yaml."""
    config = load_config("data")
    return QuestionGenerator(
        LiteLLMClient(), config.get("data_generation", {}).get("model")
    ).generate(chunk)
