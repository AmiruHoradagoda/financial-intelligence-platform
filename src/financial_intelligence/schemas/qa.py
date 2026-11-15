"""LLM question batches and the existing question JSONL row contract."""

from collections import Counter
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

QuestionCategory = Literal["hard_fact", "strategic_summary", "stylistic"]
CATEGORY_COUNTS = {"hard_fact": 4, "strategic_summary": 3, "stylistic": 3}


class GeneratedQuestion(BaseModel):
    model_config = ConfigDict(strict=True)

    category: QuestionCategory
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


class QuestionBatch(BaseModel):
    model_config = ConfigDict(strict=True)

    questions: list[GeneratedQuestion] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        counts = Counter(item.category for item in self.questions)
        if counts != CATEGORY_COUNTS:
            raise ValueError("Question categories must follow the 4/3/3 distribution")
        normalized = [
            " ".join(item.question.casefold().split()) for item in self.questions
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate questions are not allowed")
        return self


class QuestionRecord(GeneratedQuestion):
    """A generated question with its original source metadata."""

    chunk_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_number: int = Field(ge=1)


class AnswerResponse(BaseModel):
    """Structured answer; unanswerable responses carry no answer or evidence."""

    model_config = ConfigDict(strict=True, extra="forbid")

    answer: str | None
    is_answerable: bool
    evidence_quote: str | None
    evidence_quotes: list[str] | None = Field(default=None, min_length=1, max_length=8)

    @field_validator("evidence_quotes")
    @classmethod
    def validate_quotes(cls, value):
        if value is not None and any(not quote.strip() for quote in value):
            raise ValueError("Evidence spans must be nonblank")
        return value

    @model_serializer(mode="wrap")
    def preserve_legacy_shape(self, handler):
        result = handler(self)
        if result.get("evidence_quotes") is None:
            result.pop("evidence_quotes", None)
        return result

    @model_validator(mode="after")
    def validate_answerability(self) -> Self:
        if self.is_answerable:
            if not self.answer or not self.answer.strip():
                raise ValueError("Answerable responses require a nonblank answer")
            if (
                not (self.evidence_quote and self.evidence_quote.strip())
                and not self.evidence_quotes
            ):
                raise ValueError("Answerable responses require nonblank evidence")
        elif (
            self.answer is not None
            or self.evidence_quote is not None
            or self.evidence_quotes is not None
        ):
            raise ValueError("Unanswerable responses require null answer and evidence")
        return self


class QAPairRecord(QuestionRecord, AnswerResponse):
    """The original question and source metadata together with its answer."""
