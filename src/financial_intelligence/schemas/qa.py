"""LLM question batches and the existing question JSONL row contract."""

from collections import Counter
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
