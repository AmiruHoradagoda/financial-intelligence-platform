"""Validated chunk output contract; audit functions still accept malformed dicts."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Chunk(BaseModel):
    model_config = ConfigDict(strict=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    pipeline: str = Field(min_length=1)
    strategy: Literal["fixed", "sliding"]
    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("Character span must match text length")
        return self
