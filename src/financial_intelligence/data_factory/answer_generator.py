"""Injected answer generation with strict response and source-quote validation."""

import hashlib
import re

from pydantic import ValidationError

from financial_intelligence.data_factory.prompts import build_answer_prompt
from financial_intelligence.data_factory.validation import parse_llm_json
from financial_intelligence.infrastructure.llm_client import (
    InvalidLLMResponse,
    LLMClient,
)
from financial_intelligence.schemas.qa import AnswerResponse, QuestionRecord


def _quote_diagnostics(quote: str, text: str) -> dict:
    """Classify exact-match failures; normalization is diagnostic only."""

    def whitespace(value: str) -> str:
        return " ".join(value.split())

    def apostrophes(value: str) -> str:
        return value.translate(str.maketrans({"\u2019": "'", "\u2018": "'"}))

    if whitespace(quote) in whitespace(text):
        reason = "whitespace_mismatch"
    elif apostrophes(quote) in apostrophes(text):
        reason = "unicode_apostrophe_mismatch"
    elif whitespace(apostrophes(quote)) in whitespace(apostrophes(text)):
        reason = "whitespace_and_unicode_apostrophe_mismatch"
    elif quote.replace("\\n", "\n").replace("\\t", "\t") in text:
        reason = "literal_escape_mismatch"
    else:
        reason = "no_contiguous_verbatim_span"
    # Only a short quote preview is logged, never the source or raw response.
    preview = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", quote)
    preview = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", preview)
    preview = re.sub(
        r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*[\"']?[^\s,;\"']+",
        r"\1=[REDACTED]",
        preview,
    )
    represented = ascii(preview[:160])
    return {
        "reason": reason,
        "quote_repr": represented[:512],
        "quote_length": len(quote),
        "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        "quote_repr_truncated": len(preview) > 160 or len(represented) > 512,
    }


class EvidenceQuoteMismatch(InvalidLLMResponse):
    """An unsupported quote with bounded diagnostics safe for batch failure logs."""

    def __init__(self, quote: str, text: str) -> None:
        super().__init__("Evidence quote is not an exact substring of the source")
        self.diagnostics = _quote_diagnostics(quote, text)


class AnswerabilityReviewRequired(InvalidLLMResponse):
    """Exact evidence exists, but automatic semantic acceptance is unsafe."""

    def __init__(self, reason: str, *, checkpoint: str | None = None) -> None:
        context = f"; checkpoint {checkpoint} preserved" if checkpoint else ""
        super().__init__(f"Answer requires source review: {reason}{context}")
        self.reason = reason


def _check_answer_support(question: str, answer: AnswerResponse) -> None:
    """Conservative triage, not an entailment model or automatic causal inference.

    Causal requests need review even when another sentence contains 'because':
    lexical causal markers cannot establish the requested causal relationship.
    Nonextractive responses also need review; a matching quote alone is insufficient.
    """
    if not answer.is_answerable:
        return
    if re.search(
        r"\b(?:why|reasons?|caus(?:e|es|ed|ing)|explain\s+how)\b", question, re.I
    ):
        raise AnswerabilityReviewRequired("causal_request_requires_review")
    if re.search(
        r"\b(?:summari[sz]e|summary|compare|comparison|both|and)\b", question, re.I
    ):
        raise AnswerabilityReviewRequired("multi_claim_request_requires_review")
    # This normalization is ONLY for answer/evidence coverage, never for admitting
    # or changing an evidence quote. Even extractive coverage is not full entailment.
    response = " ".join(answer.answer.casefold().split())
    evidence = " ".join(answer.evidence_quote.casefold().split())
    if response not in evidence:
        raise AnswerabilityReviewRequired("nonextractive_answer_requires_review")


def validate_grounded_answer(
    data: dict, text: str, *, question: str | None = None
) -> AnswerResponse:
    """Check exact evidence and, when supplied, conservative question-aware support."""
    try:
        answer = AnswerResponse.model_validate(data)
    except ValidationError as error:
        # Pydantic's default error string includes input values, potentially an
        # entire returned answer or document. Keep those out of persisted logs.
        raise InvalidLLMResponse(
            "Invalid answer response: schema validation failed"
        ) from error
    if answer.is_answerable and answer.evidence_quote not in text:
        raise EvidenceQuoteMismatch(answer.evidence_quote, text)
    if answer.is_answerable:
        # Save the actual original source slice, never a normalized reconstruction.
        start = text.index(answer.evidence_quote)
        answer.evidence_quote = text[start : start + len(answer.evidence_quote)]
    if question is not None:
        _check_answer_support(question, answer)
    return answer


def match_question_source(question: dict, chunk: dict) -> QuestionRecord:
    """Reject missing or mismatched source provenance before calling a provider."""
    record = QuestionRecord.model_validate(question)
    for field in ("chunk_id", "source_file", "page_number"):
        if type(chunk.get(field)) is not type(getattr(record, field)) or (
            chunk[field] != getattr(record, field)
        ):
            raise ValueError(f"Question/source {field} mismatch: {record.chunk_id}")
    if not isinstance(chunk.get("text"), str):
        raise ValueError(f"Missing or invalid source text: {record.chunk_id}")
    return record


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient, model: str) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "Answer generation model must be configured in "
                "config/data.yaml under answer_generation.model"
            )
        self.llm_client = llm_client
        self.model = model

    def generate(self, question: dict, chunk: dict) -> dict:
        record = match_question_source(question, chunk)
        content = self.llm_client.complete(
            model=self.model,
            prompt=build_answer_prompt(chunk["text"], record.question, record.category),
        )
        return validate_grounded_answer(
            parse_llm_json(content), chunk["text"], question=record.question
        ).model_dump()
