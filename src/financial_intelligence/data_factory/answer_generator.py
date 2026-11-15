"""Injected answer generation with strict response and source-quote validation."""

import hashlib
import re

from pydantic import ValidationError

from financial_intelligence.data_factory.prompts import build_answer_prompt
from financial_intelligence.data_factory.table_context import table_support
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

    def __init__(
        self,
        reason: str,
        *,
        checkpoint: str | None = None,
        answer: dict | None = None,
        approval_blocked: bool = False,
    ) -> None:
        context = f"; checkpoint {checkpoint} preserved" if checkpoint else ""
        super().__init__(f"Answer requires source review: {reason}{context}")
        self.reason = reason
        self.answer = answer
        self.approval_blocked = approval_blocked


def _check_answer_support(question: str, answer: AnswerResponse, text: str) -> None:
    """Flag observable grounding risks while permitting paraphrases.

    These lexical checks are triage, not semantic entailment or a quality score.
    Whitespace/token normalization never admits or modifies an evidence quote.
    """
    if not answer.is_answerable:
        return

    def review(reason: str) -> None:
        raise AnswerabilityReviewRequired(
            reason,
            answer=answer.model_dump(),
            approval_blocked=reason
            in {
                "incomplete_source_boundary",
                "table_row_mismatch",
                "table_unit_mismatch",
                "table_year_value_mismatch",
                "table_change_direction_mismatch",
                "unsupported_numeric_claim",
                "unsupported_financial_unit",
            },
        )

    quotes = evidence_spans(answer)
    evidence = "\n".join(quotes)  # Support checks only; never a persisted quote.
    response = answer.answer
    tail = re.search(r"([A-Za-z]{2,})$", text.rstrip())
    ending = re.search(r"([A-Za-z]{2,})[.!?]?$", response.rstrip())
    if (
        tail
        and ending
        and tail[1] == ending[1]
        and len(response) > 30
        and any(text.rstrip().endswith(quote.rstrip()) for quote in quotes)
    ):
        review("incomplete_source_boundary")
    if re.search(
        r"(?:entire|whole).*\bsection\b|(?:rewrite|summarize).*results of operations['\"]?\s+section",
        question,
        re.I,
    ) and re.search(r"\(in\s+millions\)", text, re.I):
        review("question_scope_exceeds_excerpt")

    if re.search(
        r"\b(?:why|reasons?|caus(?:e|es|ed|ing)|explain\s+how)\b", question, re.I
    ):
        # An unrelated 'because' elsewhere cannot establish the requested cause.
        review("causal_request_requires_review")

    stopwords = set(
        "a an the is are was were be been being to of in on at for from with as by "
        "and or that this these those it its our their they we you which who what "
        "how does do did has have had can could may might will would should not "
        "according excerpt provided source report company describe described "
        "summarize summary explain explanation provide give rewrite present "
        "plain english formal tone investor friendly audience executive concise "
        "bullet points question request".split()
    )

    def terms(value: str) -> set[str]:
        words = re.findall(r"[a-z]+", value.casefold())
        # Basic plural equivalence, not fuzzy evidence matching.
        return {
            word[:-3] + "y"
            if word.endswith("ies") and len(word) > 4
            else word[:-1]
            if word.endswith("s") and not word.endswith("ss") and len(word) > 3
            else word
            for word in words
            if word not in stopwords
        }

    def numbers(value: str) -> set[str]:
        return {
            number.replace(",", "") for number in re.findall(r"\d+(?:[,.]\d+)*", value)
        }

    header_numbers, header_units, table_error = table_support(
        text, quotes, question, response
    )
    if table_error:
        review(table_error)
    if header_numbers and re.search(
        r"\b(?:because|due to|driven by|as a result of)\b", response, re.I
    ):
        review("unstated_table_cause")
    if not numbers(response) <= numbers(evidence) | header_numbers:
        review("unsupported_numeric_claim")
    units = {"million", "billion", "trillion", "percent"}
    if not terms(response).intersection(units) <= terms(evidence) | header_units:
        review("unsupported_financial_unit")
    evidence_terms = terms(evidence)
    topics = terms(question).intersection(terms(text))
    if topics and not topics.intersection(evidence_terms):
        review("evidence_does_not_address_question")
    multi_topic = re.search(r"\b(?:and|both|compare|comparison)\b", question, re.I)
    if (
        multi_topic
        and not header_numbers
        and len(topics) >= 2
        and not topics <= terms(response)
    ):
        review("incomplete_question_coverage")
    if (
        re.search(r"\bdelivery\b", question, re.I)
        and not re.search(r"\b(?:cities|products)\b", question, re.I)
        and re.search(r"\bnew (?:cities|products)\b", response, re.I)
    ):
        review("answer_scope_drift")
    if (
        re.search(r"\bnew cities\b", question, re.I)
        and not re.search(r"\bdelivery\b", question, re.I)
        and re.search(r"\b(?:delivery|service fees|carriers)\b", response, re.I)
    ):
        review("answer_scope_drift")
    for sentence in re.split(r"[.!?](?:\s|$)", response):
        if header_numbers and numbers(sentence):
            # Table amounts/years/units were checked against quoted rows plus
            # their original header; answers need not repeat the row label.
            continue
        claims = terms(sentence)
        if (
            len(claims) >= 2
            and len(claims.intersection(evidence_terms)) / len(claims) < 0.35
        ):
            review("answer_evidence_topic_mismatch")


def evidence_spans(answer: AnswerResponse) -> list[str]:
    """Return separate supplied quotes; never stitch them into stored evidence."""
    return ([answer.evidence_quote] if answer.evidence_quote is not None else []) + (
        answer.evidence_quotes or []
    )


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
    for quote in evidence_spans(answer):
        if quote not in text:
            raise EvidenceQuoteMismatch(quote, text)
    if answer.is_answerable:
        # Save the actual original source slice, never a normalized reconstruction.
        if answer.evidence_quote is not None:
            start = text.index(answer.evidence_quote)
            answer.evidence_quote = text[start : start + len(answer.evidence_quote)]
        if answer.evidence_quotes is not None:
            answer.evidence_quotes = [
                text[text.index(q) : text.index(q) + len(q)]
                for q in answer.evidence_quotes
            ]
    if question is not None:
        _check_answer_support(question, answer, text)
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
