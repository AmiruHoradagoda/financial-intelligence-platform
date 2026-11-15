"""Focused financial table, evidence span, boundary and offline-decision checks."""

import json

import pytest
from pydantic import ValidationError

from financial_intelligence.cli import review_answer
from financial_intelligence.data_factory.answer_batch import (
    _metadata,
    _question_id,
    export_answer_records,
)
from financial_intelligence.data_factory.answer_generator import (
    AnswerabilityReviewRequired,
    EvidenceQuoteMismatch,
    validate_grounded_answer,
)
from financial_intelligence.infrastructure.jsonl import save_jsonl
from financial_intelligence.schemas.qa import AnswerResponse, QAPairRecord

TABLE = """Results of Operations
Consolidated statements of operations (in millions):
Year Ended December 31,
               2023          2024
Revenue                         $   37,281  $   43,978
Sales and marketing                   4,356       4,337
Research and development              3,164       3,109
Depreciation and amortization           823         711
Total costs and expenses              36,171      41,179"""


def row(label):
    return next(line for line in TABLE.splitlines() if line.startswith(label))


def response(answer, quote):
    return {"answer": answer, "is_answerable": True, "evidence_quote": quote}


@pytest.mark.parametrize(
    "question,answer,label",
    [
        (
            "How much was spent on research and development in 2023?",
            "$3,164 million",
            "Research",
        ),
        (
            "What was depreciation and amortization for 2024?",
            "$711 million",
            "Depreciation",
        ),
        ("What were total costs and expenses in 2023?", "$36,171 million", "Total"),
        (
            "Explain the changes in sales and marketing from 2023 to 2024 in a formal tone.",
            "Sales and marketing expenses decreased from $4,356 million in 2023 to $4,337 million in 2024.",
            "Sales",
        ),
    ],
)
def test_table_heading_and_column_context_support_historical_values(
    question, answer, label
):
    result = validate_grounded_answer(
        response(answer, row(label)), TABLE, question=question
    )
    assert result.answer == answer
    assert result.evidence_quote == row(label)


@pytest.mark.parametrize(
    "question,answer,label,reason",
    [
        (
            "What was research and development in 2023?",
            "$3,109 million",
            "Research",
            "table_year_value_mismatch",
        ),
        (
            "What was depreciation and amortization in 2024?",
            "$823 million",
            "Depreciation",
            "table_year_value_mismatch",
        ),
        (
            "What was research and development in 2023?",
            "$3,164 billion",
            "Research",
            "table_unit_mismatch",
        ),
        (
            "What was research and development in 2023?",
            "$823 million",
            "Depreciation",
            "table_row_mismatch",
        ),
        (
            "Compare sales and marketing for 2023 and 2024.",
            "Sales and marketing: $4,337 million in 2023 and $4,356 million in 2024.",
            "Sales",
            "table_year_value_mismatch",
        ),
        (
            "Compare sales and marketing for 2023 and 2024.",
            "Sales and marketing increased from $4,356 million in 2023 to $4,337 million in 2024.",
            "Sales",
            "table_change_direction_mismatch",
        ),
    ],
)
def test_table_values_cannot_be_borrowed_from_another_year_row_or_unit(
    question, answer, label, reason
):
    with pytest.raises(AnswerabilityReviewRequired) as error:
        validate_grounded_answer(response(answer, row(label)), TABLE, question=question)
    assert error.value.reason == reason
    assert error.value.approval_blocked


def test_table_context_does_not_allow_unquoted_other_row_amounts():
    with pytest.raises(AnswerabilityReviewRequired):
        validate_grounded_answer(
            response("Revenue was $3,164 million in 2024.", row("Revenue")),
            TABLE,
            question="What was revenue in 2024?",
        )


def test_legacy_single_quote_serialization_remains_unchanged():
    old = response("Revenue", "Revenue")
    assert AnswerResponse.model_validate(old).model_dump() == old


def test_multiple_short_table_quotes_are_preserved_as_separate_exact_spans():
    data = {
        "answer": "2023 revenue was $37,281 million and total costs and expenses were $36,171 million.",
        "is_answerable": True,
        "evidence_quote": None,
        "evidence_quotes": [row("Revenue"), row("Total")],
    }
    validated = validate_grounded_answer(
        data, TABLE, question="Summarize revenue and total costs and expenses for 2023."
    )
    assert validated.model_dump() == data
    assert all(quote in TABLE for quote in validated.evidence_quotes)
    metadata = {
        "chunk_id": "c1",
        "source_file": "uber_2024.pdf",
        "page_number": 57,
        "category": "strategic_summary",
        "question": "Summarize the excerpt.",
    }
    assert QAPairRecord.model_validate({**metadata, **data}).model_dump() == {
        **metadata,
        **data,
    }


@pytest.mark.parametrize(
    "quotes",
    [
        [row("Revenue"), "invented row"],
        [row("Revenue").replace("   ", " ")],
        [row("Revenue") + " ... " + row("Total")],
    ],
)
def test_every_span_must_match_without_stitching_or_whitespace_changes(quotes):
    with pytest.raises(EvidenceQuoteMismatch):
        validate_grounded_answer(
            {
                "answer": "Revenue",
                "is_answerable": True,
                "evidence_quote": None,
                "evidence_quotes": quotes,
            },
            TABLE,
        )


def test_unanswerable_cannot_carry_any_evidence_span():
    with pytest.raises(ValidationError):
        AnswerResponse.model_validate(
            {
                "answer": None,
                "is_answerable": False,
                "evidence_quote": None,
                "evidence_quotes": [row("Revenue")],
            }
        )


def test_table_does_not_explain_a_cause_merely_by_stating_two_values():
    with pytest.raises(AnswerabilityReviewRequired, match="causal_request"):
        validate_grounded_answer(
            response(
                "Sales and marketing were $4,356 million and $4,337 million.",
                row("Sales"),
            ),
            TABLE,
            question="Why did sales and marketing decrease?",
        )


def test_boundary_fragment_is_flagged_even_when_quote_is_exact():
    source = "Our success depends on attracting Drivers, consumers, merch"
    with pytest.raises(AnswerabilityReviewRequired) as error:
        validate_grounded_answer(
            response(source, source), source, question="Summarize market success."
        )
    assert error.value.reason == "incomplete_source_boundary"
    assert error.value.approval_blocked


def test_offline_decisions_approve_without_touching_checkpoint_or_registry(tmp_path):
    chunk = {
        "chunk_id": "c1",
        "source_file": "uber_2024.pdf",
        "page_number": 57,
        "text": TABLE,
    }
    question = {k: chunk[k] for k in ("chunk_id", "source_file", "page_number")}
    question.update(
        category="hard_fact", question="How much was research and development in 2023?"
    )
    data = response("$3,164 million", row("Research"))
    qid = _question_id(question)
    checkpoint = tmp_path / f"{qid}.json"
    envelope = {
        "question_id": qid,
        "metadata": _metadata(question, chunk, "fake/model"),
        "record": {**question, **data},
        "review_required": True,
        "review_reason": "unsupported_financial_unit",
    }
    checkpoint.write_text(json.dumps(envelope), encoding="utf-8")
    registry = tmp_path / "review_required.jsonl"
    registry.write_text(
        json.dumps({**envelope, "review_status": "required"}) + "\n", encoding="utf-8"
    )
    before = checkpoint.read_bytes(), registry.read_bytes()
    questions_path, chunks_path = (
        tmp_path / "questions.jsonl",
        tmp_path / "chunks.jsonl",
    )
    save_jsonl([question], questions_path)
    save_jsonl([chunk], chunks_path)
    args = [
        "--checkpoint-dir",
        str(tmp_path),
        "--question-id",
        qid,
        "--decision",
        "approve",
        "--reviewer",
        "offline-test-reviewer",
        "--notes",
        "Verified 2023 cell and in-millions heading.",
        "--questions",
        str(questions_path),
        "--chunks",
        str(chunks_path),
    ]
    review_answer.main([*args, "--dry-run"])
    assert not (tmp_path / "review_decisions.jsonl").exists()
    review_answer.main(args)
    assert (
        export_answer_records(
            [question], [chunk], "fake/model", tmp_path, tmp_path / "qa.jsonl"
        )
        == 1
    )
    assert (checkpoint.read_bytes(), registry.read_bytes()) == before
    # Decisions cannot be reused after any checkpoint mutation.
    checkpoint.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        export_answer_records(
            [question], [chunk], "fake/model", tmp_path, tmp_path / "qa.jsonl"
        )


def test_reversed_year_header_changes_the_cell_mapping():
    reversed_header = TABLE.replace("2023          2024", "2024          2023")
    with pytest.raises(AnswerabilityReviewRequired, match="table_year_value_mismatch"):
        validate_grounded_answer(
            response("$3,164 million", row("Research")),
            reversed_header,
            question="How much was research and development in 2023?",
        )


def test_units_cannot_be_invented_when_the_heading_does_not_supply_them():
    with pytest.raises(AnswerabilityReviewRequired, match="unsupported_financial_unit"):
        validate_grounded_answer(
            response("$3,164 million", row("Research")),
            TABLE.replace("(in millions)", ""),
            question="How much was research and development in 2023?",
        )


def test_multi_quote_checkpoint_resume_retains_all_exact_spans(tmp_path):
    from financial_intelligence.data_factory.answer_batch import run_answer_batch
    from financial_intelligence.data_factory.answer_generator import AnswerGenerator
    from financial_intelligence.infrastructure.jsonl import load_chunks

    question = {
        "chunk_id": "c1",
        "source_file": "uber_2024.pdf",
        "page_number": 57,
        "category": "strategic_summary",
        "question": "Summarize revenue and total costs and expenses for 2023.",
    }
    chunk = {
        "chunk_id": "c1",
        "source_file": "uber_2024.pdf",
        "page_number": 57,
        "text": TABLE,
    }
    data = {
        "answer": "2023 revenue was $37,281 million and total costs and expenses were $36,171 million.",
        "is_answerable": True,
        "evidence_quote": None,
        "evidence_quotes": [row("Revenue"), row("Total")],
    }

    class FakeLLM:
        calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            return json.dumps(data)

    client = FakeLLM()
    generator = AnswerGenerator(client, "fake/model")
    output = tmp_path / "qa.jsonl"
    result = run_answer_batch(
        [question], [chunk], generator, tmp_path, output_path=output
    )
    assert result["generated"] == 1
    assert load_chunks(output) == [{**question, **data}]
    before = (tmp_path / f"{_question_id(question)}.json").read_bytes()
    result = run_answer_batch(
        [question], [chunk], generator, tmp_path, output_path=output
    )
    assert result["skipped"] == 1 and client.calls == 1
    assert (tmp_path / f"{_question_id(question)}.json").read_bytes() == before


def test_offline_approval_cannot_override_a_boundary_fragment(tmp_path):
    chunk = {
        "chunk_id": "c1",
        "source_file": "uber_2024.pdf",
        "page_number": 18,
        "text": "Our success depends on attracting Drivers, consumers, merch",
    }
    question = {key: chunk[key] for key in ("chunk_id", "source_file", "page_number")}
    question.update(category="strategic_summary", question="Summarize market success.")
    qid = _question_id(question)
    checkpoint = tmp_path / f"{qid}.json"
    checkpoint.write_text(
        json.dumps(
            {
                "question_id": qid,
                "metadata": _metadata(question, chunk, "fake/model"),
                "record": {**question, **response(chunk["text"], chunk["text"])},
            }
        ),
        encoding="utf-8",
    )
    questions_path, chunks_path = (
        tmp_path / "questions.jsonl",
        tmp_path / "chunks.jsonl",
    )
    save_jsonl([question], questions_path)
    save_jsonl([chunk], chunks_path)
    before = checkpoint.read_bytes()
    args = [
        "--checkpoint-dir",
        str(tmp_path),
        "--question-id",
        qid,
        "--decision",
        "approve",
        "--reviewer",
        "reviewer",
        "--notes",
        "Checked source.",
        "--questions",
        str(questions_path),
        "--chunks",
        str(chunks_path),
    ]
    with pytest.raises(SystemExit) as error:
        review_answer.main(args)
    assert error.value.code == 1
    assert not (tmp_path / "review_decisions.jsonl").exists()
    assert checkpoint.read_bytes() == before
    args[args.index("approve")] = "reject"
    review_answer.main(args)
    assert (
        export_answer_records(
            [question], [chunk], "fake/model", tmp_path, tmp_path / "qa.jsonl"
        )
        == 0
    )
    assert checkpoint.read_bytes() == before


def test_table_excerpt_does_not_justify_rewriting_an_entire_section():
    with pytest.raises(
        AnswerabilityReviewRequired, match="question_scope_exceeds_excerpt"
    ):
        validate_grounded_answer(
            response("Revenue was $43,978 million in 2024.", row("Revenue")),
            TABLE,
            question="Rewrite the 'Results of Operations' section in plain English.",
        )


def test_year_before_row_label_is_still_checked():
    with pytest.raises(AnswerabilityReviewRequired, match="table_year_value_mismatch"):
        validate_grounded_answer(
            response(
                "In 2023 sales and marketing were $4,337 million; in 2024 sales and marketing were $4,356 million.",
                row("Sales"),
            ),
            TABLE,
            question="Compare sales and marketing for 2023 and 2024.",
        )
