"""Grounded-answer contracts, batches, and CLI behavior without provider access."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_intelligence.cli import generate_answers as cli
from financial_intelligence.data_factory import answer_batch as batch
from financial_intelligence.data_factory.answer_generator import (
    AnswerGenerator,
    EvidenceQuoteMismatch,
    validate_grounded_answer,
)
from financial_intelligence.data_factory.prompts import build_answer_prompt
from financial_intelligence.infrastructure.jsonl import load_chunks, save_jsonl
from financial_intelligence.infrastructure.llm_client import InvalidLLMResponse
from financial_intelligence.schemas.qa import AnswerResponse, QAPairRecord


@pytest.fixture
def chunk():
    return {
        "chunk_id": "uber_2024_data_factory_p57_c1",
        "source_file": "uber_2024.pdf",
        "page_number": 57,
        "text": "Revenue in 2024 was $43,978 million.\nRevenue in 2023 was $37,281 million.",
    }


@pytest.fixture
def question(chunk):
    return {
        **{key: chunk[key] for key in ("chunk_id", "source_file", "page_number")},
        "category": "hard_fact",
        "question": "What was revenue in 2024?",
    }


@pytest.fixture
def questions(question):
    return [
        {**question, "question": f"{question['question']} Request {i}"}
        for i in range(8)
    ]


@pytest.fixture
def answer():
    return {
        "answer": "Revenue in 2024 was $43,978 million.",
        "is_answerable": True,
        "evidence_quote": "Revenue in 2024 was $43,978 million.",
    }


class FakeLLM:
    def __init__(self, answer, outcomes=()):
        self.response = json.dumps(answer)
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, *, model, prompt):
        self.calls.append({"model": model, "prompt": prompt})
        outcome = self.outcomes.pop(0) if self.outcomes else self.response
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_grounded_answer_preserves_financial_units_years_and_metadata(
    chunk, question, answer
):
    client = FakeLLM(answer)
    generated = AnswerGenerator(client, "answer/model").generate(question, chunk)
    assert generated == answer
    pair = QAPairRecord.model_validate({**question, **generated}).model_dump()
    assert pair == {**question, **answer}
    assert pair["answer"] == "Revenue in 2024 was $43,978 million."
    assert client.calls == [
        {
            "model": "answer/model",
            "prompt": build_answer_prompt(
                chunk["text"], question["question"], question["category"]
            ),
        }
    ]


def test_unanswerable_question_has_nulls(chunk, question):
    response = {"answer": None, "is_answerable": False, "evidence_quote": None}
    question["question"] = "Why did revenue increase?"
    assert (
        AnswerGenerator(FakeLLM(response), "model").generate(question, chunk)
        == response
    )


@pytest.mark.parametrize(
    "update",
    [
        {"answer": None},
        {"answer": " "},
        {"evidence_quote": None},
        {"evidence_quote": "\n"},
        {"is_answerable": "true"},
        {"is_answerable": False},
        {"unexpected": "field"},
    ],
)
def test_invalid_answer_schema_rejected(answer, update):
    with pytest.raises(ValidationError):
        AnswerResponse.model_validate({**answer, **update})


@pytest.mark.parametrize(
    "answer,evidence", [("", None), (None, "quote"), ("answer", "quote")]
)
def test_unanswerable_requires_actual_nulls(answer, evidence):
    with pytest.raises(ValidationError):
        AnswerResponse.model_validate(
            {
                "answer": answer,
                "is_answerable": False,
                "evidence_quote": evidence,
            }
        )


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "[]",
        "{}",
        '{"answer": "Revenue", "is_answerable": true}',
        '{"answer": "Revenue", "is_answerable": true, "evidence_quote": null}',
    ],
)
def test_generator_rejects_invalid_json_and_missing_evidence(
    chunk, question, answer, response
):
    client = FakeLLM(answer, [response])
    with pytest.raises(InvalidLLMResponse):
        AnswerGenerator(client, "model").generate(question, chunk)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "quote",
    [
        "Revenue was $99 million.",
        "revenue in 2024 was $43,978 million.",
        "Revenue in 2024 was $43,978 million. Revenue in 2023",
    ],
)
def test_quote_must_match_exactly(chunk, question, answer, quote):
    answer["evidence_quote"] = quote
    with pytest.raises(InvalidLLMResponse, match="exact substring"):
        AnswerGenerator(FakeLLM(answer), "model").generate(question, chunk)


def test_exact_quote_preserves_newlines_and_stylistic_format(chunk, question):
    question.update(
        category="stylistic",
        question="Give the two revenue figures in two bullet points.",
    )
    answer = {
        "answer": "- 2024: $43,978 million\n- 2023: $37,281 million",
        "is_answerable": True,
        "evidence_quote": chunk["text"],
    }
    client = FakeLLM(answer)
    assert AnswerGenerator(client, "model").generate(question, chunk) == answer
    assert question["question"] in client.calls[0]["prompt"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_id", "other"),
        ("source_file", "other.pdf"),
        ("page_number", 1),
        ("page_number", True),
    ],
)
def test_source_mismatch_rejected_before_calls(chunk, question, answer, field, value):
    chunk[field] = value
    client = FakeLLM(answer)
    with pytest.raises(ValueError, match="mismatch"):
        AnswerGenerator(client, "model").generate(question, chunk)
    assert not client.calls


@pytest.mark.parametrize("model", [None, "", "  "])
def test_answer_model_must_be_configured(answer, model):
    with pytest.raises(ValueError, match="answer_generation.model"):
        AnswerGenerator(FakeLLM(answer), model)


def test_answer_prompt_grounding_contract():
    prompt = build_answer_prompt("SOURCE", "QUESTION", "stylistic")
    for rule in (
        "ONLY the original chunk",
        "never use external knowledge",
        "units, years",
        "boundaries",
        "format, tone, and audience",
        "exact contiguous",
        "unanswerable",
        "JSON null",
    ):
        assert rule in prompt
    data = json.loads(prompt.split("INPUT:\n", 1)[1])
    assert data == {
        "category": "stylistic",
        "question": "QUESTION",
        "original_chunk": "SOURCE",
    }


def test_answer_batch_defaults_resume_and_export(tmp_path, chunk, questions, answer):
    client = FakeLLM(answer)
    generator = AnswerGenerator(client, "answer/model")
    checkpoints = tmp_path / "checkpoints"
    output = tmp_path / "qa.jsonl"
    first = batch.run_answer_batch(
        questions, [chunk], generator, checkpoints, output_path=output
    )
    assert first["generated"] == first["attempted"] == 5
    assert first["remaining"] == 3
    original = {p.name: p.read_bytes() for p in checkpoints.glob("*.json")}
    saved = json.loads(next(iter(original.values())))
    assert saved["metadata"]["model"] == "answer/model"
    assert len(saved["metadata"]["prompt_fingerprint"]) == 64
    assert len(load_chunks(output)) == 5

    second = batch.run_answer_batch(
        questions, [chunk], generator, checkpoints, output_path=output
    )
    assert second["generated"] == 3
    assert second["skipped"] == 5
    assert second["remaining"] == 0
    assert len(client.calls) == 8
    assert all(
        (checkpoints / name).read_bytes() == content
        for name, content in original.items()
    )
    assert load_chunks(output) == [{**q, **answer} for q in questions]
    assert not list(checkpoints.glob("*.tmp"))

    # Reordering inputs never repeats completed calls; export follows input order.
    third = batch.run_answer_batch(
        questions[::-1], [chunk], generator, checkpoints, output_path=output
    )
    assert third["skipped"] == 8
    assert third["generated"] == 0
    assert len(client.calls) == 8
    assert load_chunks(output) == [{**q, **answer} for q in questions[::-1]]
    assert (
        batch.export_answer_records(
            questions, [chunk], generator.model, checkpoints, output
        )
        == 8
    )
    assert load_chunks(output) == [{**q, **answer} for q in questions]


def test_duplicates_do_not_spend_or_export_twice(tmp_path, chunk, question, answer):
    client = FakeLLM(answer)
    result = batch.run_answer_batch(
        [question, question],
        [chunk],
        AnswerGenerator(client, "model"),
        tmp_path,
        output_path=tmp_path / "qa.jsonl",
    )
    assert result["generated"] == result["duplicates"] == result["exported"] == 1
    assert len(client.calls) == 1


def test_repeated_question_in_different_chunks_is_distinct(
    tmp_path, chunk, question, answer
):
    other_chunk = {**chunk, "chunk_id": "other"}
    other_question = {**question, "chunk_id": "other"}
    client = FakeLLM(answer)
    result = batch.run_answer_batch(
        [question, other_question],
        [chunk, other_chunk],
        AnswerGenerator(client, "model"),
        tmp_path,
    )
    assert result["generated"] == 2
    assert len(client.calls) == 2


def test_unanswerable_is_successfully_checkpointed(tmp_path, chunk, question):
    answer = {"answer": None, "is_answerable": False, "evidence_quote": None}
    client = FakeLLM(answer)
    generator = AnswerGenerator(client, "model")
    output = tmp_path / "qa.jsonl"
    result = batch.run_answer_batch(
        [question], [chunk], generator, tmp_path, output_path=output
    )
    assert result["generated"] == result["unanswerable"] == 1
    assert result["failed"] == 0
    assert load_chunks(output) == [{**question, **answer}]
    assert (
        batch.run_answer_batch([question], [chunk], generator, tmp_path)["skipped"] == 1
    )
    assert len(client.calls) == 1


def test_dry_run_never_calls_or_writes(tmp_path, chunk, questions, answer):
    client = FakeLLM(answer)
    result = batch.run_answer_batch(
        questions,
        [chunk],
        AnswerGenerator(client, "model"),
        tmp_path / "checkpoints",
        dry_run=True,
        output_path=tmp_path / "qa.jsonl",
    )
    assert result["planned"] == 5
    assert result["remaining"] == 8
    assert result["generated"] == 0
    assert not client.calls
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    ["model", "prompt", "text", "category", "evidence", "metadata", "id", "corrupt"],
)
def test_incompatible_checkpoints_fail_before_spending(
    tmp_path, chunk, questions, answer, monkeypatch, change
):
    client = FakeLLM(answer)
    generator = AnswerGenerator(client, "model")
    batch.run_answer_batch(questions, [chunk], generator, tmp_path, max_new_questions=1)
    path = next(tmp_path.glob("*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    if change == "model":
        generator.model = "new/model"
    elif change == "prompt":
        monkeypatch.setattr(batch, "build_answer_prompt", lambda *args: "changed")
    elif change == "text":
        chunk["text"] += " More text."
    elif change == "category":
        questions[0]["category"] = "stylistic"
    elif change == "evidence":
        saved["record"]["evidence_quote"] = "Unsupported quote"
    elif change == "metadata":
        saved["record"]["page_number"] += 1
    elif change == "id":
        saved["question_id"] = "other"
    path.write_text("{" if change == "corrupt" else json.dumps(saved), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible answer checkpoint"):
        batch.run_answer_batch(questions, [chunk], generator, tmp_path)
    assert len(client.calls) == 1
    assert path.read_bytes() == before


class APIError(Exception):
    def __init__(self, status):
        self.status_code = status
        super().__init__(f"API status {status}")


@pytest.mark.parametrize(
    "error", [TimeoutError(), ConnectionError(), APIError(429), APIError(503)]
)
def test_transient_retries_are_bounded_with_backoff(
    tmp_path, chunk, question, answer, monkeypatch, error
):
    delays = []
    monkeypatch.setattr(batch.time, "sleep", delays.append)
    client = FakeLLM(answer, [error, error])
    result = batch.run_answer_batch(
        [question], [chunk], AnswerGenerator(client, "model"), tmp_path
    )
    assert result["generated"] == 1
    assert len(client.calls) == 3
    assert delays == [1, 2]


def test_failed_questions_are_logged_and_successes_exported(
    tmp_path, chunk, questions, answer, monkeypatch
):
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    client = FakeLLM(answer, [APIError(503)] * 3)
    generator = AnswerGenerator(client, "model")
    output = tmp_path / "qa.jsonl"
    result = batch.run_answer_batch(
        questions, [chunk], generator, tmp_path, max_new_questions=2, output_path=output
    )
    assert result["failed"] == result["generated"] == 1
    assert result["attempted"] == 2
    assert result["remaining"] == 7
    assert len(client.calls) == 4
    failure = load_chunks(tmp_path / "failures.jsonl")[0]
    assert failure["question"] == questions[0]["question"]
    assert failure["attempts"] == 3
    assert failure["metadata"]["model"] == "model"
    assert load_chunks(output) == [{**questions[1], **answer}]
    result = batch.run_answer_batch(
        questions,
        [chunk],
        generator,
        tmp_path,
        max_new_questions=1,
        retry_failed_only=True,
    )
    assert result["skipped"] == result["generated"] == 1


@pytest.mark.parametrize(
    "failure",
    [
        "not JSON",
        APIError(401),
        APIError(400),
        '{"answer":"invented", "is_answerable":true, "evidence_quote":"not in source"}',
    ],
)
def test_invalid_or_permanent_failures_not_retried(
    tmp_path, chunk, questions, answer, failure
):
    client = FakeLLM(answer, [failure])
    result = batch.run_answer_batch(
        questions,
        [chunk],
        AnswerGenerator(client, "model"),
        tmp_path,
        max_new_questions=2,
    )
    assert result["failed"] == result["generated"] == 1
    assert len(client.calls) == 2
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate_chunk", "conflicting_question"]
)
def test_invalid_joins_and_conflicts_fail_before_calls(
    tmp_path, chunk, question, answer, problem
):
    chunks = [chunk]
    questions = [question]
    if problem == "missing":
        questions[0] = {**question, "chunk_id": "absent"}
    elif problem == "duplicate_chunk":
        chunks.append(chunk)
    else:
        questions.append({**question, "category": "stylistic"})
    client = FakeLLM(answer)
    with pytest.raises(ValueError):
        batch.run_answer_batch(
            questions, chunks, AnswerGenerator(client, "model"), tmp_path
        )
    assert not client.calls
    assert not list(tmp_path.iterdir())


def test_interrupted_atomic_write_keeps_previous_successes(
    tmp_path, chunk, questions, answer, monkeypatch
):
    generator = AnswerGenerator(FakeLLM(answer), "model")
    output = tmp_path / "qa.jsonl"
    batch.run_answer_batch(
        questions, [chunk], generator, tmp_path, max_new_questions=1, output_path=output
    )
    original = output.read_bytes()
    replace = Path.replace

    def interrupted(path, target):
        if target == output:
            raise OSError("interrupted export")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted export"):
        batch.run_answer_batch(
            questions,
            [chunk],
            generator,
            tmp_path,
            max_new_questions=1,
            output_path=output,
        )
    assert output.read_bytes() == original
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(Path, "replace", replace)
    assert (
        batch.export_answer_records(
            questions, [chunk], generator.model, tmp_path, output
        )
        == 2
    )


@pytest.mark.parametrize(
    "options", [{"max_new_questions": 0}, {"max_attempts": 0}, {"max_attempts": 4}]
)
def test_invalid_limits_do_not_spend(tmp_path, chunk, question, answer, options):
    client = FakeLLM(answer)
    with pytest.raises(ValueError):
        batch.run_answer_batch(
            [question], [chunk], AnswerGenerator(client, "model"), tmp_path, **options
        )
    assert not client.calls


def test_cli_dry_run_pilot_resume_export_and_input_preservation(
    tmp_path, chunk, questions, answer, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    questions_path = tmp_path / "data/generated/questions.jsonl"
    chunks_path = tmp_path / "data/processed/data_factory/uber_2024_chunks.jsonl"
    save_jsonl(questions, questions_path)
    save_jsonl([chunk], chunks_path)
    original_questions = questions_path.read_bytes()
    original_chunks = chunks_path.read_bytes()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda name: {
            "data_generation": {"model": "ignored/question-model"},
            "answer_generation": {"model": "answer/model"},
        },
    )
    client = FakeLLM(answer)

    def make_client(**kwargs):
        assert kwargs == {"num_retries": 0}
        return client

    monkeypatch.setattr(cli, "LiteLLMClient", make_client)
    cli.main(["--dry-run"])
    assert not client.calls
    assert not (tmp_path / "data/generated/answer_checkpoints").exists()
    output = tmp_path / "data/generated/qa_pairs_preview.jsonl"
    assert not output.exists()
    cli.main([])
    assert len(client.calls) == 5
    assert all(call["model"] == "answer/model" for call in client.calls)
    assert len(load_chunks(output)) == 5
    cli.main(["--max-new-questions", "1"])
    assert len(client.calls) == 6
    cli.main(["--export-only"])
    assert len(client.calls) == 6
    assert len(load_chunks(output)) == 6
    assert questions_path.read_bytes() == original_questions
    assert chunks_path.read_bytes() == original_chunks
    assert "Skipped completed: 5" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        cli.main(["--output", str(questions_path)])
    assert error.value.code == 2


def test_cli_missing_answer_model_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda name: {"data_generation": {"model": "question/model"}},
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["--dry-run"])
    assert error.value.code == 1


def test_cli_failures_exit_nonzero_after_preserving_successes(
    tmp_path, chunk, questions, answer, monkeypatch
):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli, "load_config", lambda name: {"answer_generation": {"model": "model"}}
    )
    save_jsonl(questions, tmp_path / "data/generated/questions.jsonl")
    save_jsonl([chunk], tmp_path / "data/processed/data_factory/uber_2024_chunks.jsonl")
    client = FakeLLM(answer, ["invalid"])
    monkeypatch.setattr(cli, "LiteLLMClient", lambda **kwargs: client)
    with pytest.raises(SystemExit) as error:
        cli.main(["--max-new-questions", "2"])
    assert error.value.code == 1
    assert len(load_chunks(tmp_path / "data/generated/qa_pairs_preview.jsonl")) == 1


def test_cli_import_does_not_load_provider(capsys):
    # All generation tests inject clients; this suite has no need to import LiteLLM.
    assert "litellm" not in sys.modules
    assert callable(cli.main)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "quote",
    ["Uber\u2019s mission\nhelps people move.", "helps people move.", "mission\nhelps"],
)
def test_verbatim_quotes_preserve_unicode_and_newlines(quote):
    source = "Uber\u2019s mission\nhelps people move."
    response = {
        "answer": "It helps people move.",
        "is_answerable": True,
        "evidence_quote": quote,
    }
    assert validate_grounded_answer(response, source).evidence_quote == quote


@pytest.mark.parametrize(
    "quote,reason",
    [
        ("Uber\u2019s mission helps people move.", "whitespace_mismatch"),
        ("Uber's mission\nhelps people move.", "unicode_apostrophe_mismatch"),
        (
            "Uber's mission helps people move.",
            "whitespace_and_unicode_apostrophe_mismatch",
        ),
        ("Uber\u2019s mission\\nhelps people move.", "literal_escape_mismatch"),
        ("Uber transports people worldwide.", "no_contiguous_verbatim_span"),
        ("Uber\u2019s mission...helps people move.", "no_contiguous_verbatim_span"),
        ("uber\u2019s mission\nhelps people move.", "no_contiguous_verbatim_span"),
    ],
)
def test_quote_mismatch_diagnostics_never_relax_acceptance(quote, reason):
    source = "Uber\u2019s mission\nhelps people move."
    response = {
        "answer": "It helps people move.",
        "is_answerable": True,
        "evidence_quote": quote,
    }
    with pytest.raises(EvidenceQuoteMismatch) as error:
        validate_grounded_answer(response, source)
    assert error.value.diagnostics["reason"] == reason
    assert error.value.diagnostics["quote_repr"] == ascii(quote)
    assert error.value.diagnostics["quote_length"] == len(quote)
    assert source not in str(error.value)


def test_schema_failures_do_not_include_answer_or_private_document():
    private_text = "PRIVATE DOCUMENT CONTENT " * 200
    with pytest.raises(InvalidLLMResponse) as error:
        validate_grounded_answer(
            {"answer": private_text, "is_answerable": True}, "source"
        )
    assert str(error.value) == "Invalid answer response: schema validation failed"
    assert "PRIVATE DOCUMENT" not in str(error.value)


def test_failure_log_has_bounded_redacted_quote_diagnostics(
    tmp_path, chunk, question, answer
):
    secret = "sk-privateCredential123"
    answer["evidence_quote"] = (
        secret + " Bearer hiddenCredential api_key=otherCredential " + "PRIVATE " * 500
    )
    client = FakeLLM(answer)
    result = batch.run_answer_batch(
        [question], [chunk], AnswerGenerator(client, "model"), tmp_path
    )
    assert result["failed"] == 1
    assert result["generated"] == 0
    assert len(client.calls) == 1
    logged = (tmp_path / "failures.jsonl").read_text(encoding="utf-8")
    for sensitive in [
        secret,
        "hiddenCredential",
        "otherCredential",
        chunk["text"],
        answer["evidence_quote"],
    ]:
        assert sensitive not in logged
    diagnostics = json.loads(logged)["evidence_diagnostics"]
    assert diagnostics["reason"] == "no_contiguous_verbatim_span"
    assert diagnostics["quote_repr_truncated"] is True
    assert len(diagnostics["quote_repr"]) <= 512
    assert diagnostics["quote_length"] == len(answer["evidence_quote"])
    assert "REDACTED" in diagnostics["quote_repr"]
    assert len(list(tmp_path.glob("*.json"))) == 0


def test_provider_exception_does_not_log_credentials_or_full_prompt(
    tmp_path, chunk, question, answer
):
    error = RuntimeError("Authorization: Bearer SECRET " + chunk["text"])
    client = FakeLLM(answer, [error])
    batch.run_answer_batch(
        [question], [chunk], AnswerGenerator(client, "model"), tmp_path
    )
    failure = load_chunks(tmp_path / "failures.jsonl")[0]
    assert failure["error"] == "Answer generation or validation failed"
    assert "SECRET" not in json.dumps(failure)
    assert chunk["text"] not in json.dumps(failure)


def test_prompt_explicitly_requires_short_verbatim_span_and_json_escaping():
    prompt = build_answer_prompt("SOURCE", "QUESTION", "hard_fact")
    for instruction in [
        "ONE short",
        "copied verbatim",
        "shortest supporting clause",
        "EVERY character",
        "Unicode characters",
        "line breaks",
        "No paraphrasing",
        "stitched fragments",
        "double-escape",
    ]:
        assert instruction in prompt
    assert r"line breaks as \n and tabs as \t" in prompt


def test_explicit_prior_prompt_reuse_preserves_checkpoint_bytes(
    tmp_path, chunk, questions, answer, monkeypatch
):
    client = FakeLLM(answer)
    generator = AnswerGenerator(client, "model")
    batch.run_answer_batch(questions, [chunk], generator, tmp_path, max_new_questions=1)
    path = next(tmp_path.glob("*.json"))
    original = path.read_bytes()
    old_fingerprint = json.loads(original)["metadata"]["prompt_fingerprint"]
    monkeypatch.setattr(batch, "build_answer_prompt", lambda *args: "revised prompt")
    with pytest.raises(ValueError, match="incompatible"):
        batch.run_answer_batch(questions, [chunk], generator, tmp_path, dry_run=True)
    assert len(client.calls) == 1
    result = batch.run_answer_batch(
        questions,
        [chunk],
        generator,
        tmp_path,
        max_new_questions=1,
        allowed_prompt_fingerprints=(old_fingerprint,),
    )
    assert result["skipped"] == result["generated"] == 1
    assert len(client.calls) == 2
    assert path.read_bytes() == original
    assert (
        batch.export_answer_records(
            questions,
            [chunk],
            generator.model,
            tmp_path,
            tmp_path / "qa.jsonl",
            allowed_prompt_fingerprints=(old_fingerprint,),
        )
        == 2
    )
    with pytest.raises(ValueError, match="incompatible"):
        batch.export_answer_records(
            questions, [chunk], generator.model, tmp_path, tmp_path / "qa.jsonl"
        )


@pytest.mark.parametrize("change", ["model", "source", "question", "evidence"])
def test_allowing_prior_prompt_does_not_bypass_other_checkpoint_checks(
    tmp_path, chunk, question, answer, monkeypatch, change
):
    client = FakeLLM(answer)
    generator = AnswerGenerator(client, "model")
    batch.run_answer_batch([question], [chunk], generator, tmp_path)
    path = next(tmp_path.glob("*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    old_fingerprint = saved["metadata"]["prompt_fingerprint"]
    monkeypatch.setattr(batch, "build_answer_prompt", lambda *args: "new prompt")
    if change == "model":
        generator.model = "new/model"
    elif change == "source":
        chunk["text"] += " Other source text."
    elif change == "question":
        question["category"] = "stylistic"
    else:
        saved["record"]["evidence_quote"] = "Not present in source"
        path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        batch.run_answer_batch(
            [question],
            [chunk],
            generator,
            tmp_path,
            allowed_prompt_fingerprints=(old_fingerprint,),
        )
    assert len(client.calls) == 1


def test_cli_validates_prior_prompt_fingerprint_without_calls(monkeypatch):
    with pytest.raises(SystemExit) as error:
        cli.main(["--allow-prompt-fingerprint", "not-a-hash", "--dry-run"])
    assert error.value.code == 2


def test_retry_failed_only_does_not_attempt_untried_or_completed_questions(
    tmp_path, chunk, questions, answer
):
    client = FakeLLM(answer, ["invalid JSON"])
    generator = AnswerGenerator(client, "model")
    batch.run_answer_batch(questions, [chunk], generator, tmp_path, max_new_questions=2)
    assert len(client.calls) == 2
    # Untried questions now precede the failed one in the input.
    reordered = questions[2:] + questions[:2]
    result = batch.run_answer_batch(
        reordered, [chunk], generator, tmp_path, retry_failed_only=True, dry_run=True
    )
    assert result["planned"] == result["skipped"] == 1
    assert len(client.calls) == 2
    output = tmp_path / "qa.jsonl"
    result = batch.run_answer_batch(
        reordered,
        [chunk],
        generator,
        tmp_path,
        retry_failed_only=True,
        output_path=output,
    )
    assert result["generated"] == 1
    assert len(client.calls) == 3
    assert questions[0]["question"] in client.calls[-1]["prompt"]
    assert len(load_chunks(output)) == 2
    result = batch.run_answer_batch(
        reordered, [chunk], generator, tmp_path, retry_failed_only=True
    )
    assert result["planned"] == 0
    assert len(client.calls) == 3


def test_retry_failed_with_no_log_makes_no_calls(tmp_path, chunk, questions, answer):
    client = FakeLLM(answer)
    result = batch.run_answer_batch(
        questions,
        [chunk],
        AnswerGenerator(client, "model"),
        tmp_path,
        retry_failed_only=True,
        dry_run=True,
    )
    assert result["planned"] == 0
    assert result["remaining"] == len(questions)
    assert not client.calls
    assert not list(tmp_path.iterdir())


def test_scheduler_processes_unattempted_questions_before_historical_failures(
    tmp_path, chunk, question, answer
):
    questions = [{**question, "question": f"Revenue request {i}?"} for i in range(30)]
    client = FakeLLM(answer)
    generator = AnswerGenerator(client, "model")
    output = tmp_path / "qa.jsonl"
    batch.run_answer_batch(
        questions, [chunk], generator, tmp_path, max_new_questions=21
    )
    original_checkpoints = {
        path.name: path.read_bytes() for path in tmp_path.glob("*.json")
    }
    client.outcomes = ["invalid JSON"] * 4
    first_failures = batch.run_answer_batch(
        questions, [chunk], generator, tmp_path, max_new_questions=4
    )
    assert first_failures["failed_pending"] == first_failures["failed"] == 4
    assert first_failures["remaining"] == 9
    assert first_failures["unattempted"] == 5
    assert first_failures["unanswerable"] == 0
    failure_path = tmp_path / "failures.jsonl"
    # Repeated entries and IDs outside the current input must not inflate counts.
    with failure_path.open("a", encoding="utf-8") as file:
        file.write(failure_path.read_text(encoding="utf-8").splitlines()[0] + "\n")
        file.write(json.dumps({"question_id": "unrelated historical failure"}) + "\n")
    original_failures = failure_path.read_bytes()
    calls = len(client.calls)

    dry = batch.run_answer_batch(questions, [chunk], generator, tmp_path, dry_run=True)
    assert dry["planned"] == dry["unattempted"] == 5
    assert dry["skipped"] == dry["completed_total"] == 21
    assert dry["skipped_failed"] == dry["failed_pending"] == 4
    assert dry["remaining"] == 9
    assert dry["failed"] == 0
    assert len(client.calls) == calls

    normal = batch.run_answer_batch(
        questions, [chunk], generator, tmp_path, output_path=output
    )
    assert len(client.calls) == calls + 5
    assert normal["generated"] == normal["attempted"] == 5
    assert normal["completed_total"] == normal["exported"] == 26
    assert (
        normal["remaining"] == normal["skipped_failed"] == normal["failed_pending"] == 4
    )
    assert normal["failed"] == normal["unattempted"] == normal["unanswerable"] == 0
    assert {
        json.loads(call["prompt"].split("INPUT:\n", 1)[1])["question"]
        for call in client.calls[-5:]
    } == {q["question"] for q in questions[25:]}
    assert failure_path.read_bytes() == original_failures
    assert all(
        (tmp_path / name).read_bytes() == contents
        for name, contents in original_checkpoints.items()
    )
    assert all(row["is_answerable"] for row in load_chunks(output))

    exhausted = batch.run_answer_batch(questions, [chunk], generator, tmp_path)
    assert exhausted["planned"] == exhausted["attempted"] == exhausted["generated"] == 0
    assert exhausted["remaining"] == exhausted["skipped_failed"] == 4
    assert len(client.calls) == calls + 5
    retry_dry = batch.run_answer_batch(
        questions, [chunk], generator, tmp_path, retry_failed_only=True, dry_run=True
    )
    assert retry_dry["planned"] == 4
    assert retry_dry["skipped_failed"] == 0

    # A repeated explicit failure remains unresolved and is not made unanswerable.
    client.outcomes = ["invalid JSON"]
    retry_failure = batch.run_answer_batch(
        questions,
        [chunk],
        generator,
        tmp_path,
        max_new_questions=1,
        retry_failed_only=True,
    )
    assert retry_failure["failed"] == 1
    assert retry_failure["failed_pending"] == retry_failure["remaining"] == 4
    assert retry_failure["unanswerable"] == retry_failure["generated"] == 0
    assert retry_failure["unattempted"] == 0
    assert failure_path.read_bytes().startswith(original_failures)

    retry = batch.run_answer_batch(
        questions,
        [chunk],
        generator,
        tmp_path,
        retry_failed_only=True,
        output_path=output,
    )
    assert retry["generated"] == retry["attempted"] == 4
    assert retry["completed_total"] == retry["exported"] == 30
    assert retry["remaining"] == retry["failed_pending"] == retry["unattempted"] == 0
    assert len(load_chunks(output)) == 30
    assert all(
        (tmp_path / name).read_bytes() == contents
        for name, contents in original_checkpoints.items()
    )
    # Successful checkpoints override failure-log history in both scheduling modes.
    for retry_mode in (False, True):
        done = batch.run_answer_batch(
            questions,
            [chunk],
            generator,
            tmp_path,
            retry_failed_only=retry_mode,
            dry_run=True,
        )
        assert done["planned"] == done["failed_pending"] == done["skipped_failed"] == 0
        assert done["completed_total"] == done["skipped"] == 30


def test_cli_normal_run_skips_failures_and_reports_progress(
    tmp_path, chunk, questions, answer, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli, "load_config", lambda name: {"answer_generation": {"model": "model"}}
    )
    save_jsonl(questions, tmp_path / "data/generated/questions.jsonl")
    save_jsonl([chunk], tmp_path / "data/processed/data_factory/uber_2024_chunks.jsonl")
    client = FakeLLM(answer, ["invalid JSON"] * 2)
    monkeypatch.setattr(cli, "LiteLLMClient", lambda **kwargs: client)
    with pytest.raises(SystemExit) as error:
        cli.main(["--max-new-questions", "2"])
    assert error.value.code == 1
    capsys.readouterr()
    cli.main(["--max-new-questions", "2", "--dry-run"])
    assert "Historical failures skipped: 2" in capsys.readouterr().out
    assert len(client.calls) == 2
    cli.main(["--max-new-questions", "2"])
    output = capsys.readouterr().out
    for line in [
        "Historical failures skipped: 2",
        "Completed total: 2",
        "Failed this run: 0",
        "Unattempted: 4",
        "Unresolved failed: 2",
        "Remaining (including failed): 6",
    ]:
        assert line in output
    assert len(client.calls) == 4
    cli.main(["--max-new-questions", "2", "--retry-failed"])
    output = capsys.readouterr().out
    for line in [
        "Historical failures skipped: 0",
        "Completed total: 4",
        "Unattempted: 4",
        "Unresolved failed: 0",
    ]:
        assert line in output
    assert len(client.calls) == 6
