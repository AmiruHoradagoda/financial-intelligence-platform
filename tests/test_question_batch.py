"""Offline batch, persistence, retry, and CLI contracts."""

import json
from pathlib import Path

import pytest

from financial_intelligence.cli import batch_generate_questions as cli
from financial_intelligence.data_factory import batch
from financial_intelligence.data_factory.question_generator import QuestionGenerator
from financial_intelligence.infrastructure.jsonl import load_chunks


@pytest.fixture
def chunks():
    return [
        {
            "chunk_id": f"c{i}",
            "source_file": "uber_2024.pdf",
            "page_number": i + 1,
            "text": f"Report excerpt {i}",
        }
        for i in range(8)
    ]


@pytest.fixture
def questions():
    categories = ["hard_fact"] * 4 + ["strategic_summary"] * 3 + ["stylistic"] * 3
    return [
        {"category": category, "question": f"Question {i}?"}
        for i, category in enumerate(categories)
    ]


class FakeLLM:
    def __init__(self, questions, outcomes=()):
        self.content = json.dumps({"questions": questions})
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, *, model, prompt):
        self.calls.append((model, prompt))
        outcome = self.outcomes.pop(0) if self.outcomes else self.content
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_incremental_resume_and_export(tmp_path, chunks, questions):
    client = FakeLLM(questions)
    generator = QuestionGenerator(client, "fake/model")
    checkpoints = tmp_path / "checkpoints"
    output = tmp_path / "questions.jsonl"
    first = batch.run_question_batch(chunks, generator, checkpoints, output_path=output)
    assert first["generated"] == first["attempted"] == 5
    assert first["exported"] == 50
    original = (checkpoints / "c0.json").read_bytes()
    saved = json.loads(original)
    assert saved["metadata"]["model"] == "fake/model"
    assert len(saved["metadata"]["prompt_fingerprint"]) == 64
    assert saved["questions"] == questions

    second = batch.run_question_batch(
        chunks, generator, checkpoints, output_path=output
    )
    assert second["skipped"] == 5
    assert second["generated"] == 3
    assert len(client.calls) == 8
    assert (checkpoints / "c0.json").read_bytes() == original
    rows = load_chunks(output)
    assert len(rows) == 80
    assert rows[0] == {
        **{key: chunks[0][key] for key in ("chunk_id", "source_file", "page_number")},
        **questions[0],
    }
    assert len({(row["chunk_id"], row["question"]) for row in rows}) == 80
    assert not list(checkpoints.glob("*.tmp"))
    assert (
        batch.export_completed_records(chunks, generator.model, checkpoints, output)
        == 80
    )
    assert load_chunks(output) == rows


def test_dry_run_has_no_side_effects(tmp_path, chunks, questions):
    client = FakeLLM(questions)
    result = batch.run_question_batch(
        chunks,
        QuestionGenerator(client, "fake/model"),
        tmp_path / "checkpoints",
        dry_run=True,
        output_path=tmp_path / "output.jsonl",
    )
    assert result["planned"] == 5
    assert result["attempted"] == result["generated"] == 0
    assert client.calls == []
    assert list(tmp_path.iterdir()) == []


class APIError(Exception):
    def __init__(self, status):
        self.status_code = status
        super().__init__(f"API status {status}")


@pytest.mark.parametrize(
    "error",
    [TimeoutError(), ConnectionError(), APIError(429), APIError(408), APIError(503)],
)
def test_transient_retry_backoff(tmp_path, chunks, questions, monkeypatch, error):
    delays = []
    monkeypatch.setattr(batch.time, "sleep", delays.append)
    client = FakeLLM(questions, [error, error])
    result = batch.run_question_batch(
        chunks, QuestionGenerator(client, "fake/model"), tmp_path, max_new_chunks=1
    )
    assert result["generated"] == 1
    assert len(client.calls) == 3
    assert delays == [1, 2]
    assert not (tmp_path / "failures.jsonl").exists()


def test_retry_exhaustion_continues_within_budget(
    tmp_path, chunks, questions, monkeypatch
):
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    client = FakeLLM(questions, [APIError(503)] * 3)
    generator = QuestionGenerator(client, "fake/model")
    result = batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=2)
    assert result["attempted"] == 2
    assert result["failed"] == result["generated"] == 1
    assert len(client.calls) == 4
    assert not (tmp_path / "c0.json").exists()
    assert (tmp_path / "c1.json").exists()
    failure = load_chunks(tmp_path / "failures.jsonl")[0]
    assert failure["chunk_id"] == "c0"
    assert failure["attempts"] == 3
    assert failure["error_type"] == "APIError"
    resumed = batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    assert resumed["skipped"] == 1
    assert resumed["generated"] == 1
    assert (tmp_path / "c0.json").exists()


@pytest.mark.parametrize(
    "response",
    [
        "bad JSON",
        '{"questions": []}',
        '{"questions": [{"category": "hard_fact", "question": "Only one?"}]}',
        APIError(401),
        APIError(400),
    ],
)
def test_invalid_or_permanent_errors_never_retry(tmp_path, chunks, questions, response):
    client = FakeLLM(questions, [response])
    output = tmp_path / "questions.jsonl"
    result = batch.run_question_batch(
        chunks,
        QuestionGenerator(client, "fake/model"),
        tmp_path,
        max_new_chunks=2,
        output_path=output,
    )
    assert result["failed"] == result["generated"] == 1
    assert len(client.calls) == 2
    assert not (tmp_path / "c0.json").exists()
    assert len(load_chunks(output)) == 10
    assert {row["chunk_id"] for row in load_chunks(output)} == {"c1"}


@pytest.mark.parametrize(
    "change",
    ["model", "prompt", "text", "source", "invalid", "legacy", "corrupt", "id"],
)
def test_checkpoint_incompatibility_stops_before_spending(
    tmp_path, chunks, questions, monkeypatch, change
):
    client = FakeLLM(questions)
    generator = QuestionGenerator(client, "fake/model")
    batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    path = tmp_path / "c0.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    if change == "model":
        generator.model = "different/model"
    elif change == "prompt":
        monkeypatch.setattr(batch, "build_question_prompt", lambda text: "new prompt")
    elif change == "text":
        chunks[0]["text"] += " changed"
    elif change == "source":
        saved["page_number"] = 999
    elif change == "invalid":
        saved["questions"] = []
    elif change == "legacy":
        del saved["metadata"]
    elif change == "id":
        saved["chunk_id"] = "other"
    path.write_text("{" if change == "corrupt" else json.dumps(saved), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        batch.run_question_batch(chunks, generator, tmp_path)
    assert len(client.calls) == 1
    assert path.read_bytes() == before


def test_interrupted_checkpoint_preserves_successes(
    tmp_path, chunks, questions, monkeypatch
):
    client = FakeLLM(questions)
    generator = QuestionGenerator(client, "fake/model")
    batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    original = (tmp_path / "c0.json").read_bytes()
    replace = Path.replace

    def interrupted(path, target):
        if target.name == "c1.json":
            # The complete temp file exists; final checkpoint has not appeared.
            assert len(json.loads(path.read_text(encoding="utf-8"))["questions"]) == 10
            assert not target.exists()
            raise OSError("interrupted replace")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    assert (tmp_path / "c0.json").read_bytes() == original
    assert not (tmp_path / "c1.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(Path, "replace", replace)
    result = batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    assert result["skipped"] == result["generated"] == 1


def test_invalid_export_preserves_previous_output(tmp_path, chunks, questions):
    generator = QuestionGenerator(FakeLLM(questions), "fake/model")
    batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    output = tmp_path / "questions.jsonl"
    output.write_text("previous export", encoding="utf-8")
    (tmp_path / "c0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        batch.export_completed_records(chunks, generator.model, tmp_path, output)
    assert output.read_text(encoding="utf-8") == "previous export"


@pytest.mark.parametrize("chunk_id", ["../escape", "c/1", "", ".."])
def test_unsafe_chunk_ids_rejected(tmp_path, chunks, questions, chunk_id):
    chunks[0]["chunk_id"] = chunk_id
    client = FakeLLM(questions)
    with pytest.raises(ValueError, match="Invalid chunk ID"):
        batch.run_question_batch(
            chunks, QuestionGenerator(client, "fake/model"), tmp_path
        )
    assert not client.calls


def test_duplicate_ids_rejected_before_spending(tmp_path, chunks, questions):
    client = FakeLLM(questions)
    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        batch.run_question_batch(
            [chunks[0], chunks[0]], QuestionGenerator(client, "fake/model"), tmp_path
        )
    assert not client.calls


def test_cli_defaults_dry_run_export_and_resume(
    tmp_path, chunks, questions, monkeypatch
):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda name: {
            "documents": {"files": ["uber_2024.pdf"], "output_dir": "data/processed"},
            "data_generation": {"model": "fake/model"},
        },
    )
    monkeypatch.setattr(cli, "load_chunks", lambda path: chunks)
    client = FakeLLM(questions)

    def make_client(**kwargs):
        assert kwargs == {"num_retries": 0}
        return client

    monkeypatch.setattr(cli, "LiteLLMClient", make_client)
    cli.main(["--dry-run"])
    assert not client.calls
    assert list(tmp_path.iterdir()) == []
    cli.main([])
    assert len(client.calls) == 5
    output = tmp_path / "data/generated/questions.jsonl"
    assert len(load_chunks(output)) == 50
    cli.main(["--export-only"])
    assert len(client.calls) == 5
    cli.main(["--max-new-chunks", "1"])
    assert len(client.calls) == 6
    assert len(load_chunks(output)) == 60
    with pytest.raises(SystemExit) as error:
        cli.main(["--max-new-chunks", "0"])
    assert error.value.code == 2


def test_cli_failures_exit_nonzero_after_export(
    tmp_path, chunks, questions, monkeypatch
):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda name: {
            "documents": {"files": ["uber_2024.pdf"], "output_dir": "data/processed"},
            "data_generation": {"model": "fake/model"},
        },
    )
    monkeypatch.setattr(cli, "load_chunks", lambda path: chunks)
    client = FakeLLM(questions, ["invalid"])
    monkeypatch.setattr(cli, "LiteLLMClient", lambda **kwargs: client)
    with pytest.raises(SystemExit) as error:
        cli.main(["--max-new-chunks", "2"])
    assert error.value.code == 1
    assert len(load_chunks(tmp_path / "data/generated/questions.jsonl")) == 10


def test_batch_boundary_rejects_unvalidated_generator_output(tmp_path, chunks):
    class InvalidGenerator:
        model = "fake/model"

        def generate(self, chunk):
            return [{"category": "hard_fact", "question": "Incomplete batch?"}]

    result = batch.run_question_batch(
        chunks, InvalidGenerator(), tmp_path, max_new_chunks=1
    )
    assert result["failed"] == 1
    assert result["generated"] == 0
    assert not (tmp_path / "c0.json").exists()
    assert (
        load_chunks(tmp_path / "failures.jsonl")[0]["error_type"] == "ValidationError"
    )


def test_atomic_export_failure_keeps_previous_export(
    tmp_path, chunks, questions, monkeypatch
):
    generator = QuestionGenerator(FakeLLM(questions), "fake/model")
    batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    output = tmp_path / "questions.jsonl"
    batch.export_completed_records(chunks, generator.model, tmp_path, output)
    original = output.read_bytes()
    batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    replace = Path.replace

    def interrupted(path, target):
        if target == output:
            raise OSError("export interrupted")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="export interrupted"):
        batch.export_completed_records(chunks, generator.model, tmp_path, output)
    assert output.read_bytes() == original
    assert (tmp_path / "c1.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "options", [{"max_new_chunks": 0}, {"max_attempts": 0}, {"max_attempts": 4}]
)
def test_invalid_limits_rejected_before_spending(tmp_path, chunks, questions, options):
    client = FakeLLM(questions)
    with pytest.raises(ValueError):
        batch.run_question_batch(
            chunks, QuestionGenerator(client, "fake/model"), tmp_path, **options
        )
    assert not client.calls


def test_exclusions_are_checkpointed_resumed_and_not_exported(
    tmp_path, chunks, questions
):
    reason = "Promotional fragments cannot support ten distinct business questions"
    client = FakeLLM(questions, [json.dumps({"unprocessable_reason": reason})])
    generator = QuestionGenerator(client, "fake/model")
    output = tmp_path / "questions.jsonl"
    result = batch.run_question_batch(
        chunks, generator, tmp_path, max_new_chunks=2, output_path=output
    )
    assert result["excluded"] == result["generated"] == 1
    assert result["failed"] == 0
    assert result["remaining"] == 6
    assert result["exclusions"] == [{"chunk_id": "c0", "reason": reason}]
    saved = json.loads((tmp_path / "c0.json").read_text(encoding="utf-8"))
    assert saved["status"] == "excluded"
    assert saved["reason"] == reason
    assert "questions" not in saved
    assert saved["metadata"]["model"] == "fake/model"
    assert not (tmp_path / "failures.jsonl").exists()
    assert len(client.calls) == 2
    assert len(load_chunks(output)) == 10
    assert {row["chunk_id"] for row in load_chunks(output)} == {"c1"}
    assert (
        batch.export_completed_records(chunks, generator.model, tmp_path, output) == 10
    )

    dry = batch.run_question_batch(chunks, generator, tmp_path, dry_run=True)
    assert dry["skipped"] == dry["skipped_excluded"] == 1
    assert dry["remaining"] == 6
    assert dry["planned"] == 5
    assert dry["exclusions"] == result["exclusions"]
    assert len(client.calls) == 2

    resumed = batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    assert resumed["skipped_excluded"] == resumed["skipped"] == 1
    assert resumed["generated"] == 1
    assert resumed["remaining"] == 5
    assert len(client.calls) == 3


@pytest.mark.parametrize("change", ["prompt", "model", "reason", "questions", "status"])
def test_excluded_checkpoint_validation(
    tmp_path, chunks, questions, monkeypatch, change
):
    client = FakeLLM(questions, ['{"unprocessable_reason": "Too little evidence"}'])
    generator = QuestionGenerator(client, "fake/model")
    batch.run_question_batch(chunks, generator, tmp_path, max_new_chunks=1)
    path = tmp_path / "c0.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    if change == "prompt":
        monkeypatch.setattr(
            batch, "build_question_prompt", lambda text: "revised prompt"
        )
    elif change == "model":
        generator.model = "another/model"
    elif change == "questions":
        saved["questions"] = questions
    elif change == "status":
        saved["status"] = "unknown"
    else:
        saved["reason"] = " "
    path.write_text(json.dumps(saved), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        batch.run_question_batch(chunks, generator, tmp_path)
    assert len(client.calls) == 1
    assert path.read_bytes() == before


def test_empty_chunk_is_excluded_and_later_chunks_continue(tmp_path, chunks, questions):
    chunks[0]["text"] = " \n\t"
    client = FakeLLM(questions)
    result = batch.run_question_batch(
        chunks, QuestionGenerator(client, "fake/model"), tmp_path, max_new_chunks=2
    )
    assert result["excluded"] == result["generated"] == 1
    assert result["attempted"] == 2
    assert result["remaining"] == 6
    assert len(client.calls) == 1


def test_reporting_counts_include_failed_chunks_in_remaining(
    tmp_path, chunks, questions
):
    client = FakeLLM(questions, ["invalid JSON"])
    result = batch.run_question_batch(
        chunks, QuestionGenerator(client, "fake/model"), tmp_path, max_new_chunks=2
    )
    assert result["failed"] == result["generated"] == 1
    assert result["remaining"] == 7
    assert result["total"] == (
        result["skipped"]
        + result["skipped_excluded"]
        + result["generated"]
        + result["excluded"]
        + result["remaining"]
    )


def test_cli_reports_exclusion_reason_and_remaining(
    tmp_path, chunks, questions, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda name: {
            "documents": {"files": ["uber_2024.pdf"], "output_dir": "data/processed"},
            "data_generation": {"model": "fake/model"},
        },
    )
    monkeypatch.setattr(cli, "load_chunks", lambda path: chunks)
    client = FakeLLM(
        questions, ['{"unprocessable_reason": "Only a truncated heading"}']
    )
    monkeypatch.setattr(cli, "LiteLLMClient", lambda **kwargs: client)
    cli.main(["--max-new-chunks", "2"])
    output = capsys.readouterr().out
    for expected in [
        "Processed successfully: 1",
        "Skipped existing: 0",
        "Failed: 0",
        "Excluded new: 1",
        "Remaining (including retryable failures): 6",
        "c0",
        "Only a truncated heading",
    ]:
        assert expected in output
    assert len(client.calls) == 2
