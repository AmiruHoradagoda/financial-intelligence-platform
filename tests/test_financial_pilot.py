"""Offline integration of targeted questions with the existing answer pilot CLI."""

import json
from collections import Counter

import pytest

from financial_intelligence.cli import batch_generate_questions as questions_cli
from financial_intelligence.cli import generate_answers as answers_cli
from financial_intelligence.infrastructure.jsonl import load_chunks, save_jsonl
from financial_intelligence.schemas.qa import QAPairRecord

TARGETS = ["uber_2024_data_factory_p57_c1", "uber_2024_data_factory_p18_c0"]


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    config = {
        "documents": {"files": ["uber_2024.pdf"], "output_dir": "data/processed"},
        "data_generation": {"model": "fake/questions"},
        "answer_generation": {"model": "fake/answers"},
    }
    chunks = [
        {
            "chunk_id": TARGETS[1],
            "source_file": "uber_2024.pdf",
            "page_number": 18,
            "text": "Lower Delivery fees may result in a negative Revenue Margin.",
        },
        {
            "chunk_id": "uber_2024_data_factory_p2_c0",
            "source_file": "uber_2024.pdf",
            "page_number": 2,
            "text": "A mission statement.",
        },
        {
            "chunk_id": TARGETS[0],
            "source_file": "uber_2024.pdf",
            "page_number": 57,
            "text": "Revenue in 2024 was $43,978 million.",
        },
    ]
    save_jsonl(chunks, tmp_path / "data/processed/data_factory/uber_2024_chunks.jsonl")
    old_output = tmp_path / "data/generated/questions.jsonl"
    old_qa = tmp_path / "data/generated/qa_pairs_preview.jsonl"
    old_checkpoint = tmp_path / "data/generated/answer_checkpoints/old.json"
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_text('{"previous_checkpoint": true}', encoding="utf-8")
    save_jsonl([{"previous_questions": 30}], old_output)
    save_jsonl([{"previous_qa": 26}], old_qa)

    class QuestionLLM:
        calls = 0

        def complete(self, *, model, prompt):
            assert model == "fake/questions"
            self.calls += 1
            categories = (
                ["hard_fact"] * 4 + ["strategic_summary"] * 3 + ["stylistic"] * 3
            )
            return json.dumps(
                {
                    "questions": [
                        {"category": category, "question": f"Financial question {i}?"}
                        for i, category in enumerate(categories)
                    ]
                }
            )

    class AnswerLLM:
        calls = 0

        def __init__(self):
            self.outcomes = {}

        def complete(self, *, model, prompt):
            assert model == "fake/answers"
            self.calls += 1
            if self.calls in self.outcomes:
                outcome = self.outcomes[self.calls]
                if isinstance(outcome, Exception):
                    raise outcome
                return json.dumps(outcome)
            if self.calls == 1:
                return json.dumps(
                    {"answer": None, "is_answerable": False, "evidence_quote": None}
                )
            source = json.loads(prompt.split("INPUT:\n", 1)[1])["original_chunk"]
            return json.dumps(
                {"answer": source, "is_answerable": True, "evidence_quote": source}
            )

    question_client, answer_client = QuestionLLM(), AnswerLLM()
    for module, client in [
        (questions_cli, question_client),
        (answers_cli, answer_client),
    ]:
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(module, "load_config", lambda name: config)

        def make_client(*, num_retries, client=client):
            assert num_retries == 0
            return client

        monkeypatch.setattr(module, "LiteLLMClient", make_client)
    return (
        tmp_path,
        config,
        chunks,
        question_client,
        answer_client,
        [old_output, old_qa, old_checkpoint],
    )


def test_two_chunk_question_answer_pilot_is_isolated_and_resumable(pilot):
    root, config, chunks, question_client, answer_client, previous_files = pilot
    originals = {path: path.read_bytes() for path in previous_files}
    output = root / "data/generated/questions_financial_pilot.jsonl"
    checkpoints = root / "data/generated/question_checkpoints_financial_pilot"
    question_args = [
        "--chunk-ids",
        *TARGETS,
        "--max-new-chunks",
        "2",
        "--checkpoint-dir",
        str(checkpoints),
        "--output",
        str(output),
    ]
    questions_cli.main([*question_args, "--dry-run"])
    assert question_client.calls == 0
    assert not output.exists() and not checkpoints.exists()
    questions_cli.main(question_args)
    rows = load_chunks(output)
    assert question_client.calls == 2
    assert len(rows) == 20
    assert [row["chunk_id"] for row in rows] == [TARGETS[0]] * 10 + [TARGETS[1]] * 10
    for chunk_id in TARGETS:
        assert Counter(r["category"] for r in rows if r["chunk_id"] == chunk_id) == {
            "hard_fact": 4,
            "strategic_summary": 3,
            "stylistic": 3,
        }
    questions_cli.main(question_args)
    questions_cli.main([*question_args, "--export-only"])
    assert question_client.calls == 2
    assert load_chunks(output) == rows

    answer_output = root / "data/generated/qa_pairs_financial_pilot.jsonl"
    answer_checkpoints = root / "data/generated/answer_checkpoints_financial_pilot"
    answer_args = [
        "--questions",
        str(output),
        "--max-new-questions",
        "20",
        "--checkpoint-dir",
        str(answer_checkpoints),
        "--output",
        str(answer_output),
    ]
    answers_cli.main([*answer_args, "--dry-run"])
    assert answer_client.calls == 0 and not answer_output.exists()
    answers_cli.main(answer_args)
    pairs = load_chunks(answer_output)
    assert answer_client.calls == len(pairs) == 20
    assert sum(not pair["is_answerable"] for pair in pairs) == 1
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    for question, pair in zip(rows, pairs, strict=True):
        QAPairRecord.model_validate(pair)
        assert all(pair[key] == value for key, value in question.items())
        if pair["is_answerable"]:
            assert pair["evidence_quote"] in by_id[pair["chunk_id"]]["text"]
        else:
            assert pair["answer"] is pair["evidence_quote"] is None
    answers_cli.main(answer_args)
    assert answer_client.calls == 20
    assert load_chunks(answer_output) == pairs
    assert all(path.read_bytes() == content for path, content in originals.items())
    for directory, model in [
        (checkpoints, "fake/questions"),
        (answer_checkpoints, "fake/answers"),
    ]:
        for path in directory.glob("*.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
            assert metadata["model"] == model
            assert len(metadata["prompt_fingerprint"]) == 64
    # Chunk selection must not weaken the existing compatibility checks.
    config["data_generation"]["model"] = "changed/model"
    with pytest.raises(SystemExit) as error:
        questions_cli.main(question_args)
    assert error.value.code == 1
    assert question_client.calls == 2


@pytest.mark.parametrize("ids", [[TARGETS[0], TARGETS[0]], ["missing_chunk"]])
def test_bad_chunk_selection_fails_before_calls_or_writes(pilot, ids):
    root, _, _, question_client, _, _ = pilot
    output = root / "data/generated/new_questions.jsonl"
    with pytest.raises(SystemExit) as error:
        questions_cli.main(["--chunk-ids", *ids, "--output", str(output)])
    assert error.value.code == 2
    assert question_client.calls == 0
    assert not output.exists()


@pytest.mark.parametrize(
    "filename",
    [
        "questions.jsonl",
        "questions_preview.jsonl",
        "questions_quality_pilot.jsonl",
        "qa_pairs_preview.jsonl",
    ],
)
def test_targeted_generation_protects_previous_pilot_paths(pilot, filename):
    root, _, _, question_client, _, previous_files = pilot
    before = {path: path.read_bytes() for path in previous_files}
    with pytest.raises(SystemExit) as error:
        questions_cli.main(
            [
                "--chunk-ids",
                *TARGETS,
                "--output",
                str(root / "data/generated" / filename),
            ]
        )
    assert error.value.code == 2
    assert question_client.calls == 0
    assert all(path.read_bytes() == content for path, content in before.items())


def test_pilot_separates_insufficient_evidence_validation_and_api_failures(pilot):
    root, _, _, _, answer_client, _ = pilot
    question_output = root / "data/generated/questions_financial_pilot.jsonl"
    questions_cli.main(
        [
            "--chunk-ids",
            *TARGETS,
            "--max-new-chunks",
            "2",
            "--checkpoint-dir",
            str(root / "question_pilot"),
            "--output",
            str(question_output),
        ]
    )

    class ProviderError(Exception):
        status_code = 401

    answer_client.outcomes = {
        2: {
            "answer": "Unsupported answer",
            "is_answerable": True,
            "evidence_quote": "Not present in either source",
        },
        3: ProviderError("Authentication failed"),
    }
    checkpoints = root / "answer_pilot"
    output = root / "data/generated/qa_pairs_financial_pilot.jsonl"
    with pytest.raises(SystemExit) as error:
        answers_cli.main(
            [
                "--questions",
                str(question_output),
                "--max-new-questions",
                "20",
                "--checkpoint-dir",
                str(checkpoints),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 1
    rows = load_chunks(output)
    assert len(rows) == 18
    assert sum(not row["is_answerable"] for row in rows) == 1
    failures = load_chunks(checkpoints / "failures.jsonl")
    assert [row["failure_kind"] for row in failures] == [
        "validation_failure",
        "api_failure",
    ]
    assert [row["error_type"] for row in failures] == [
        "EvidenceQuoteMismatch",
        "ProviderError",
    ]
    assert answer_client.calls == 20
