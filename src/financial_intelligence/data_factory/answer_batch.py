"""Bounded answer pilot with per-question checkpoints and atomic JSONL export."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from financial_intelligence.data_factory.answer_generator import (
    AnswerGenerator,
    EvidenceQuoteMismatch,
    match_question_source,
    validate_grounded_answer,
)
from financial_intelligence.data_factory.prompts import build_answer_prompt
from financial_intelligence.infrastructure.batch_utils import (
    atomic_write,
    is_transient_error,
)
from financial_intelligence.infrastructure.jsonl import load_chunks
from financial_intelligence.schemas.qa import AnswerResponse, QAPairRecord


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _question_id(question: dict) -> str:
    # Stable across file reordering; repeated wording in different chunks is distinct.
    identity = [question["chunk_id"], " ".join(question["question"].casefold().split())]
    return _fingerprint(json.dumps(identity, ensure_ascii=False))


def _metadata(question: dict, chunk: dict, model: str) -> dict:
    return {
        "version": 1,
        "model": model,
        "prompt_fingerprint": _fingerprint(build_answer_prompt("", "", "")),
        "text_fingerprint": _fingerprint(chunk["text"]),
        "question_fingerprint": _fingerprint(
            json.dumps(question, ensure_ascii=False, sort_keys=True)
        ),
    }


def _prepare(questions: list[dict], chunks: list[dict]) -> tuple[list, int]:
    """Validate the join and deduplicate questions before any spending or writes."""
    by_chunk = {}
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Invalid source chunk ID")
        if chunk_id in by_chunk:
            raise ValueError(f"Duplicate source chunk ID: {chunk_id}")
        by_chunk[chunk_id] = chunk
    seen = {}
    prepared = []
    duplicates = 0
    for question in questions:
        chunk = by_chunk.get(question.get("chunk_id"))
        if chunk is None:
            raise ValueError(
                f"No original chunk for question: {question.get('chunk_id')}"
            )
        question = match_question_source(question, chunk).model_dump()
        question_id = _question_id(question)
        if question_id in seen:
            if seen[question_id] != question:
                raise ValueError(f"Conflicting duplicate question: {question_id}")
            duplicates += 1
            continue
        seen[question_id] = question
        prepared.append((question_id, question, chunk))
    return prepared, duplicates


def _read_checkpoint(
    path: Path,
    question: dict,
    chunk: dict,
    model: str,
    allowed_prompt_fingerprints: tuple[str, ...] = (),
) -> dict:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["question_id"] != _question_id(question):
            raise ValueError("question identity changed")
        metadata = dict(saved["metadata"])
        expected = _metadata(question, chunk, model)
        prior_prompt = metadata.get("prompt_fingerprint")
        allowed_prior_prompt = prior_prompt != expected["prompt_fingerprint"] and (
            prior_prompt in allowed_prompt_fingerprints
        )
        if allowed_prior_prompt:
            metadata["prompt_fingerprint"] = expected["prompt_fingerprint"]
        if metadata != expected:
            raise ValueError("model, prompt, question, or source changed")
        record = QAPairRecord.model_validate(saved["record"]).model_dump()
        if any(record[key] != value for key, value in question.items()):
            raise ValueError("original question metadata changed")
        validate_grounded_answer(
            {key: record[key] for key in AnswerResponse.model_fields}, chunk["text"]
        )
        if allowed_prior_prompt:
            print(
                f"Reused validated prior-prompt checkpoint: {path.name} ({prior_prompt})"
            )
        return record
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid or incompatible answer checkpoint {path}. "
            "Archive it or use a separate answer checkpoint directory."
        ) from error


def _export(prepared: list, completed: dict, output_path: Path) -> int:
    content = "".join(
        json.dumps(completed[question_id], ensure_ascii=False) + "\n"
        for question_id, _, _ in prepared
        if question_id in completed
    )
    atomic_write(output_path, content)
    return len(completed)


def export_answer_records(
    questions: list[dict],
    chunks: list[dict],
    model: str,
    checkpoint_dir: Path,
    output_path: Path,
    *,
    allowed_prompt_fingerprints: tuple[str, ...] = (),
) -> int:
    """Rebuild an export from compatible, revalidated checkpoints without API calls."""
    prepared, _ = _prepare(questions, chunks)
    completed = {}
    for question_id, question, chunk in prepared:
        path = checkpoint_dir / f"{question_id}.json"
        if path.exists():
            completed[question_id] = _read_checkpoint(
                path, question, chunk, model, allowed_prompt_fingerprints
            )
    return _export(prepared, completed, output_path)


def run_answer_batch(
    questions: list[dict],
    chunks: list[dict],
    generator: AnswerGenerator,
    checkpoint_dir: Path,
    max_new_questions: int = 5,
    *,
    dry_run: bool = False,
    max_attempts: int = 3,
    output_path: Path | None = None,
    allowed_prompt_fingerprints: tuple[str, ...] = (),
    retry_failed_only: bool = False,
) -> dict:
    """Attempt at most N unfinished questions; valid unanswerable results complete too.

    Every relevant checkpoint is validated before calls. Normal runs select only
    unattempted questions; retry mode selects only historical failures. Failures
    stay unresolved until explicitly retried. Use one writer per directory.
    """
    if max_new_questions < 1:
        raise ValueError("max_new_questions must be positive")
    if not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")
    prepared, duplicates = _prepare(questions, chunks)
    pending = []
    completed = {}
    for question_id, question, chunk in prepared:
        path = checkpoint_dir / f"{question_id}.json"
        if path.exists():
            completed[question_id] = _read_checkpoint(
                path, question, chunk, generator.model, allowed_prompt_fingerprints
            )
        else:
            pending.append((question_id, question, chunk, path))
    failure_path = checkpoint_dir / "failures.jsonl"
    failures = load_chunks(failure_path) if failure_path.exists() else []
    failed_ids = {
        row["question_id"]
        for row in failures
        if isinstance(row.get("question_id"), str)
    }
    failed_pending = sum(entry[0] in failed_ids for entry in pending)
    candidates = [
        entry for entry in pending if (entry[0] in failed_ids) == retry_failed_only
    ]
    result = {
        "generated": 0,
        "skipped": len(completed),
        "duplicates": duplicates,
        "failed": 0,
        "attempted": 0,
        "planned": min(len(candidates), max_new_questions),
        "remaining": len(pending),
        "skipped_failed": 0 if retry_failed_only else failed_pending,
        "failed_pending": failed_pending,
        "unattempted": len(pending) - failed_pending,
        "completed_total": len(completed),
        "unanswerable": 0,
        "exported": 0,
    }
    if dry_run:
        return result

    for question_id, question, chunk, path in candidates[:max_new_questions]:
        result["attempted"] += 1
        if question_id not in failed_ids:
            result["unattempted"] -= 1
        for attempt in range(1, max_attempts + 1):
            try:
                answer = validate_grounded_answer(
                    generator.generate(question, chunk), chunk["text"]
                )
                record = QAPairRecord.model_validate(
                    {**question, **answer.model_dump()}
                ).model_dump()
            except Exception as error:
                if is_transient_error(error) and attempt < max_attempts:
                    time.sleep(2 ** (attempt - 1))
                    continue
                failure = {
                    "question_id": question_id,
                    **question,
                    "metadata": _metadata(question, chunk, generator.model),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attempts": attempt,
                    "error_type": type(error).__name__,
                    # Provider and validation exception strings may contain
                    # credentials, raw responses, or full prompts/documents.
                    "error": (
                        "Evidence quote is not an exact substring of the source"
                        if isinstance(error, EvidenceQuoteMismatch)
                        else "Answer generation or validation failed"
                    ),
                }
                if isinstance(error, EvidenceQuoteMismatch):
                    failure["evidence_diagnostics"] = error.diagnostics
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                with (checkpoint_dir / "failures.jsonl").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                result["failed"] += 1
                if question_id not in failed_ids:
                    result["failed_pending"] += 1
                    failed_ids.add(question_id)
                print(f"Failed: {question_id} ({type(error).__name__})")
                break
            else:
                atomic_write(
                    path,
                    json.dumps(
                        {
                            "question_id": question_id,
                            "metadata": _metadata(question, chunk, generator.model),
                            "record": record,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
                completed[question_id] = record
                result["generated"] += 1
                result["completed_total"] += 1
                if question_id in failed_ids:
                    result["failed_pending"] -= 1
                result["remaining"] -= 1
                result["unanswerable"] += int(not record["is_answerable"])
                print(f"Completed: {question_id}")
                break
    if output_path is not None:
        result["exported"] = _export(prepared, completed, output_path)
    return result
