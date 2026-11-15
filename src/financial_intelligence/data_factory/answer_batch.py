"""Bounded answer pilot with per-question checkpoints and atomic JSONL export."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from financial_intelligence.data_factory.answer_generator import (
    AnswerabilityReviewRequired,
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
from financial_intelligence.infrastructure.llm_client import InvalidLLMResponse
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


def _compatible_metadata(actual, expected, allowed_prompt_fingerprints) -> bool:
    actual = dict(actual)
    if actual.get("prompt_fingerprint") in allowed_prompt_fingerprints:
        actual["prompt_fingerprint"] = expected["prompt_fingerprint"]
    return actual == expected


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
        answer_data = {key: record.get(key) for key in AnswerResponse.model_fields}
        validate_grounded_answer(answer_data, chunk["text"])
        validate_grounded_answer(
            answer_data, chunk["text"], question=question["question"]
        )
        if saved.get("review_required"):
            raise AnswerabilityReviewRequired(
                saved["review_reason"], answer=answer_data
            )
        if allowed_prior_prompt:
            print(
                f"Reused validated prior-prompt checkpoint: {path.name} ({prior_prompt})"
            )
        return record
    except AnswerabilityReviewRequired as error:
        # A valid schema/fingerprint can still need semantic review. Preserve the
        # file and report that distinction instead of claiming metadata corruption.
        raise AnswerabilityReviewRequired(
            error.reason,
            checkpoint=path.name,
            answer=error.answer,
            approval_blocked=error.approval_blocked,
        ) from error
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


def _review_entry(question_id, question, chunk, model, error) -> dict:
    answer = validate_grounded_answer(error.answer, chunk["text"])
    return {
        "question_id": question_id,
        "metadata": _metadata(question, chunk, model),
        "record": QAPairRecord.model_validate(
            {**question, **answer.model_dump()}
        ).model_dump(),
        "review_required": True,
        "review_status": "required",
        "review_reason": error.reason,
        "approval_blocked": error.approval_blocked,
    }


def _collect_checkpoints(prepared, model, checkpoint_dir, allowed_prompt_fingerprints):
    review_path = checkpoint_dir / "review_required.jsonl"
    registry = (
        {row["question_id"]: row for row in load_chunks(review_path)}
        if review_path.exists()
        else {}
    )
    decision_path = checkpoint_dir / "review_decisions.jsonl"
    decisions = (
        {r["question_id"]: r for r in load_chunks(decision_path)}
        if decision_path.exists()
        else {}
    )
    completed, reviews, pending = {}, {}, []
    for question_id, question, chunk in prepared:
        path = checkpoint_dir / f"{question_id}.json"
        if not path.exists():
            pending.append((question_id, question, chunk, path))
            continue
        try:
            record = _read_checkpoint(
                path, question, chunk, model, allowed_prompt_fingerprints
            )
        except AnswerabilityReviewRequired as error:
            current_review = _review_entry(question_id, question, chunk, model, error)
            record = current_review["record"]
        else:
            current_review = None
        decision = decisions.get(question_id)
        if decision:
            if (
                decision.get("checkpoint_sha256")
                != hashlib.sha256(path.read_bytes()).hexdigest()
                or decision.get("decision") not in ("approve", "reject")
                or not all(
                    isinstance(decision.get(k), str) and decision[k].strip()
                    for k in ("reviewer", "notes")
                )
            ):
                raise ValueError(f"Invalid or stale review decision: {question_id}")
            if decision["decision"] == "approve" and not (
                current_review and current_review.get("approval_blocked")
            ):
                completed[question_id] = record
                print(f"Reused offline-approved record: {question_id}")
                continue
            reviews[question_id] = {
                **(
                    current_review
                    or registry.get(question_id)
                    or {
                        "question_id": question_id,
                        "metadata": _metadata(question, chunk, model),
                        "record": record,
                        "review_reason": "offline_source_review",
                    }
                ),
                "review_status": "rejected"
                if decision["decision"] == "reject"
                else "required",
            }
            continue
        if question_id in registry:
            saved_review = registry[question_id]
            if (
                not _compatible_metadata(
                    saved_review["metadata"],
                    _metadata(question, chunk, model),
                    allowed_prompt_fingerprints,
                )
                or saved_review["record"] != record
            ):
                raise ValueError(
                    f"Review record changed: {question_id}; explicit review required"
                )
            if saved_review.get("review_status") == "approved" and not (
                current_review and current_review.get("approval_blocked")
            ):
                if not all(
                    isinstance(saved_review.get(key), str) and saved_review[key].strip()
                    for key in ("reviewed_by", "review_notes")
                ):
                    raise ValueError(
                        f"Approval requires reviewer and notes: {question_id}"
                    )
                completed[question_id] = record
                print(f"Reused explicitly approved review record: {question_id}")
            else:
                reviews[question_id] = saved_review
        elif current_review is not None:
            reviews[question_id] = current_review
        else:
            completed[question_id] = record
    return completed, reviews, pending


def _persist_reviews(checkpoint_dir: Path, reviews: dict) -> None:
    if not reviews:
        return
    path = checkpoint_dir / "review_required.jsonl"
    existing = {r["question_id"]: r for r in load_chunks(path)} if path.exists() else {}
    # Preserve review entries for questions outside this invocation's input.
    existing.update(reviews)
    atomic_write(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in existing.values()
        ),
    )


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
    completed, reviews, _ = _collect_checkpoints(
        prepared, model, checkpoint_dir, allowed_prompt_fingerprints
    )
    _persist_reviews(checkpoint_dir, reviews)
    for question_id, review in reviews.items():
        print(
            f"Excluded from training export for review: {question_id} ({review['review_reason']})"
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
    completed, reviews, pending = _collect_checkpoints(
        prepared, generator.model, checkpoint_dir, allowed_prompt_fingerprints
    )
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
        "remaining": len(pending) + len(reviews),
        "review_required": len(reviews),
        "review_new": 0,
        "rejected": sum(
            row.get("review_status") == "rejected" for row in reviews.values()
        ),
        "review_records": [
            {"question_id": key, "reason": row["review_reason"]}
            for key, row in reviews.items()
        ],
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
                    generator.generate(question, chunk),
                    chunk["text"],
                    question=question["question"],
                )
                record = QAPairRecord.model_validate(
                    {**question, **answer.model_dump()}
                ).model_dump()
            except AnswerabilityReviewRequired as error:
                review = _review_entry(
                    question_id, question, chunk, generator.model, error
                )
                atomic_write(path, json.dumps(review, ensure_ascii=False, indent=2))
                reviews[question_id] = review
                result["review_required"] += 1
                result["review_new"] += 1
                result["review_records"].append(
                    {"question_id": question_id, "reason": error.reason}
                )
                if question_id in failed_ids:
                    result["failed_pending"] -= 1
                print(f"Held for review: {question_id} ({error.reason})")
                break
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
                    "failure_kind": (
                        "validation_failure"
                        if isinstance(error, (InvalidLLMResponse, ValidationError))
                        else "api_failure"
                        if is_transient_error(error)
                        or isinstance(getattr(error, "status_code", None), int)
                        else "generation_failure"
                    ),
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
    _persist_reviews(checkpoint_dir, reviews)
    if output_path is not None:
        result["exported"] = _export(prepared, completed, output_path)
    return result
