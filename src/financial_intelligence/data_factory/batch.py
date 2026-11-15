"""Bounded question generation with validated, atomic per-chunk checkpoints."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from financial_intelligence.data_factory.prompts import build_question_prompt
from financial_intelligence.data_factory.question_generator import (
    InsufficientInformation,
    QuestionGenerator,
)
from financial_intelligence.infrastructure.batch_utils import (
    atomic_write as _atomic_write,
)
from financial_intelligence.infrastructure.batch_utils import (
    is_transient_error as _transient,
)
from financial_intelligence.schemas.qa import QuestionBatch, QuestionRecord


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metadata(chunk: dict, model: str) -> dict:
    return {
        "version": 1,
        "model": model,
        "prompt_fingerprint": _fingerprint(build_question_prompt("")),
        "text_fingerprint": _fingerprint(chunk["text"]),
    }


def _source(chunk: dict) -> dict:
    return {key: chunk[key] for key in ("chunk_id", "source_file", "page_number")}


def _checkpoint_path(directory: Path, chunk: dict) -> Path:
    chunk_id = chunk["chunk_id"]
    if not isinstance(chunk_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", chunk_id):
        raise ValueError("Invalid chunk ID")
    # Validate source metadata before spending on generation.
    if not isinstance(chunk["source_file"], str) or not chunk["source_file"].strip():
        raise ValueError(f"Invalid source file: {chunk_id}")
    if type(chunk["page_number"]) is not int or chunk["page_number"] < 1:
        raise ValueError(f"Invalid page number: {chunk_id}")
    if not isinstance(chunk["text"], str):
        raise ValueError(f"Invalid source text: {chunk_id}")
    return directory / f"{chunk_id}.json"


def _read_checkpoint(path: Path, chunk: dict, model: str) -> dict:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["metadata"] != _metadata(chunk, model):
            raise ValueError("model, prompt, or source text changed")
        if _source(saved) != _source(chunk):
            raise ValueError("source metadata changed")
        if saved.get("status") == "excluded":
            if (
                not isinstance(saved.get("reason"), str)
                or not saved["reason"].strip()
                or "questions" in saved
            ):
                raise ValueError("invalid exclusion reason or unexpected questions")
            return saved
        if saved.get("status", "completed") != "completed":
            raise ValueError("unknown checkpoint status")
        QuestionBatch.model_validate({"questions": saved["questions"]})
        for question in saved["questions"]:
            QuestionRecord.model_validate({**_source(saved), **question})
        return saved
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid or incompatible checkpoint {path}. "
            "Use a separate checkpoint directory or archive this checkpoint "
            "before regenerating."
        ) from error


def export_completed_records(
    chunks: list[dict], model: str, checkpoint_dir: Path, output_path: Path
) -> int:
    """Atomically export compatible checkpoints in source chunk order."""
    rows = []
    seen = set()
    for chunk in chunks:
        path = _checkpoint_path(checkpoint_dir, chunk)
        if chunk["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk ID: {chunk['chunk_id']}")
        seen.add(chunk["chunk_id"])
        if not path.exists():
            continue
        saved = _read_checkpoint(path, chunk, model)
        if saved.get("status") == "excluded":
            continue
        for question in saved["questions"]:
            row = {**_source(saved), **question}
            QuestionRecord.model_validate(row)
            rows.append(json.dumps(row, ensure_ascii=False) + "\n")
    _atomic_write(output_path, "".join(rows))
    return len(rows)


def run_question_batch(
    chunks: list[dict],
    generator: QuestionGenerator,
    checkpoint_dir: Path,
    max_new_chunks: int = 5,
    *,
    dry_run: bool = False,
    max_attempts: int = 3,
    output_path: Path | None = None,
) -> dict:
    """Attempt at most N unfinished chunks; retry only transient API failures.

    Existing checkpoints are checked before any API calls. Invalid or incompatible
    checkpoints stop the run without overwriting them. Run one writer per directory.
    """
    if max_new_chunks < 1:
        raise ValueError("max_new_chunks must be positive")
    if not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")

    pending = []
    seen = set()
    skipped = 0
    skipped_excluded = 0
    exclusions = []
    for chunk in chunks:
        path = _checkpoint_path(checkpoint_dir, chunk)
        if chunk["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk ID: {chunk['chunk_id']}")
        seen.add(chunk["chunk_id"])
        if path.exists():
            saved = _read_checkpoint(path, chunk, generator.model)
            if saved.get("status") == "excluded":
                skipped_excluded += 1
                exclusions.append(
                    {"chunk_id": chunk["chunk_id"], "reason": saved["reason"]}
                )
            else:
                skipped += 1
        else:
            pending.append((chunk, path))

    selected = pending[:max_new_chunks]
    result = {
        "generated": 0,
        "skipped": skipped,
        "failed": 0,
        "planned": len(selected),
        "attempted": 0,
        "exported": 0,
        "excluded": 0,
        "skipped_excluded": skipped_excluded,
        "exclusions": exclusions,
        "remaining": len(pending),
        "total": len(chunks),
    }
    if dry_run:
        return result

    for chunk, path in selected:
        result["attempted"] += 1
        for attempt in range(1, max_attempts + 1):
            try:
                questions = generator.generate(chunk)
                QuestionBatch.model_validate({"questions": questions})
            except InsufficientInformation as error:
                record = {
                    **_source(chunk),
                    "metadata": _metadata(chunk, generator.model),
                    "status": "excluded",
                    "reason": str(error),
                }
                _atomic_write(path, json.dumps(record, ensure_ascii=False, indent=2))
                result["excluded"] += 1
                result["exclusions"].append(
                    {"chunk_id": chunk["chunk_id"], "reason": str(error)}
                )
                result["remaining"] -= 1
                break
            except Exception as error:
                if _transient(error) and attempt < max_attempts:
                    time.sleep(2 ** (attempt - 1))
                    continue
                failure = {
                    **_source(chunk),
                    "metadata": _metadata(chunk, generator.model),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attempts": attempt,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                with (checkpoint_dir / "failures.jsonl").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                result["failed"] += 1
                print(f"Failed: {chunk['chunk_id']} ({type(error).__name__})")
                break
            else:
                record = {
                    **_source(chunk),
                    "metadata": _metadata(chunk, generator.model),
                    "questions": questions,
                }
                # Storage errors stop the run; never report an unsaved success.
                _atomic_write(path, json.dumps(record, ensure_ascii=False, indent=2))
                result["generated"] += 1
                result["remaining"] -= 1
                print(f"Completed: {chunk['chunk_id']}")
                break

    if output_path is not None:
        result["exported"] = export_completed_records(
            chunks, generator.model, checkpoint_dir, output_path
        )
    return result
