
import json
import re
from pathlib import Path

from financial_intelligence.data_factory.question_generator import (
    QuestionGenerator,
)
from financial_intelligence.schemas.qa import QuestionBatch


def run_question_batch(
    chunks: list[dict],
    generator: QuestionGenerator,
    checkpoint_dir: Path,
    max_new_chunks: int = 2,
) -> dict:
    """Generate and checkpoint a limited number of new chunks."""

    if max_new_chunks < 1:
        raise ValueError("max_new_chunks must be positive")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        if not isinstance(chunk_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", chunk_id
        ):
            raise ValueError("Invalid chunk ID")

        checkpoint_path = checkpoint_dir / f"{chunk_id}.json"

        # Resume: validate previously completed work.
        if checkpoint_path.exists():
            saved = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )

            if saved["chunk_id"] != chunk_id:
                raise ValueError("Checkpoint ID mismatch")

            QuestionBatch.model_validate({
                "questions": saved["questions"]
            })

            skipped += 1
            continue

        # Cost protection: limit NEW LLM calls.
        if generated >= max_new_chunks:
            break

        questions = generator.generate(chunk)

        # Defensive validation at the batch boundary.
        QuestionBatch.model_validate({
            "questions": questions
        })

        record = {
            "chunk_id": chunk_id,
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "questions": questions,
        }

        # Write first, then replace the final checkpoint.
        temp_path = checkpoint_path.with_suffix(".json.tmp")

        temp_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        temp_path.replace(checkpoint_path)

        generated += 1
        print(f"Completed: {chunk_id}")

    return {
        "generated": generated,
        "skipped": skipped,
    }
