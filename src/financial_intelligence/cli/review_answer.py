"""Append an explicit offline decision without rewriting a checkpoint or registry."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from financial_intelligence.config import PROJECT_ROOT
from financial_intelligence.data_factory.answer_batch import (
    _question_id,
    _read_checkpoint,
)
from financial_intelligence.data_factory.answer_generator import (
    AnswerabilityReviewRequired,
)
from financial_intelligence.infrastructure.jsonl import load_chunks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--decision", choices=("approve", "reject"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data/generated/questions.jsonl",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "data/processed/data_factory/uber_2024_chunks.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.reviewer.strip() or not args.notes.strip():
        parser.error("Reviewer and notes must be nonblank")
    try:
        by_id = {_question_id(q): q for q in load_chunks(args.questions)}
        if args.question_id not in by_id:
            raise ValueError("Unknown question ID")
        question = by_id[args.question_id]
        chunk = next(
            c for c in load_chunks(args.chunks) if c["chunk_id"] == question["chunk_id"]
        )
        # Use the matched generated hash, never an arbitrary supplied filesystem path.
        path = args.checkpoint_dir / f"{_question_id(question)}.json"
        raw = path.read_bytes()
        saved = json.loads(raw)
        metadata = saved["metadata"]
        try:
            _read_checkpoint(
                path,
                question,
                chunk,
                metadata["model"],
                (metadata["prompt_fingerprint"],),
            )
        except AnswerabilityReviewRequired as error:
            if args.decision == "approve" and error.approval_blocked:
                raise ValueError(
                    f"Cannot approve a structural/source contradiction: {error.reason}"
                ) from error
        decision = {
            "question_id": args.question_id,
            "decision": args.decision,
            "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
            "reviewer": args.reviewer.strip(),
            "notes": args.notes.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if not args.dry_run:
            with (args.checkpoint_dir / "review_decisions.jsonl").open(
                "a", encoding="utf-8"
            ) as file:
                file.write(json.dumps(decision, ensure_ascii=False) + "\n")
        print(
            f"{'Dry run' if args.dry_run else 'Recorded'}: {args.decision} {args.question_id}; checkpoint unchanged"
        )
    except (ValueError, OSError, KeyError, StopIteration) as error:
        parser.exit(1, f"Review decision stopped: {error}\n")


if __name__ == "__main__":
    main()
