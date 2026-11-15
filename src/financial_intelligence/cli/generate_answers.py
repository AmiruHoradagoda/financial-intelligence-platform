"""Generate a small grounded-answer pilot using a separately configured LLM B."""

import argparse
from pathlib import Path

from financial_intelligence.config import PROJECT_ROOT, load_config
from financial_intelligence.data_factory.answer_batch import (
    export_answer_records,
    run_answer_batch,
)
from financial_intelligence.data_factory.answer_generator import AnswerGenerator
from financial_intelligence.infrastructure.jsonl import load_chunks
from financial_intelligence.infrastructure.llm_client import LiteLLMClient


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-questions", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Attempt only unfinished questions listed in the checkpoint failure log",
    )
    parser.add_argument(
        "--allow-prompt-fingerprint",
        action="append",
        default=[],
        metavar="SHA256",
        help="Explicitly reuse revalidated checkpoints from this prior prompt only",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--export-only", action="store_true")
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "data/generated/answer_checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/generated/qa_pairs_preview.jsonl",
    )
    args = parser.parse_args(argv)
    if args.max_new_questions < 1:
        parser.error("--max-new-questions must be positive")
    if args.retry_failed and args.export_only:
        parser.error("--retry-failed cannot be combined with --export-only")
    for fingerprint in args.allow_prompt_fingerprint:
        if len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            parser.error("--allow-prompt-fingerprint requires a lowercase SHA-256 hash")
    if args.output.resolve() in {args.questions.resolve(), args.chunks.resolve()}:
        parser.error("--output must not overwrite either input file")

    try:
        config = load_config("data")
        generator = AnswerGenerator(
            LiteLLMClient(num_retries=0),
            config.get("answer_generation", {}).get("model"),
        )
        questions = load_chunks(args.questions)
        chunks = load_chunks(args.chunks)
        if args.export_only:
            count = export_answer_records(
                questions,
                chunks,
                generator.model,
                args.checkpoint_dir,
                args.output,
                allowed_prompt_fingerprints=tuple(args.allow_prompt_fingerprint),
            )
            print(f"Exported: {count} Q&A records to {args.output}")
            return
        result = run_answer_batch(
            questions,
            chunks,
            generator,
            args.checkpoint_dir,
            max_new_questions=args.max_new_questions,
            max_attempts=args.max_attempts,
            dry_run=args.dry_run,
            output_path=args.output,
            allowed_prompt_fingerprints=tuple(args.allow_prompt_fingerprint),
            retry_failed_only=args.retry_failed,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Answer batch stopped: {error}\n")
    print(
        f"{'Dry run planned' if args.dry_run else 'Attempted'}: "
        f"{result['planned'] if args.dry_run else result['attempted']} questions"
    )
    print(f"Completed new: {result['generated']}")
    print(f"Unanswerable new: {result['unanswerable']}")
    print(f"Skipped completed: {result['skipped']}")
    print(f"Historical failures skipped: {result['skipped_failed']}")
    print(f"Duplicate questions skipped: {result['duplicates']}")
    print(f"Failed this run: {result['failed']}")
    print(f"Completed total: {result['completed_total']}")
    print(f"Review required: {result['review_required']}")
    print(f"Rejected by source review: {result['rejected']}")
    for review in result["review_records"]:
        print(f"Review: {review['question_id']} ({review['reason']})")
    print(f"Remaining (including failed/review): {result['remaining']}")
    print(f"Unattempted: {result['unattempted']}")
    print(f"Unresolved failed: {result['failed_pending']}")
    if not args.dry_run:
        print(f"Exported: {result['exported']} Q&A records to {args.output}")
    if result["failed"]:
        parser.exit(1, f"See {args.checkpoint_dir / 'failures.jsonl'}\n")


if __name__ == "__main__":
    main()
