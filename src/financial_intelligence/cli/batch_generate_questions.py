"""Generate a bounded resumable batch; dry runs never call the provider."""

import argparse
from pathlib import Path

from financial_intelligence.config import PROJECT_ROOT, load_config
from financial_intelligence.data_factory.batch import (
    export_completed_records,
    run_question_batch,
)
from financial_intelligence.data_factory.question_generator import QuestionGenerator
from financial_intelligence.infrastructure.jsonl import load_chunks
from financial_intelligence.infrastructure.llm_client import LiteLLMClient


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-chunks", type=int, default=5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--export-only",
        action="store_true",
        help="Export compatible checkpoints without API calls",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "data/generated/question_checkpoints",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/generated/questions.jsonl"
    )
    args = parser.parse_args(argv)
    if args.max_new_chunks < 1:
        parser.error("--max-new-chunks must be positive")

    config = load_config("data")
    document_id = Path(config["documents"]["files"][0]).stem
    chunks_path = (
        PROJECT_ROOT
        / config["documents"]["output_dir"]
        / "data_factory"
        / f"{document_id}_chunks.jsonl"
    )
    chunks = load_chunks(chunks_path)
    generator = QuestionGenerator(
        LiteLLMClient(num_retries=0), config.get("data_generation", {}).get("model")
    )
    try:
        if args.export_only and not args.dry_run:
            count = export_completed_records(
                chunks, generator.model, args.checkpoint_dir, args.output
            )
            print(f"Exported: {count} questions to {args.output}")
            return
        result = run_question_batch(
            chunks,
            generator,
            args.checkpoint_dir,
            max_new_chunks=args.max_new_chunks,
            dry_run=args.dry_run,
            output_path=args.output,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Batch stopped: {error}\n")
    print(
        f"{'Dry run planned' if args.dry_run else 'Attempted'}: "
        f"{result['planned'] if args.dry_run else result['attempted']} chunks"
    )
    print(f"Processed successfully: {result['generated']}")
    print(f"Skipped existing: {result['skipped']}")
    print(f"Failed: {result['failed']}")
    print(f"Excluded new: {result['excluded']}")
    print(f"Skipped previously excluded: {result['skipped_excluded']}")
    print(f"Remaining (including retryable failures): {result['remaining']}")
    for exclusion in result["exclusions"]:
        print(f"Excluded: {exclusion['chunk_id']} — {exclusion['reason']}")
    if not args.dry_run:
        print(f"Exported: {result['exported']} questions to {args.output}")
    if result["failed"]:
        parser.exit(1, f"See {args.checkpoint_dir / 'failures.jsonl'}\n")


if __name__ == "__main__":
    main()
