
import argparse

from financial_intelligence.config import PROJECT_ROOT, load_config
from financial_intelligence.data_factory.batch import run_question_batch
from financial_intelligence.data_factory.question_generator import (
    QuestionGenerator,
)
from financial_intelligence.infrastructure.jsonl import load_chunks
from financial_intelligence.infrastructure.llm_client import LiteLLMClient


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-new-chunks",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    config = load_config("data")

    document = config["documents"]["files"][0]
    document_id = document.removesuffix(".pdf")

    chunks_path = (
        PROJECT_ROOT
        / config["documents"]["output_dir"]
        / "data_factory"
        / f"{document_id}_chunks.jsonl"
    )

    chunks = load_chunks(chunks_path)

    generator = QuestionGenerator(
        LiteLLMClient(),
        config["data_generation"]["model"],
    )

    result = run_question_batch(
        chunks=chunks,
        generator=generator,
        checkpoint_dir=(
            PROJECT_ROOT
            / "data/generated/question_checkpoints"
        ),
        max_new_chunks=args.max_new_chunks,
    )

    print(f"Generated: {result['generated']}")
    print(f"Skipped existing: {result['skipped']}")


if __name__ == "__main__":
    main()
