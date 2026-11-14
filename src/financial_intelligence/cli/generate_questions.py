"""Generate the existing two-chunk question preview (makes paid API calls)."""

import json

from financial_intelligence.config import PROJECT_ROOT, load_config
from financial_intelligence.data_factory.question_generator import QuestionGenerator
from financial_intelligence.infrastructure.jsonl import load_chunks
from financial_intelligence.infrastructure.llm_client import LiteLLMClient
from financial_intelligence.schemas.qa import QuestionRecord


def main() -> None:
    config = load_config("data")
    generator = QuestionGenerator(
        LiteLLMClient(), config.get("data_generation", {}).get("model")
    )

    chunks_path = (
        PROJECT_ROOT
        / config["documents"]["output_dir"]
        / "data_factory"
        / "uber_2024_chunks.jsonl"
    )

    chunks = load_chunks(chunks_path)

    # Preview only: skip very short chunks.
    sample_chunks = [chunk for chunk in chunks if len(chunk["text"].strip()) >= 500][:2]

    output_path = PROJECT_ROOT / "data/generated/questions_preview.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in sample_chunks:
            questions = generator.generate(chunk)

            for item in questions:
                record = {
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                    "category": item["category"],
                    "question": item["question"],
                }

                QuestionRecord.model_validate(record)

                file.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"{chunk['chunk_id']}: {len(questions)} questions")

    print(f"Preview saved: {output_path}")


if __name__ == "__main__":
    main()
