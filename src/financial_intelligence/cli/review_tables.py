"""Run the review_tables workflow."""

import csv
from pathlib import Path

from financial_intelligence.config import PROJECT_ROOT, load_config
from financial_intelligence.infrastructure.jsonl import load_chunks
from financial_intelligence.ingestion.table_quality import select_review_pages


def main() -> None:
    config = load_config("data")

    processed_dir = PROJECT_ROOT / config["documents"]["output_dir"]

    review_rows = []

    # Process every configured PDF
    for document_name in config["documents"]["files"]:
        document_id = Path(document_name).stem

        chunks_path = processed_dir / "data_factory" / f"{document_id}_chunks.jsonl"

        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")

        chunks = load_chunks(chunks_path)

        samples = select_review_pages(chunks, top_n=5, random_n=3)

        for page, reason, score in samples:
            review_rows.append(
                {
                    "document": document_name,
                    "page_number": page,
                    "selection_reason": reason,
                    "numeric_tokens": score,
                    "values_correct": "",
                    "table_structure_correct": "",
                    "notes": "",
                }
            )

        print(f"{document_name}: {len(samples)} pages selected")

    # Save combined review sheet
    output_path = PROJECT_ROOT / "data/evaluation/table_quality_review.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "document",
                "page_number",
                "selection_reason",
                "numeric_tokens",
                "values_correct",
                "table_structure_correct",
                "notes",
            ],
        )

        writer.writeheader()
        writer.writerows(review_rows)

    print(f"Total review pages: {len(review_rows)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
