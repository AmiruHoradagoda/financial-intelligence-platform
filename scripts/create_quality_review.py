
import csv

from financial_intelligence.config import PROJECT_ROOT
from financial_intelligence.infrastructure.jsonl import load_chunks


questions_path = (
    PROJECT_ROOT / "data/generated/questions_quality_pilot.jsonl"
)

chunks_path = (
    PROJECT_ROOT
    / "data/processed/data_factory/uber_2024_chunks.jsonl"
)

output_path = (
    PROJECT_ROOT / "data/evaluation/question_quality_review.csv"
)


def main():
    # Prevent accidentally overwriting human review work.
    if output_path.exists():
        raise FileExistsError(
            f"Review file already exists: {output_path}"
        )

    questions = load_chunks(questions_path)
    chunks = load_chunks(chunks_path)

    # Efficient lookup by chunk ID.
    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    rows = []

    for index, question in enumerate(questions, start=1):
        chunk_id = question["chunk_id"]

        if chunk_id not in chunks_by_id:
            raise ValueError(f"Missing source chunk: {chunk_id}")

        source = chunks_by_id[chunk_id]

        rows.append({
            "question_id": f"pilot_{index:03d}",
            "chunk_id": chunk_id,
            "page_number": question["page_number"],
            "category": question["category"],
            "question": question["question"],
            "source_text": source["text"],
            "answerable": "",
            "category_correct": "",
            "useful": "",
            "non_redundant": "",
            "evidence_quote": "",
            "verdict": "",
            "notes": "",
        })

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Questions: {len(rows)}")
    print(f"Review file: {output_path}")


if __name__ == "__main__":
    main()
