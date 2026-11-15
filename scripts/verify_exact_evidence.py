
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


chunks = read_jsonl(
    "data/processed/data_factory/uber_2024_chunks.jsonl"
)

answers = read_jsonl(
    "data/generated/qa_pairs_preview.jsonl"
)

sources = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}

failed = 0

for index, item in enumerate(answers, start=1):
    if item["is_answerable"]:
        valid = item["evidence_quote"] in sources[item["chunk_id"]]
    else:
        valid = (
            item["answer"] is None
            and item["evidence_quote"] is None
        )

    if not valid:
        failed += 1

    print(f"Question {index}: {'PASS' if valid else 'FAIL'}")

print(f"\nTotal: {len(answers)}, Failed: {failed}")
