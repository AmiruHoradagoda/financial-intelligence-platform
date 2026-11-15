from financial_intelligence.config import PROJECT_ROOT
from financial_intelligence.infrastructure.jsonl import load_chunks

chunks = load_chunks(
    PROJECT_ROOT
    / "data/processed/data_factory/uber_2024_chunks.jsonl"
)

answers = load_chunks(
    PROJECT_ROOT / "data/generated/qa_pairs_preview.jsonl"
)

source_by_id = {
    chunk["chunk_id"]: chunk["text"]
    for chunk in chunks
}

for index, record in enumerate(answers, start=1):
    evidence = record["evidence_quote"]
    source = source_by_id[record["chunk_id"]]

    if record["is_answerable"]:
        valid = isinstance(evidence, str) and evidence in source
    else:
        valid = evidence is None and record["answer"] is None

    print(f"Question {index}: {'PASS' if valid else 'FAIL'}")