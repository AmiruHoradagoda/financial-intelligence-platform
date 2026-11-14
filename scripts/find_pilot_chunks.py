from financial_intelligence.config import PROJECT_ROOT
from financial_intelligence.infrastructure.jsonl import load_chunks


path = (
    PROJECT_ROOT
    / "data/processed/data_factory/uber_2024_chunks.jsonl"
)

chunks = load_chunks(path)

keywords = [
    "revenue",
    "operating income",
    "consolidated statements of operations",
    "gross bookings",
]

for keyword in keywords:
    print(f"\n=== {keyword.upper()} ===")

    matches = [
        chunk
        for chunk in chunks
        if keyword in chunk["text"].lower()
        and len(chunk["text"].strip()) >= 500
    ]

    for chunk in matches[:5]:
        print(
            f"\nID: {chunk['chunk_id']}"
            f" | Page: {chunk['page_number']}"
        )
        print(repr(chunk["text"][:250]))