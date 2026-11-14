from pathlib import Path

from financial_intelligence.config import PROJECT_ROOT
from financial_intelligence.infrastructure.jsonl import load_chunks


def show_extracted_page(document: str, page: int) -> None:

    document_id = Path(document).stem

    path = PROJECT_ROOT / "data/processed/data_factory" / f"{document_id}_chunks.jsonl"

    chunks = load_chunks(path)

    page_chunks = sorted(
        [chunk for chunk in chunks if chunk["page_number"] == page],
        key=lambda c: c["start_char"],
    )

    text = "".join(chunk["text"] for chunk in page_chunks)

    print(text)


if __name__ == "__main__":
    show_extracted_page("uber_2024.pdf", 42)
