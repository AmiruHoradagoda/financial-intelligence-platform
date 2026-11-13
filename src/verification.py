from pathlib import Path

from src.data_quality import load_chunks
from src.config_loader import PROJECT_ROOT


def show_extracted_page(document: str, page: int):

    document_id = Path(document).stem

    path = (
        PROJECT_ROOT
        / "data/processed/data_factory"
        / f"{document_id}_chunks.jsonl"
    )

    chunks = load_chunks(path)

    page_chunks = sorted(
        [
            chunk for chunk in chunks
            if chunk["page_number"] == page
        ],
        key=lambda c: c["start_char"]
    )

    text = "".join(chunk["text"] for chunk in page_chunks)

    print(text)

show_extracted_page("uber_2023.pdf", 42)