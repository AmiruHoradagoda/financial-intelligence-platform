"""PDF text extraction and metadata-rich chunk assembly."""

import re
from pathlib import Path

import pymupdf

from financial_intelligence.schemas.chunk import Chunk

from .chunking import create_chunks


def clean_text(text: str) -> str:
    """Normalize basic whitespace while retaining line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pdf(
    pdf_path: Path,
    pipeline_name: str,
    settings: dict,
) -> tuple[list[dict], list[int]]:
    """Extract page text and create metadata-rich chunks."""

    records = []
    empty_pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            text = clean_text(page.get_text("text", sort=True))

            if not text:
                empty_pages.append(page_number)
                continue

            chunks = create_chunks(
                text=text,
                strategy=settings["strategy"],
                size=int(settings["chunk_size"]),
                overlap=int(settings["chunk_overlap"]),
            )

            for chunk_index, chunk in enumerate(chunks):
                records.append(
                    Chunk.model_validate(
                        {
                            "chunk_id": (
                                f"{pdf_path.stem}_{pipeline_name}_"
                                f"p{page_number}_c{chunk_index}"
                            ),
                            "document_id": pdf_path.stem,
                            "source_file": pdf_path.name,
                            "pipeline": pipeline_name,
                            "strategy": settings["strategy"],
                            "page_number": page_number,
                            "start_char": chunk["start_char"],
                            "end_char": chunk["end_char"],
                            "text": chunk["text"],
                        }
                    ).model_dump()
                )

    return records, empty_pages
