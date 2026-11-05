
import json
import re
from pathlib import Path

import pymupdf

from src.config_loader import PROJECT_ROOT, load_config


def clean_text(text: str) -> str:
    """Normalize basic whitespace while retaining line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fixed_chunking(text: str, size: int) -> list[dict]:
    """Split text into non-overlapping fixed-size chunks."""
    if size <= 0:
        raise ValueError("Chunk size must be positive")

    chunks = []

    for start in range(0, len(text), size):
        end = min(start + size, len(text))

        if text[start:end].strip():
            chunks.append({
                "text": text[start:end],
                "start_char": start,
                "end_char": end,
            })

    return chunks


def sliding_chunking(
    text: str,
    size: int,
    overlap: int,
) -> list[dict]:
    """Create overlapping chunks, preferring whitespace boundaries."""

    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid size or overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        # Prefer a natural split near the end of the chunk.
        if end < len(text):
            search_start = start + int(size * 0.7)

            if search_start < end:
                boundary = max(
                    text.rfind("\n", search_start, end),
                    text.rfind(" ", search_start, end),
                )

                if boundary >= search_start:
                    end = boundary + 1

        if text[start:end].strip():
            chunks.append({
                "text": text[start:end],
                "start_char": start,
                "end_char": end,
            })

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def create_chunks(
    text: str,
    strategy: str,
    size: int,
    overlap: int,
) -> list[dict]:
    """Select a chunking strategy from configuration."""

    if strategy == "fixed":
        if overlap != 0:
            raise ValueError("Fixed chunking requires zero overlap")

        return fixed_chunking(text, size)

    if strategy == "sliding":
        return sliding_chunking(text, size, overlap)

    raise ValueError(f"Unsupported chunking strategy: {strategy}")


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

            text = clean_text(
                page.get_text("text", sort=True)
            )

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
                records.append({
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
                })

    return records, empty_pages


def save_jsonl(records: list[dict], output_path: Path):
    """Persist chunks to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    config = load_config("data")

    documents = config["documents"]
    chunking_configs = config["chunking"]

    input_dir = PROJECT_ROOT / documents["input_dir"]
    output_dir = PROJECT_ROOT / documents["output_dir"]

    for filename in documents["files"]:
        pdf_path = input_dir / filename

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        for pipeline_name, settings in chunking_configs.items():

            records, empty_pages = extract_pdf(
                pdf_path=pdf_path,
                pipeline_name=pipeline_name,
                settings=settings,
            )

            output_path = (
                output_dir
                / pipeline_name
                / f"{pdf_path.stem}_chunks.jsonl"
            )

            save_jsonl(records, output_path)

            print(f"\nPipeline: {pipeline_name}")
            print(f"Strategy: {settings['strategy']}")
            print(f"Generated chunks: {len(records)}")
            print(f"Empty pages: {empty_pages[:20]}")
            print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
