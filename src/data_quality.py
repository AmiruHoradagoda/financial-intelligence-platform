
import json
from pathlib import Path


def load_chunks(file_path: Path) -> list[dict]:
    """Load and parse a JSONL file."""

    chunks = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from error

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Expected JSON object at line {line_number}"
                )

            chunks.append(chunk)

    return chunks


def validate_chunk_offsets(chunk: dict, page_text: str) -> bool:
    """Check whether a chunk matches its original page text."""

    start = chunk.get("start_char")
    end = chunk.get("end_char")
    text = chunk.get("text")

    if type(start) is not int or type(end) is not int:
        return False

    if not isinstance(text, str):
        return False

    if not (0 <= start < end <= len(page_text)):
        return False

    return page_text[start:end] == text

def find_duplicate_chunk_ids(chunks: list[dict]) -> list[str]:

    duplicates = {}
    chunks_set = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id in chunks_set:
            duplicates[chunk_id] = None
        else:
            chunks_set.add(chunk_id)

    return list(duplicates.keys())