"""JSONL file boundaries shared by ingestion and generation."""

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
                raise ValueError(f"Invalid JSON at line {line_number}") from error

            if not isinstance(chunk, dict):
                raise ValueError(f"Expected JSON object at line {line_number}")

            chunks.append(chunk)

    return chunks


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Persist chunks to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
