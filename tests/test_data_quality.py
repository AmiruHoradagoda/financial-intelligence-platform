import json

from src.data_quality import (
    load_chunks,
    validate_chunk_offsets,
    find_duplicate_chunk_ids
)


def test_load_chunks(tmp_path):
    path = tmp_path / "chunks.jsonl"

    records = [
        {"chunk_id": "c1", "text": "Revenue"},
        {"chunk_id": "c2", "text": "Profit"},
    ]

    path.write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )

    assert load_chunks(path) == records


def test_validate_chunk_offsets():
    page_text = "Uber Revenue increased"

    valid_chunk = {
        "start_char": 5,
        "end_char": 12,
        "text": "Revenue",
    }

    assert validate_chunk_offsets(valid_chunk, page_text)

    invalid_chunk = {
        "start_char": -1,
        "end_char": 12,
        "text": "Revenue",
    }

    assert not validate_chunk_offsets(invalid_chunk, page_text)





def test_duplicate_ids():
    chunks = [
        {"chunk_id": "c1"},
        {"chunk_id": "c2"},
        {"chunk_id": "c1"},
        {"chunk_id": "c3"},
        {"chunk_id": "c2"},
    ]

    assert find_duplicate_chunk_ids(chunks) == ["c1", "c2"]


def test_no_duplicates():
    chunks = [
        {"chunk_id": "c1"},
        {"chunk_id": "c2"},
    ]

    assert find_duplicate_chunk_ids(chunks) == []


def test_multiple_occurrences():
    chunks = [
        {"chunk_id": "c1"},
        {"chunk_id": "c1"},
        {"chunk_id": "c1"},
    ]

    assert find_duplicate_chunk_ids(chunks) == ["c1"]


def test_empty_chunks():
    assert find_duplicate_chunk_ids([]) == []