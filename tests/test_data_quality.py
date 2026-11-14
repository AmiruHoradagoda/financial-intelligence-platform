import json

from financial_intelligence.ingestion.quality import (
    audit_document,
    find_duplicate_chunk_ids,
    load_chunks,
    validate_chunk_offsets,
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


def test_valid_document():
    page_text = "Uber Revenue"

    chunks = [
        {
            "chunk_id": "c1",
            "page_number": 1,
            "start_char": 0,
            "end_char": len(page_text),
            "text": page_text,
        }
    ]

    report = audit_document(chunks, {1: page_text})

    assert report["status"] == "PASS"
    assert report["duplicate_ids"] == []
    assert report["coverage_gaps"] == []


def test_duplicate_document_chunks():
    page_text = "Uber Revenue"

    chunks = [
        {
            "chunk_id": "c1",
            "page_number": 1,
            "start_char": 0,
            "end_char": 4,
            "text": "Uber",
        },
        {
            "chunk_id": "c1",
            "page_number": 1,
            "start_char": 5,
            "end_char": 12,
            "text": "Revenue",
        },
    ]

    report = audit_document(chunks, {1: page_text})

    assert report["status"] == "FAIL"
    assert report["duplicate_ids"] == ["c1"]


def test_oversized_chunk_detection():
    page_text = "A" * 1501
    max_chunk_size = 1500
    chunks = [
        {
            "chunk_id": "c1",
            "page_number": 1,
            "start_char": 0,
            "end_char": len(page_text),
            "text": page_text,
        }
    ]

    report = audit_document(chunks, {1: page_text}, max_chunk_size)

    assert report["status"] == "FAIL"
    assert report["oversized_chunks"] == ["c1"]
