import importlib

import pymupdf
import pytest
from pydantic import ValidationError

from financial_intelligence.infrastructure.jsonl import load_chunks, save_jsonl
from financial_intelligence.ingestion.chunking import (
    create_chunks,
    fixed_chunking,
    sliding_chunking,
)
from financial_intelligence.ingestion.pdf_reader import clean_text, extract_pdf
from financial_intelligence.ingestion.quality import (
    audit_document,
    validate_chunk_offsets,
)
from financial_intelligence.ingestion.table_quality import select_review_pages
from financial_intelligence.schemas.chunk import Chunk


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_blank_chunking(text):
    assert fixed_chunking(text, 10) == []
    assert sliding_chunking(text, 10, 3) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10)])
def test_invalid_chunk_settings(size, overlap):
    with pytest.raises(ValueError):
        sliding_chunking("Example", size, overlap)


def test_fixed_size_and_overlap_constraints():
    with pytest.raises(ValueError):
        fixed_chunking("Text", 0)
    with pytest.raises(ValueError):
        create_chunks("Text", "fixed", 10, 1)


def test_sliding_whitespace_boundary_and_coverage():
    text = "one two three four five six seven eight nine ten " * 5
    chunks = sliding_chunking(text, 30, 5)
    assert chunks[0]["text"].endswith(" ")
    assert chunks[0]["end_char"] < 30
    assert chunks[-1]["end_char"] == len(text)
    for previous, current in zip(chunks, chunks[1:]):
        assert current["start_char"] == previous["end_char"] - 5
    records = [
        dict(chunk, chunk_id=f"c{i}", page_number=1) for i, chunk in enumerate(chunks)
    ]
    assert audit_document(records, {1: text}, max_chunk_size=30)["status"] == "PASS"


def test_large_overlap_still_advances():
    text = "a " * 30
    chunks = sliding_chunking(text, 10, 9)
    assert chunks[-1]["end_char"] == len(text)
    assert all(a["start_char"] < b["start_char"] for a, b in zip(chunks, chunks[1:]))


def test_clean_text_retains_line_breaks():
    assert clean_text("  Revenue  \r\n\r\n\r\nProfit\t\r") == "Revenue\n\nProfit"


def test_pdf_extraction_metadata_offsets_and_empty_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Revenue increased. Profit increased.")
        pdf.new_page()
        pdf.save(pdf_path)
    settings = {"strategy": "fixed", "chunk_size": 15, "chunk_overlap": 0}
    records, empty = extract_pdf(pdf_path, "data_factory", settings)
    assert empty == [2]
    assert [r["chunk_id"] for r in records] == [
        f"sample_data_factory_p1_c{i}" for i in range(len(records))
    ]
    text = "".join(r["text"] for r in records)
    for record in records:
        assert record["document_id"] == "sample"
        assert record["source_file"] == "sample.pdf"
        assert record["pipeline"] == "data_factory"
        assert record["page_number"] == 1
        assert validate_chunk_offsets(record, text)
        assert Chunk.model_validate(record).model_dump() == record
    with pytest.raises(ValidationError):
        Chunk.model_validate({**records[0], "end_char": 1000})


@pytest.mark.parametrize(
    "record",
    [
        {"start_char": True, "end_char": 2, "text": "b"},
        {"start_char": 0, "end_char": 2, "text": "x"},
    ],
)
def test_bad_offsets(record):
    assert not validate_chunk_offsets(record, "ab")


def test_audit_reports_invalid_rows_and_gaps():
    records = [
        {"chunk_id": "", "page_number": 1, "start_char": 0, "end_char": 1, "text": "A"},
        {"chunk_id": "c1", "page_number": True, "text": "x"},
        {
            "chunk_id": "c2",
            "page_number": 1,
            "start_char": -1,
            "end_char": 2,
            "text": "x",
        },
    ]
    report = audit_document(records, {1: "A B", 2: "\n "})
    assert report == {
        "total_chunks": 3,
        "duplicate_ids": [],
        "invalid_ids": [0],
        "invalid_pages": [1],
        "invalid_offsets": [2],
        "oversized_chunks": [],
        "coverage_gaps": [{"page_number": 1, "start": 1, "end": 3}],
        "status": "FAIL",
    }


@pytest.mark.parametrize("line", ['{"bad":', "[]"])
def test_jsonl_rejects_malformed_rows(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + line, encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_chunks(path)


def test_jsonl_round_trip_unicode_and_blank_lines(tmp_path):
    path = tmp_path / "nested/chunks.jsonl"
    records = [{"text": "Revenue £10 — café"}]
    save_jsonl(records, path)
    assert "café" in path.read_text(encoding="utf-8")
    assert load_chunks(path) == records
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert load_chunks(path) == records


def test_table_sampling_deterministic():
    records = [{"page_number": page, "text": "100 " * page} for page in range(1, 11)]
    first = select_review_pages(records)
    assert first == select_review_pages(records)
    assert [page for page, reason, score in first[:5]] == [10, 9, 8, 7, 6]
    assert len(first) == len({page for page, _, _ in first}) == 8
    assert select_review_pages([]) == []


@pytest.mark.parametrize(
    "module",
    [
        "cli.ingest",
        "cli.audit_data",
        "cli.review_tables",
        "ingestion.verification",
    ],
)
def test_package_modules_import_without_execution(module, capsys):
    assert importlib.import_module(f"financial_intelligence.{module}") is not None
    assert capsys.readouterr().out == ""
