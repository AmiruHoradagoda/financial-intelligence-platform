import importlib
import json

import pymupdf

from financial_intelligence.cli import (
    audit_data,
    ingest,
    review_tables,
)
from financial_intelligence.config import PROJECT_ROOT, find_project_root, load_config
from financial_intelligence.infrastructure.jsonl import load_chunks


def test_workflows_write_existing_formats_in_isolated_directory(tmp_path, monkeypatch):
    config = {
        "data_generation": {"model": "fake/model"},
        "documents": {
            "input_dir": "data/raw",
            "output_dir": "data/processed",
            "files": ["uber_2024.pdf"],
        },
        "chunking": {
            "data_factory": {
                "strategy": "fixed",
                "chunk_size": 1500,
                "chunk_overlap": 0,
            },
            "rag": {"strategy": "sliding", "chunk_size": 700, "chunk_overlap": 100},
        },
    }
    pdf_path = tmp_path / "data/raw/uber_2024.pdf"
    pdf_path.parent.mkdir(parents=True)
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for line in range(25):
            page.insert_text((50, 50 + line * 20), "Revenue grew by 10 percent. " * 3)
        pdf.save(pdf_path)
    for module in (ingest, audit_data, review_tables):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(module, "load_config", lambda name: config)

    ingest.main()
    audit_data.main()
    review_tables.main()
    for pipeline in ("data_factory", "rag"):
        chunks = load_chunks(
            tmp_path / f"data/processed/{pipeline}/uber_2024_chunks.jsonl"
        )
        assert chunks
        report = json.loads(
            (tmp_path / f"data/evaluation/{pipeline}_audit.json").read_text()
        )
        assert report["status"] == "PASS"
        assert report["total_chunks"] == len(chunks)
    review = (tmp_path / "data/evaluation/table_quality_review.csv").read_text(
        encoding="utf-8-sig"
    )
    assert review.splitlines()[0] == (
    "document,page_number,selection_reason,numeric_tokens,"
    "values_correct,table_structure_correct,notes"
    )


def test_root_detection_after_changing_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("FINANCIAL_INTELLIGENCE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_project_root() == PROJECT_ROOT
    assert load_config("data")["documents"]["files"] == ["uber_2024.pdf"]


def test_root_override(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    monkeypatch.setenv("FINANCIAL_INTELLIGENCE_ROOT", str(tmp_path))
    assert find_project_root() == tmp_path


def test_entry_points_import_without_running(capsys):
    for name in ("ingest", "audit_data", "review_tables"):
        module = importlib.import_module(
            f"financial_intelligence.cli.{name}"
        )
        assert callable(module.main)

    assert capsys.readouterr().out == ""
