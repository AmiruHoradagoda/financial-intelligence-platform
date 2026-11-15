import importlib
import json
from pathlib import Path

import pymupdf

from financial_intelligence.cli import (
    audit_data,
    generate_questions,
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
    for module in (ingest, audit_data, review_tables, generate_questions):
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

    categories = ["hard_fact"] * 4 + ["strategic_summary"] * 3 + ["stylistic"] * 3
    questions = [
        {"category": category, "question": f"Question {i}?"}
        for i, category in enumerate(categories)
    ]

    class FakeLLM:
        def complete(self, *, model, prompt):
            assert model == "fake/model"
            return json.dumps({"questions": questions})

    monkeypatch.setenv("QUESTION_MODEL", "ignored/env-model")
    monkeypatch.setattr(generate_questions, "LiteLLMClient", FakeLLM)
    # The existing pilot CLI selects these three specific chunks.
    pilot_chunks = [
        {
            "chunk_id": chunk_id,
            "source_file": "uber_2024.pdf",
            "page_number": page,
            "text": "Revenue grew by 10 percent. " * 30,
        }
        for chunk_id, page in [
            ("uber_2024_data_factory_p2_c0", 2),
            ("uber_2024_data_factory_p18_c0", 18),
            ("uber_2024_data_factory_p57_c1", 57),
        ]
    ]
    monkeypatch.setattr(generate_questions, "load_chunks", lambda path: pilot_chunks)
    generate_questions.main()
    rows = load_chunks(tmp_path / "data/generated/questions_quality_pilot.jsonl")
    assert len(rows) == 30
    assert list(rows[0]) == [
        "chunk_id",
        "source_file",
        "page_number",
        "category",
        "question",
    ]
    for index, chunk in enumerate(pilot_chunks):
        assert all(
            row["chunk_id"] == chunk["chunk_id"]
            and row["source_file"] == chunk["source_file"]
            and row["page_number"] == chunk["page_number"]
            for row in rows[index * 10 : (index + 1) * 10]
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
    for name in ("ingest", "audit_data", "review_tables", "generate_questions"):
        module = importlib.import_module(f"financial_intelligence.cli.{name}")
        assert callable(module.main)
        assert (
            Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
        ).is_file()
    assert capsys.readouterr().out == ""
