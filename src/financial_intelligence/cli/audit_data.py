"""Run the audit_data workflow."""

import json

import pymupdf

from financial_intelligence.config import PROJECT_ROOT, load_config
from financial_intelligence.infrastructure.jsonl import load_chunks
from financial_intelligence.ingestion.pdf_reader import clean_text
from financial_intelligence.ingestion.quality import audit_document


def main() -> None:
    config = load_config("data")

    pdf_path = (
        PROJECT_ROOT
        / config["documents"]["input_dir"]
        / config["documents"]["files"][0]
    )

    processed_dir = PROJECT_ROOT / config["documents"]["output_dir"]

    # Extract original cleaned text for offset checking
    with pymupdf.open(pdf_path) as pdf:
        page_texts = {
            i + 1: clean_text(page.get_text("text", sort=True))
            for i, page in enumerate(pdf)
        }

    # Audit both chunking pipelines
    for pipeline in ("data_factory", "rag"):
        chunk_path = processed_dir / pipeline / f"{pdf_path.stem}_chunks.jsonl"

        chunks = load_chunks(chunk_path)

        report = audit_document(
            chunks,
            page_texts,
            max_chunk_size=int(config["chunking"][pipeline]["chunk_size"]),
        )

        output_path = PROJECT_ROOT / "data/evaluation" / f"{pipeline}_audit.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"\nPipeline: {pipeline}")
        print(f"Status: {report['status']}")
        print(f"Total chunks: {report['total_chunks']}")
        print(f"Duplicate IDs: {len(report['duplicate_ids'])}")
        print(f"Invalid offsets: {len(report['invalid_offsets'])}")
        print(f"Coverage gaps: {len(report['coverage_gaps'])}")


if __name__ == "__main__":
    main()
