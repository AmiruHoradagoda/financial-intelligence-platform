
import json
import pymupdf
from pathlib import Path
from collections import defaultdict
from src.config_loader import PROJECT_ROOT, load_config
from src.ingestion import clean_text

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


#orchestrator function
def audit_document(
    chunks: list[dict],
    page_texts: dict[int, str],
    max_chunk_size=1500
) -> dict:
    """
    Audit chunk metadata, offsets and text coverage.

    page_texts:
        {1: "Page one text", 2: "Page two text"}
    """

    invalid_ids = []
    invalid_pages = []
    invalid_offsets = []
    coverage_gaps = []
    oversized_chunks=[]
    

    valid_id_chunks = []
    chunks_by_page = defaultdict(list)

    # Step 1: Validate each chunk
    for index, chunk in enumerate(chunks):

        chunk_id = chunk.get("chunk_id")

        # Check ID
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            invalid_ids.append(index)
        else:
            valid_id_chunks.append(chunk)

        # Check page number
        page_number = chunk.get("page_number")

        if (
            type(page_number) is not int
            or page_number not in page_texts
        ):
            invalid_pages.append(index)
            continue

        # Check offsets against original page
        page_text = page_texts[page_number]

        if not validate_chunk_offsets(chunk, page_text):
            invalid_offsets.append(index)
            continue

        chunks_by_page[page_number].append(chunk)

    # Step 2: Detect duplicate IDs
    duplicate_ids = find_duplicate_chunk_ids(
        valid_id_chunks
    )

    # Step 3: Detect missing text coverage
    for page_number, page_text in page_texts.items():

        if not page_text.strip():
            continue

        page_chunks = sorted(
            chunks_by_page[page_number],
            key=lambda c: c["start_char"]
        )

        covered_until = 0

        for chunk in page_chunks:
            start = chunk["start_char"]
            end = chunk["end_char"]

            # Find a gap not covered by any chunk
            if start > covered_until:
                missing = page_text[covered_until:start]

                if missing.strip():
                    coverage_gaps.append({
                        "page_number": page_number,
                        "start": covered_until,
                        "end": start,
                    })

            covered_until = max(covered_until, end)

        # Check text after the final chunk
        if page_text[covered_until:].strip():
            coverage_gaps.append({
                "page_number": page_number,
                "start": covered_until,
                "end": len(page_text),
            })

    # Step 4: check whether there oversized chunks
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id')
        text = chunk.get("text")
        if isinstance(text, str) and len(text) > max_chunk_size:
            oversized_chunks.append(chunk_id)



    # Step 5: Build report
    report = {
        "total_chunks": len(chunks),
        "duplicate_ids": duplicate_ids,
        "invalid_ids": invalid_ids,
        "invalid_pages": invalid_pages,
        "invalid_offsets": invalid_offsets,
        "oversized_chunks":oversized_chunks,
        "coverage_gaps": coverage_gaps,
    }

    has_errors = any([
        duplicate_ids,
        invalid_ids,
        invalid_pages,
        invalid_offsets,
        coverage_gaps,
        oversized_chunks
    ])

    report["status"] = "FAIL" if has_errors else "PASS"

    return report






def main():
    config = load_config("data")

    pdf_path = (
        PROJECT_ROOT
        / config["documents"]["input_dir"]
        / config["documents"]["files"][0]
    )

    processed_dir = (
        PROJECT_ROOT
        / config["documents"]["output_dir"]
    )

    # Extract original cleaned text for offset checking
    with pymupdf.open(pdf_path) as pdf:
        page_texts = {
            i + 1: clean_text(page.get_text("text", sort=True))
            for i, page in enumerate(pdf)
        }

    # Audit both chunking pipelines
    for pipeline in ("data_factory", "rag"):

        chunk_path = (
            processed_dir
            / pipeline
            / f"{pdf_path.stem}_chunks.jsonl"
        )

        chunks = load_chunks(chunk_path)

        report = audit_document(chunks, page_texts)

        output_path = (
            PROJECT_ROOT
            / "data/evaluation"
            / f"{pipeline}_audit.json"
        )

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        output_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8"
        )

        print(f"\nPipeline: {pipeline}")
        print(f"Status: {report['status']}")
        print(f"Total chunks: {report['total_chunks']}")
        print(f"Duplicate IDs: {len(report['duplicate_ids'])}")
        print(f"Invalid offsets: {len(report['invalid_offsets'])}")
        print(f"Coverage gaps: {len(report['coverage_gaps'])}")


if __name__ == "__main__":
    main()

