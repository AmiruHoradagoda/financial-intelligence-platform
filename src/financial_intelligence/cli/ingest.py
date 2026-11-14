"""Run the ingest workflow."""

from financial_intelligence.config import PROJECT_ROOT, load_config
from financial_intelligence.infrastructure.jsonl import save_jsonl
from financial_intelligence.ingestion.pdf_reader import extract_pdf


def main() -> None:
    config = load_config("data")

    documents = config["documents"]
    chunking_configs = config["chunking"]

    input_dir = PROJECT_ROOT / documents["input_dir"]
    output_dir = PROJECT_ROOT / documents["output_dir"]

    for filename in documents["files"]:
        pdf_path = input_dir / filename

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        for pipeline_name, settings in chunking_configs.items():
            records, empty_pages = extract_pdf(
                pdf_path=pdf_path,
                pipeline_name=pipeline_name,
                settings=settings,
            )

            output_path = output_dir / pipeline_name / f"{pdf_path.stem}_chunks.jsonl"

            save_jsonl(records, output_path)

            print(f"\nPipeline: {pipeline_name}")
            print(f"Strategy: {settings['strategy']}")
            print(f"Generated chunks: {len(records)}")
            print(f"Empty pages: {empty_pages[:20]}")
            print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
