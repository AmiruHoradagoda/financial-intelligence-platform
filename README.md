# Financial Intelligence Platform

Python mini-project for extracting the Uber 2024 Annual Report, auditing its  
chunks, sampling financial tables for manual review, and generating grounded  
questions with LiteLLM. Answer generation, training, and retrieval are outside  
the current implementation.

## Setup

Use Python 3.11 or newer and the existing uv lockfile:

```
uv sync --locked
```

Copy `.env.example` to `.env` if you do not already have a local environment file.  
For question generation, set `data_generation.model` in `config/data.yaml`  
(currently `openai/gpt-4.1-mini`) and supply the provider's credentials in `.env`.  
Model selection belongs in YAML; credentials stay outside source control. Ingestion, audits,  
table sampling, and tests do not require an LLM API.

Place the report at `data/raw/uber_2024.pdf`. The configured report is Uber 2024  
only. YAML keys and relative paths in `config/` are preserved.

## Commands

Run from the repository root after setup:

```
uv run python -m financial_intelligence.cli.ingest
uv run python -m financial_intelligence.cli.audit_data
uv run python -m financial_intelligence.cli.review_tables
uv run python -m financial_intelligence.cli.generate_questions
uv run pytest
```

The generation command makes provider API calls, which may incur charges. It  
keeps the original preview behavior: select the first two Data Factory chunks  
with at least 500 nonblank characters and generate ten questions for each.  
The other commands run locally. Workflow commands overwrite their respective  
output files when explicitly run; the test suite uses temporary files and fakes.

Equivalent thin script entry points are available as `uv run scripts/ingest.py`,  
`uv run scripts/audit_data.py`, `uv run scripts/review_tables.py`, and  
`uv run scripts/generate_questions.py`. Import from `financial_intelligence` and  
use the package commands above; the previous `src.*` modules have been removed.

## Package layout

```
src/financial_intelligence/
    config.py                   # Project root, .env, YAML loading
    schemas/                    # Pydantic chunk, question, batch, and row contracts
    ingestion/
        chunking.py             # Pure fixed and sliding algorithms
        pdf_reader.py           # PDF extraction and source metadata
        quality.py              # Duplicate, offset, size, and coverage audits
        table_quality.py        # Deterministic review sampling
        verification.py         # Display extracted page text
    data_factory/
        prompts.py              # Pure grounded prompt construction
        validation.py           # JSON fence parsing and validation helper
        question_generator.py   # Orchestration with an injected LLM client
    infrastructure/
        jsonl.py                # Existing JSONL loading and persistence
        llm_client.py           # LiteLLM adapter and small completion protocol
    cli/                        # Configuration, file paths, workflow execution
scripts/                        # Thin command wrappers
tests/                          # Offline unit and workflow regression tests
```

The dependency direction is from CLI workflows to feature functions and schemas;  
`QuestionGenerator` receives a client rather than invoking LiteLLM directly.  
Only the adapter imports LiteLLM, lazily when a completion is requested. Pure  
chunking and prompt construction have no provider or filesystem dependencies.  
No placeholder packages were added for future features. Existing assignment  
notebooks remain intact.

Root detection works from the installed editable package even when the working  
directory changes. For a wheel installed outside the checkout, set  
`FINANCIAL_INTELLIGENCE_ROOT` to the repository containing `config/` and `data/`.

## Preserved contracts

*   Data Factory: fixed 1,500-character chunks, no overlap.
*   RAG: sliding 1,500-character chunks with 150-character overlap and the existing  
    whitespace-boundary preference. Both use `config/data.yaml`.
*   IDs: `{document_id}_{pipeline}_p{page_number}_c{chunk_index}`, with one-based  
    pages, zero-based per-page chunk indexes, and page-relative `[start_char, end_char)`  
    offsets into cleaned text.
*   Chunk JSONL fields: `chunk_id`, `document_id`, `source_file`, `pipeline`,  
    `strategy`, `page_number`, `start_char`, `end_char`, `text`.
*   Chunk paths: `data/processed/{data_factory|rag}/uber_2024_chunks.jsonl`.
*   Audit paths: `data/evaluation/{data_factory|rag}_audit.json`. Reports preserve  
    `total_chunks`, `duplicate_ids`, `invalid_ids`, `invalid_pages`, `invalid_offsets`,  
    `oversized_chunks`, `coverage_gaps`, and `status`. Each pipeline's configured  
    chunk size is used. Audits accept malformed raw records so errors can be reported.
*   Table review: deterministic five numeric-heavy and three random pages, seed 42;  
    `data/evaluation/table_quality_review.csv` retains its columns and UTF-8 BOM.
*   Question preview: `data/generated/questions_preview.jsonl` with `chunk_id`,  
    `source_file`, `page_number`, `category`, `question` in the original field order.
*   Each question batch must contain exactly four `hard_fact`, three  
    `strategic_summary`, and three `stylistic` questions. Empty, malformed,  
    miscategorized, or duplicate questions raise `InvalidLLMResponse`. Duplicate  
    comparison ignores case and repeated whitespace. The legacy  
    `validate_generated_questions` helper returns a boolean. JSON responses with  
    complete Markdown fences remain accepted; invalid JSON is rejected.
*   LiteLLM retains the configured model, temperature `0.2`, and `max_tokens=1800`.  
    Prompts prohibit speculation and unsupported facts and constrain stylistic  
    questions to grounded explanations or rewrites. Schema validation checks the  
    response structure; factual grounding still requires content review.

The refactor retains the existing dependency versions. Pydantic, already pinned  
transitively in `uv.lock`, is now an explicit dependency. The unfinished duplicate  
question validators were replaced with the complete Pydantic batch contract, and  
importing page verification no longer prints or loads an unrelated document.

## Verification

```
uv run pytest
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv lock --check
```

Tests use fake LLM clients and synthetic PDFs; they never call a provider. They  
cover chunk boundaries and offsets, metadata, malformed JSONL, duplicate IDs,  
oversized chunks, audit coverage, deterministic sampling, response parsing,  
question contracts, the provider request settings, and all four workflows with  
temporary output paths.

For a sandbox with restricted global cache or temp directories, use a writable  
local cache and a fresh local test directory:

```
uv --cache-dir .uv-cache run --no-sync pytest -p no:cacheprovider --basetemp=.test-tmp-local
```