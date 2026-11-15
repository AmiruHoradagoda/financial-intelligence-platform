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

The generation command makes provider API calls, which may incur charges. The
existing pilot selects chunks `uber_2024_data_factory_p2_c0`,
`uber_2024_data_factory_p18_c0`, and `uber_2024_data_factory_p57_c1`, generating
ten questions per chunk into `data/generated/questions_quality_pilot.jsonl`.
The other commands run locally. Workflow commands overwrite their respective  
output files when explicitly run; the test suite uses temporary files and fakes.

Equivalent thin script entry points are available as `uv run scripts/ingest.py`,  
`uv run scripts/audit_data.py`, `uv run scripts/review_tables.py`, and  
`uv run scripts/generate_questions.py`. Import from `financial_intelligence` and  
use the package commands above; the previous `src.*` modules have been removed.

## Resumable batch question generation

Preview a five-chunk batch without API calls or file writes:

```powershell
uv run python -m financial_intelligence.cli.batch_generate_questions --max-new-chunks 5 --dry-run
```

Explicitly run up to five unfinished chunks (the default limit is also five):

```powershell
uv run python -m financial_intelligence.cli.batch_generate_questions --max-new-chunks 5
```

Repeat the command to resume. Each validated ten-question batch is saved separately
under `data/generated/question_checkpoints/` using atomic replacement. Completed
chunks are skipped. Failed and excluded chunks count toward the limit. Failed
chunks remain eligible for
a later run. Transient connection, timeout, rate-limit, and server errors receive
at most three attempts per chunk, waiting one then two seconds. The batch CLI
disables provider retries, bounding a five-chunk invocation to fifteen completion
attempts. Invalid batches and permanent API errors are logged without retries;
questions are never padded or repaired to meet the count.

When an excerpt cannot support ten useful, distinct, grounded questions, the
generator can return `{"unprocessable_reason": "specific reason"}` instead of a
question batch. The generator raises `InsufficientInformation`; the batch runner
atomically checkpoints an `excluded` status and the reason, without questions.
Blank excerpts are excluded locally without an API call. Exclusions are skipped
on resume, checked against the same model/prompt/source fingerprints, and omitted
from the question export. Other malformed or partial responses remain failures.
This is an insufficiency signal from the generating model, not an independent
semantic quality assessment. Successful batches still require exactly 4/3/3.

The CLI reports attempted chunks, successful chunks, completed checkpoints skipped,
failed chunks, new and previously recorded exclusions, each excluded ID and reason,
and remaining chunks. Remaining includes failed chunks eligible for retry but
excludes completed and explicitly excluded chunks. A dry run leaves remaining
unchanged; it does not assess unprocessed excerpts for semantic sufficiency.

Checkpoint metadata records the configured model and SHA-256 fingerprints of the
prompt template and source text. All existing checkpoints for the input chunks
are validated before generation. Missing metadata, corrupt results, or changes to
the model, prompt, text, or source metadata stop the run before API spending.
Archive incompatible checkpoints or use `--checkpoint-dir` with a new directory;
they are never silently reused or overwritten. Use one batch process per directory.

Each normal run atomically exports all compatible completed chunks, in source
order, to `data/generated/questions.jsonl`. Rows retain `chunk_id`, `source_file`,
`page_number`, `category`, and `question`. To rebuild only the export without API
calls:

```powershell
uv run python -m financial_intelligence.cli.batch_generate_questions --export-only
```

Use `--output` to choose a different export path. Terminal failures append to
`question_checkpoints/failures.jsonl` with source metadata, model and fingerprints,
timestamp, attempt count, and error details. The CLI exits with status 1 after
exporting successes if any chunk failed. A storage error stops the run; already
saved checkpoints remain available to resume or export.

The pilot file and existing pilot command are independent of these checkpoints.
Schema validation enforces the ten-question 4/3/3 contract; semantic quality and
grounded-answer validation with LLM B remain deferred. No LLM judge is included.

### Quality changes and the next pilot

The local Gemini feedback in `data/evaluation/question_quality_review_gemini.csv`
contains 19 accepts, 4 reviews, and 7 rejects out of 30. These are provisional AI
labels, not human ground truth. The source and pilot review identified:

- `pilot_013`: acquisitions are the source of expected efficiencies; the excerpt
  gives no cause for failing to realize them. The prompt now requires explicit
  causal evidence and distinguishes numerical changes from their unstated drivers.
- `pilot_016`: the network-scale sentence ends at `merch`. The prompt excludes
  incomplete boundary sentences and requires complete table rows, periods, and units.
- `pilot_007`, `pilot_017`, and `pilot_027`: repeated informational objectives.
  The prompt requires distinct objectives across all ten questions; changing only
  wording, tone, audience, or format does not establish a distinct objective.
- `pilot_003`: promotional self-description has little business value. The prompt
  prioritizes business facts, measures, operations, and stated risks over slogans.
- Partial sections must be described as the provided excerpt. Requested bullet
  counts must fit the available evidence. Insufficient content must be excluded
  rather than stretched into ten questions.

For the next pilot, run `uv run scripts/find_pilot_chunks.py` to locate candidate
IDs, then inspect their entire source excerpts, including both boundaries. Keep
the original page 2 mission, page 18 risk, and page 57 table chunks as regression
cases; the mission chunk is also an insufficiency probe, not a guaranteed success.
Add a substantive operations narrative and a financial discussion with explicitly
stated drivers. Select across sections rather than taking the first five keyword
matches; include both useful complete content and boundary stress cases.

Pass the selected source records to the existing `run_question_batch` function with
a separate checkpoint directory (for example `question_checkpoints_quality_v2`)
and `output_path` (for example `questions_quality_pilot_v2.jsonl`). Keep the original
pilot and labels intact, record all excluded IDs and reasons, and manually review
the new questions against the same source excerpts and provisional rubric. No new
pilot questions or quality scores are generated automatically by this change.

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
        batch.py                # Bounded generation, checkpoints, retries, export
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
*   Question pilot: `data/generated/questions_quality_pilot.jsonl` with `chunk_id`,
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
