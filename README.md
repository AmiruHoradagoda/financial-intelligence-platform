# Financial Intelligence Platform

An AI engineering mini-project using the **Uber 2024 Annual Report**. The current working scope is **Phase 1: Data Factory**—turning PDF chunks into source-grounded question-and-answer (Q&A) records.

**Current status:** PDF ingestion, chunking, auditing, resumable question generation (LLM A), and grounded answer generation (LLM B) are implemented. Dataset splitting, QLoRA fine-tuning, advanced RAG, and final evaluation are **not yet complete**.

## 1. Setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --locked
# If .env does not already exist:
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Model names and chunking settings are in `config/data.yaml`. Never commit `.env`.

Place the report at `data/raw/uber_2024.pdf`.

## 2. How Phase 1 works

```text
Uber 2024 PDF
    |
    v
PDF extraction + chunking
    |
    +--> data/processed/data_factory/uber_2024_chunks.jsonl
    |          |
    |          v
    |      LLM A: 10 questions per eligible chunk
    |          |
    |          v
    |      data/generated/questions.jsonl
    |          |
    |          v
    |      LLM B: answer using original source chunk
    |          |
    |          v
    |      Source/evidence validation
    |          |
    |          v
    |      data/generated/qa_pairs.jsonl
    |
    +--> data/processed/rag/uber_2024_chunks.jsonl
           (prepared for the later RAG phase)
```

LLM A requests **4 hard facts + 3 strategic summaries + 3 stylistic questions** per eligible chunk. Chunks that cannot support ten distinct questions can be excluded and recorded. LLM B uses the **same original chunk**; only accepted answers are exported to the Q&A file. Failed or review-required answers are not silently accepted.

## 3. Commands

Run commands from the repository root.

### Offline verification (no API calls)

```powershell
uv run python -m pytest -q
```

### Ingest / audit PDF (only when inputs or chunking settings change)

**Warning:** Ingestion rewrites processed chunk files. Do not rerun it casually while resuming existing generation checkpoints.

```powershell
uv run python -m financial_intelligence.cli.ingest
uv run python -m financial_intelligence.cli.audit_data
```

### Generate questions with LLM A (paid API calls unless dry-run)

```powershell
# Preview the next 5 chunks; no API calls
uv run python -m financial_intelligence.cli.batch_generate_questions --max-new-chunks 5 --dry-run

# Process up to 5 unfinished chunks
uv run python -m financial_intelligence.cli.batch_generate_questions --max-new-chunks 5
```

Successful chunk results are saved in `data/generated/question_checkpoints/`. Repeat to resume; previously completed and excluded chunks are skipped.

### Generate answers with LLM B (paid API calls unless dry-run)

```powershell
# Preview the next 10 questions; no API calls
uv run python -m financial_intelligence.cli.generate_answers --max-new-questions 10 --output data/generated/qa_pairs.jsonl --dry-run

# Attempt up to 10 unattempted questions, with one attempt per question
uv run python -m financial_intelligence.cli.generate_answers --max-new-questions 10 --max-attempts 1 --output data/generated/qa_pairs.jsonl
```

Answer checkpoints are stored in `data/generated/answer_checkpoints/`. Completed, review-required, and historical failed questions are not reattempted in the normal resume flow.

### Export accepted records again without API calls

```powershell
uv run python -m financial_intelligence.cli.batch_generate_questions --export-only
uv run python -m financial_intelligence.cli.generate_answers --export-only --output data/generated/qa_pairs.jsonl
```

These commands rebuild exports from compatible saved checkpoints; they do not retry failed questions.

## 4. Main files

| Location | Responsibility |
| --- | --- |
| `config/data.yaml` | Source, chunking, models, intended dataset split |
| `src/financial_intelligence/ingestion/` | PDF extraction, chunking and audits |
| `src/financial_intelligence/data_factory/question_generator.py` | LLM A: chunk to ten questions |
| `src/financial_intelligence/data_factory/answer_generator.py` | LLM B: question + original chunk to answer |
| `src/financial_intelligence/data_factory/batch.py` | Resumable question batches |
| `src/financial_intelligence/data_factory/answer_batch.py` | Resumable answer batches |
| `src/financial_intelligence/data_factory/validation.py` | Question/response parsing and validation |
| `src/financial_intelligence/schemas/qa.py` | Q&A data contracts |
| `src/financial_intelligence/cli/` | Command-line entry points |
| `tests/` | Offline tests |

## 5. Outputs and troubleshooting

| Output | Meaning |
| --- | --- |
| `data/processed/data_factory/uber_2024_chunks.jsonl` | Source chunks for LLM A/B |
| `data/processed/rag/uber_2024_chunks.jsonl` | Overlapping chunks for future RAG |
| `data/generated/questions.jsonl` | Exported generated questions |
| `data/generated/qa_pairs.jsonl` | Accepted Q&A pairs |
| `data/generated/question_checkpoints/` | Per-chunk question progress |
| `data/generated/answer_checkpoints/` | Per-question answer progress |
| `data/evaluation/data_factory_audit.json` | Ingestion audit |

- **`EvidenceQuoteMismatch`:** Model-provided evidence was not an exact source substring. The answer is not exported as accepted.
- **`Held for review`:** The answer has a grounding/coverage concern and remains outside accepted exports.
- **`Excluded`:** A chunk was insufficient for ten distinct grounded questions.
- **`Skipped existing`:** A compatible checkpoint was found, avoiding another API call.

Do not delete or overwrite checkpoint directories during cleanup. Use `pytest` rather than paid LLM calls when testing code changes.

## 6. Remaining assignment work

1. Prepare the required **80% `train.jsonl` / 20% `golden_test_set.jsonl`** split using accepted pairs.
2. Implement and execute **QLoRA fine-tuning**.
3. Implement the required **advanced RAG** pipeline.
4. Run **evaluation** and report measured results and limitations.

Files such as `config/finetuning.yaml`, `config/rag.yaml`, and `config/evaluation.yaml` are preparation/configuration; their presence does **not** mean those phases have been implemented or executed.
