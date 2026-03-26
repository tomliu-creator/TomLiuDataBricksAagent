# TomLiuDataBricksAagent

## Databricks Financial Statement RAG Pilot (Parts Holding Europe)

This repo contains a refactored, rerunnable Databricks pilot that ingests French annual reports for Parts Holding Europe (13 PDFs across fiscal years), parses them, chunks them, translates chunks to English for retrieval (because multilingual embeddings are not available), indexes them in Vector Search, and answers English questions with citations back to the original French evidence.

## Storage Layout

### Unity Catalog (Databricks)

Requested UC structure:

- Catalog: `uc_cmifi_dev`
- Schema: `fin_agent`
- Volume: `annual_reports`
- Folder prefix: `source=parts_holding_europe`
- Subfolders: `2011` ... `2023` (some years may be empty)

Databricks paths used by the notebooks:

- DBFS path (Spark I/O): `dbfs:/Volumes/uc_cmifi_dev/fin_agent/annual_reports/source=parts_holding_europe/<YYYY>/...pdf`
- Local/FUSE path (Python libs like PyMuPDF): `/Volumes/uc_cmifi_dev/fin_agent/annual_reports/source=parts_holding_europe/<YYYY>/...pdf`

### Local Reference Structure (Windows)

To mirror the UC layout locally (for staging / reference), this pilot documents the following structure:

`C:\\Users\\sunno\\projects\\financial_agent_local\\Volumes\\uc_cmifi_dev\\fin_agent\\annual_reports\\source=parts_holding_europe\\<YYYY>\\`

Create it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\create_local_reference_structure.ps1
```

## Design: French Evidence + English Retrieval

1. Source text is extracted in French and stored as evidence (`document_pages.page_text_fr`, `document_chunks.chunk_text_fr`).
2. Chunks are translated to English using Databricks `ai_translate(chunk_text_fr, 'en')` and stored in `document_chunks.chunk_text_en`.
3. Vector Search embeddings and retrieval run over `chunk_text_en`.
4. Final answers cite original French chunks with fiscal year + page ranges + source path for auditability.

## Tables

DDL lives in `sql/fin_agent_ddl.sql`.

Main tables:

- `uc_cmifi_dev.fin_agent.document_inventory`
- `uc_cmifi_dev.fin_agent.document_pages`
- `uc_cmifi_dev.fin_agent.document_chunks`
- `uc_cmifi_dev.fin_agent.pipeline_errors`
- `uc_cmifi_dev.fin_agent.retrieval_eval_runs`

## Notebooks (Run Order)

All notebooks are Databricks-compatible `.py` notebooks under `notebooks/`.

1. `01_catalog_setup`
2. `02_inventory_ingest`
3. `03_parse_reports`
4. `04_chunk_translate`
5. `05_vector_index`
6. `06_retrieval_eval`
7. `07_llm_query_demo`

Optional checks:

- `00_validation_checks`

## Configuration Points

### Embeddings (required for indexing)

In `05_vector_index`, set the widget:

- `embedding_model_endpoint_name`: an English embedding model serving endpoint available in your workspace.

This pilot does not assume a specific embedding model name because it varies by workspace.

### LLM Answering (Claude Sonnet 4.6 preferred)

In `07_llm_query_demo`:

- `llm_mode = anthropic_messages` (preferred when enabled)
  - `model_name = databricks-claude-sonnet-4-6` (default, change if not enabled)
- Fallback: `llm_mode = ai_query`
  - `model_name` should be a Databricks Foundation Model API endpoint name you can query

If the workspace does not have Claude Sonnet 4.6 enabled, switch `model_name` to any available chat/instruct model or disable answer generation and only inspect retrieved evidence.

## Debugging / Validation

The pipeline is designed to avoid silent no-ops:

- Inventory ingest uses deterministic IDs + `MERGE` (no append-only duplicates).
- Parsing uses `/Volumes/...` paths for PyMuPDF.
- Chunking uses `MERGE` and resets translation/index only when French chunk content changes.
- Validation snapshots appear after each stage; run `00_validation_checks` any time.
