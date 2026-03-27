# Bug Fix Summary: Notebooks 06 and 07

## Context

Notebooks `notebooks/06_retrieval_eval.py` and `notebooks/07_llm_query_demo.py` were retrieving chunks successfully from Vector Search, but LLM answers were coming back as NULL/empty or were failing due to stale widget values.

The workspace does not have the Claude Sonnet endpoints that were referenced in earlier runs, but `ai_query('databricks-gpt-oss-20b', ...)` is available and working.

## Fixes Applied

- Stale Databricks widget values:
  - Widgets persist their last-selected value across reruns. If a previous run used `llm_mode=anthropic_messages` or a non-existent model name (for example `databricks-claude-sonnet-4-6`), reruns kept failing even after the code default changed.
  - 06 and 07 now defensively override stale values after reading widgets, forcing `ai_query` with `databricks-gpt-oss-20b` when the old values are detected.
- Prompt budgeting:
  - Very large prompts can cause `ai_query` to return empty/NULL.
  - 06 and 07 now cap prompt size (character budget) and trim evidence per chunk to fit the budget.
- Fail fast on empty `ai_query` outputs:
  - 06 and 07 now raise a `RuntimeError` if `ai_query(...)` returns NULL/empty (includes prompt size in the error for diagnosis).
- Retrieval-eval runs are appended, but the notebook displays only the latest run:
  - `retrieval_eval_runs` remains append-only for history, and 06 filters the display to the current `run_ts`.

