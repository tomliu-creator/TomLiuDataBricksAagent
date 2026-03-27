# Off-Balance-Sheet Analysis Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone Databricks notebook that retrieves off-balance-sheet liability disclosures across fiscal years from the existing RAG pipeline and presents them as a structured table (items x years) with a narrative summary.

**Architecture:** Single notebook (`notebooks/08_off_balance_sheet_study.py`) that reuses `_config` and `_utils`, retrieves chunks via Vector Search per OBS category, calls `ai_query('databricks-gpt-oss-20b', ...)` to extract year-by-year findings, and pivots results into a pandas DataFrame for display. One final LLM call generates a narrative synthesis.

**Tech Stack:** Databricks notebooks, Vector Search client, `ai_query` SQL function with `databricks-gpt-oss-20b`, PySpark, pandas.

---

## Key Design Decisions

1. **Per-category retrieval** (not per-year): 5 categories = 5 VS queries + 5 LLM calls. This keeps total LLM calls at 6 (5 items + 1 narrative), well within a smooth-execution budget.
2. **Prompt asks LLM to return simple delimited format** (not JSON): `databricks-gpt-oss-20b` is a 20B model — JSON parsing from small models is unreliable. Instead, ask for `YYYY: finding` lines and parse with regex. Fall back to raw text if parsing fails.
3. **Prompt budget of 12,000 chars**: proven safe for this model from our 06/07 fixes.
4. **All widget override patterns from 07** are copied in, so stale-widget issues cannot recur.
5. **Available fiscal years**: 2011–2023 per `_config.py FISCAL_YEARS`.

## Off-Balance-Sheet Categories

These are the 5 row categories for the table:

| # | Category | Retrieval Query |
|---|----------|----------------|
| 1 | Operating leases / IFRS 16 | "operating lease commitments, IFRS 16 right-of-use assets, lease liabilities off-balance-sheet" |
| 2 | Factoring & receivables | "factoring, receivables securitization, off-balance-sheet transfer of receivables" |
| 3 | Guarantees & sureties | "guarantees given, sureties, parent company guarantees, off-balance-sheet commitments" |
| 4 | Purchase & investment commitments | "purchase commitments, capital expenditure commitments, investment obligations not on balance sheet" |
| 5 | Contingent liabilities | "contingent liabilities, litigation provisions, tax disputes, pending claims off-balance-sheet" |

---

## File Structure

- **Create:** `notebooks/08_off_balance_sheet_study.py` — the single deliverable

No other files are created or modified.

---

### Task 1: Create notebook boilerplate (config, imports, widget setup)

**Files:**
- Create: `notebooks/08_off_balance_sheet_study.py`

This task creates the file with cells 1–3: magic runs, pip install, imports, widget setup with stale-value overrides. This is a direct copy of the proven pattern from `07_llm_query_demo.py`.

- [ ] **Step 1: Create the notebook file with boilerplate**

```python
# Databricks notebook source
# NOTEBOOK FILE: 08_off_balance_sheet_study.py
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 08 Off-Balance-Sheet Liability Study
# MAGIC
# MAGIC Retrieves off-balance-sheet disclosures across fiscal years and presents:
# MAGIC 1. A table with OBS categories as rows and fiscal years as columns
# MAGIC 2. A narrative summary synthesizing the findings

# COMMAND ----------
# MAGIC %pip install -U databricks-vectorsearch

# COMMAND ----------

import json
import re
from datetime import datetime, timezone

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F
import pandas as pd

if "log_pipeline_error" not in globals():
    raise RuntimeError("Missing `log_pipeline_error`. Rerun the notebook from the top so `%run ./_utils` executes.")

# --- Widget setup ---
dbutils.widgets.text("vs_endpoint_name", DEFAULT_VS_ENDPOINT)
dbutils.widgets.text("vs_index_name", DEFAULT_VS_INDEX)
dbutils.widgets.text("top_k", "10")
dbutils.widgets.text("model_name", "databricks-gpt-oss-20b")

VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()
VS_INDEX_NAME = dbutils.widgets.get("vs_index_name").strip()
TOP_K = int(dbutils.widgets.get("top_k"))
MODEL_NAME = dbutils.widgets.get("model_name").strip()

# Override stale widget values.
_STALE = {"databricks-claude-sonnet-4-6", "databricks-claude-sonnet-4-5", "", None}
if MODEL_NAME in _STALE:
    print(f"[override] MODEL_NAME '{MODEL_NAME}' -> 'databricks-gpt-oss-20b'")
    MODEL_NAME = "databricks-gpt-oss-20b"

print("VS endpoint:", VS_ENDPOINT_NAME)
print("VS index:", VS_INDEX_NAME)
print("TOP_K:", TOP_K)
print("MODEL_NAME:", MODEL_NAME)
```

- [ ] **Step 2: Verify the file is syntactically valid**

Open the file and confirm no syntax errors. The `# COMMAND ----------` separators and `# MAGIC` prefixes follow the same pattern as all other notebooks in the repo.

- [ ] **Step 3: Commit**

```bash
git add notebooks/08_off_balance_sheet_study.py
git commit -m "feat: add 08_off_balance_sheet_study notebook boilerplate"
```

---

### Task 2: Add retrieval and LLM helper functions

**Files:**
- Modify: `notebooks/08_off_balance_sheet_study.py` (append new cell)

These helpers are simplified versions of 07's `retrieve_chunks` and `call_llm_ai_query`, plus a new function to extract year-tagged findings from LLM output.

- [ ] **Step 1: Add the helper functions cell**

Append after the widget setup cell:

```python
# COMMAND ----------

OBS_CATEGORIES = [
    {
        "name": "Operating leases / IFRS 16",
        "query": "operating lease commitments, IFRS 16 right-of-use assets, lease liabilities off-balance-sheet",
    },
    {
        "name": "Factoring & receivables",
        "query": "factoring, receivables securitization, off-balance-sheet transfer of receivables",
    },
    {
        "name": "Guarantees & sureties",
        "query": "guarantees given, sureties, parent company guarantees, off-balance-sheet commitments",
    },
    {
        "name": "Purchase & investment commitments",
        "query": "purchase commitments, capital expenditure commitments, investment obligations not on balance sheet",
    },
    {
        "name": "Contingent liabilities",
        "query": "contingent liabilities, litigation provisions, tax disputes, pending claims off-balance-sheet",
    },
]


def retrieve_chunks(query: str, top_k: int) -> list[dict]:
    """Retrieve chunks from Vector Search and join back to Delta for full text."""
    vsc = VectorSearchClient(disable_notice=True)
    idx = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
    cols = ["chunk_id", "fiscal_year", "page_start", "page_end", "file_name", "source_path_dbfs"]
    resp = idx.similarity_search(query_text=query, columns=cols, num_results=top_k)

    # Parse manifest + data_array response.
    if not isinstance(resp, dict):
        return []
    result = resp.get("result") or resp
    manifest = resp.get("manifest") or (result or {}).get("manifest") or {}
    data = (result or {}).get("data_array") or []
    names = [c.get("name") for c in manifest.get("columns", [])]
    hits = []
    for r in data:
        if names and len(names) == len(r):
            hits.append(dict(zip(names, r)))

    chunk_ids = [h["chunk_id"] for h in hits if h.get("chunk_id")]
    if not chunk_ids:
        return []

    df = (
        spark.table(CHUNKS_TABLE)
        .filter(F.col("chunk_id").isin(chunk_ids))
        .select("chunk_id", "fiscal_year", "file_name", "page_start", "page_end",
                "source_path_dbfs", "chunk_text_en")
    )
    by_id = {r["chunk_id"]: r.asDict() for r in df.collect()}
    return [by_id[h["chunk_id"]] for h in hits if h.get("chunk_id") in by_id]


def call_ai_query(prompt: str) -> str:
    """Call ai_query with the configured model. Raises on empty/NULL."""
    prompt_sql = json.dumps(prompt)
    sql = f"SELECT ai_query('{MODEL_NAME}', {prompt_sql}) AS out"
    raw = spark.sql(sql).collect()[0]["out"]
    if not raw or not raw.strip():
        raise RuntimeError(
            f"ai_query returned empty/NULL. Prompt was {len(prompt)} chars."
        )
    return raw.strip()


def build_extraction_prompt(category_name: str, chunks: list[dict],
                            max_prompt_chars: int = 12000) -> str:
    """Ask the LLM to extract year-by-year findings for one OBS category."""
    header = (
        "You are a financial statement analyst.\n"
        "From the evidence below, extract information about: "
        f"{category_name}\n\n"
        "For EACH fiscal year mentioned, write exactly one line in this format:\n"
        "YYYY: <brief finding — amounts, key terms, or 'not mentioned'>\n\n"
        "Only output the year lines. Do not add commentary.\n"
        "If a year has no relevant information, skip it.\n\n"
        "EVIDENCE:\n"
    )
    budget = max_prompt_chars - len(header)
    chars_per = max(200, budget // max(len(chunks), 1) - 100)
    lines = []
    for i, ch in enumerate(chunks, start=1):
        cite = f"[FY{ch['fiscal_year']} p{ch['page_start']}-{ch['page_end']}]"
        text = (ch.get("chunk_text_en") or "")[:chars_per]
        lines.append(f"{cite} {text}")
    return header + "\n\n".join(lines)


def parse_year_findings(llm_output: str) -> dict[int, str]:
    """Parse 'YYYY: finding' lines from LLM output into {year: finding}."""
    findings = {}
    for line in llm_output.split("\n"):
        m = re.match(r"(\d{4})\s*:\s*(.+)", line.strip())
        if m:
            year = int(m.group(1))
            if 2000 <= year <= 2030:
                findings[year] = m.group(2).strip()
    return findings
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/08_off_balance_sheet_study.py
git commit -m "feat(08): add retrieval and LLM helper functions"
```

---

### Task 3: Add the main analysis loop

**Files:**
- Modify: `notebooks/08_off_balance_sheet_study.py` (append new cell)

This cell iterates over the 5 OBS categories, retrieves evidence, calls the LLM, and collects results.

- [ ] **Step 1: Add the analysis loop cell**

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ### Run Off-Balance-Sheet Extraction

# COMMAND ----------

results = {}  # {category_name: {year: finding}}

for i, cat in enumerate(OBS_CATEGORIES, start=1):
    name = cat["name"]
    print(f"[{i}/{len(OBS_CATEGORIES)}] {name} ...")
    chunks = retrieve_chunks(cat["query"], TOP_K)
    print(f"  Retrieved {len(chunks)} chunks")
    if not chunks:
        print(f"  SKIP — no chunks found")
        results[name] = {}
        continue

    prompt = build_extraction_prompt(name, chunks)
    print(f"  Prompt: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    try:
        raw_answer = call_ai_query(prompt)
        findings = parse_year_findings(raw_answer)
        print(f"  Parsed {len(findings)} year entries")
        if not findings:
            print(f"  WARNING: no year lines parsed. Raw output:\n  {raw_answer[:300]}")
        results[name] = findings
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        results[name] = {}

print(f"\nDone. {sum(1 for v in results.values() if v)}/{len(OBS_CATEGORIES)} categories have data.")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/08_off_balance_sheet_study.py
git commit -m "feat(08): add main OBS extraction loop"
```

---

### Task 4: Build and display the pivot table

**Files:**
- Modify: `notebooks/08_off_balance_sheet_study.py` (append new cell)

- [ ] **Step 1: Add the table display cell**

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ### Off-Balance-Sheet Items by Fiscal Year

# COMMAND ----------

# Build a pandas DataFrame: rows = categories, columns = fiscal years that have data.
all_years = sorted({yr for findings in results.values() for yr in findings})
if not all_years:
    print("No year-level findings were extracted. Check the LLM outputs above.")
else:
    table_data = []
    for cat in OBS_CATEGORIES:
        name = cat["name"]
        row = {"Category": name}
        for yr in all_years:
            row[str(yr)] = results.get(name, {}).get(yr, "—")
        table_data.append(row)

    df_table = pd.DataFrame(table_data)
    df_table = df_table.set_index("Category")

    # Display in Databricks (renders as a formatted table).
    display(spark.createDataFrame(pd.DataFrame(table_data)))

    # Also print a text version for the notebook output.
    print(df_table.to_string())
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/08_off_balance_sheet_study.py
git commit -m "feat(08): add OBS pivot table display"
```

---

### Task 5: Generate narrative summary

**Files:**
- Modify: `notebooks/08_off_balance_sheet_study.py` (append new cell)

One final LLM call that reads the table and produces a narrative.

- [ ] **Step 1: Add the narrative cell**

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ### Narrative Summary

# COMMAND ----------

# Build a compact text version of the table for the LLM.
table_text_lines = []
for cat in OBS_CATEGORIES:
    name = cat["name"]
    entries = results.get(name, {})
    if entries:
        year_strs = [f"FY{yr}: {txt}" for yr, txt in sorted(entries.items())]
        table_text_lines.append(f"**{name}**\n" + "\n".join(year_strs))
    else:
        table_text_lines.append(f"**{name}**\nNo data extracted.")

narrative_prompt = (
    "You are a financial analyst writing a brief report section.\n"
    "Below is a summary of off-balance-sheet liabilities for Parts Holding Europe across fiscal years.\n"
    "Write 2-3 paragraphs of narrative analysis: key trends, material changes between years, "
    "and any risks worth highlighting. Be concise and factual.\n\n"
    + "\n\n".join(table_text_lines)
)

if any(results.values()):
    print(f"[narrative] Prompt: {len(narrative_prompt)} chars")
    try:
        narrative = call_ai_query(narrative_prompt)
        print("\n" + narrative)
    except Exception as e:
        print(f"Narrative generation failed: {type(e).__name__}: {e}")
else:
    print("No data to narrate.")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/08_off_balance_sheet_study.py
git commit -m "feat(08): add narrative summary generation"
```

---

### Task 6: Smoke test the complete notebook

- [ ] **Step 1: Run the notebook end-to-end in Databricks**

Upload `08_off_balance_sheet_study.py` to the Databricks workspace. Run All.

Expected output:
1. Widget setup cell prints `MODEL_NAME: databricks-gpt-oss-20b`
2. Loop prints `[1/5] Operating leases / IFRS 16 ...` through `[5/5]` with chunk counts and parsed year counts
3. Table displays with categories as rows, fiscal years as columns, brief findings in cells
4. Narrative prints 2-3 paragraphs of analysis

- [ ] **Step 2: If any category returns 0 parsed year lines**

Check the `WARNING: no year lines parsed` output. Common causes:
- The model returned prose instead of `YYYY: finding` format — adjust the prompt wording
- Chunks retrieved had no relevant content — try increasing `top_k` to 12

- [ ] **Step 3: Final commit**

```bash
git add notebooks/08_off_balance_sheet_study.py
git commit -m "feat(08): complete off-balance-sheet study notebook"
```
