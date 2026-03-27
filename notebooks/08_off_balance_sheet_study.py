# Databricks notebook source
# NOTEBOOK FILE: 08_off_balance_sheet_study.py
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 08 Off-Balance-Sheet Liability Study
# MAGIC Retrieves off-balance-sheet disclosures across fiscal years and presents:
# MAGIC 1. A table with OBS categories as rows and fiscal years as columns
# MAGIC 2. A narrative summary synthesizing the findings

# COMMAND ----------
# MAGIC %pip install -U databricks-vectorsearch

# COMMAND ----------

import json
import re
from datetime import datetime, timezone

import pandas as pd
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Guard against missing _utils
if "log_pipeline_error" not in globals():
    raise RuntimeError("Missing `log_pipeline_error`. Rerun the notebook from the top so `%run ./_utils` executes.")

def _ensure_text_widget(name: str, default: str, override_if: set[str] | None = None):
    try:
        cur = dbutils.widgets.get(name)
        if override_if and cur in override_if:
            dbutils.widgets.remove(name)
            dbutils.widgets.text(name, default)
            return
        if (cur is None or cur.strip() == "") and default:
            dbutils.widgets.remove(name)
            dbutils.widgets.text(name, default)
    except Exception:
        dbutils.widgets.text(name, default)

dbutils.widgets.text("vs_endpoint_name", DEFAULT_VS_ENDPOINT)
dbutils.widgets.text("vs_index_name", DEFAULT_VS_INDEX)
dbutils.widgets.text("top_k", "10")
dbutils.widgets.text("model_name", "databricks-gpt-oss-20b")

_ensure_text_widget("vs_endpoint_name", DEFAULT_VS_ENDPOINT, override_if={"", "vs_fin_agent"})
_ensure_text_widget("vs_index_name", DEFAULT_VS_INDEX, override_if={""})
_ensure_text_widget("model_name", "databricks-gpt-oss-20b", override_if={"databricks-claude-sonnet-4-6", "databricks-claude-sonnet-4-5", "", None})

VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()
VS_INDEX_NAME = dbutils.widgets.get("vs_index_name").strip()
TOP_K = int(dbutils.widgets.get("top_k"))
MODEL_NAME = dbutils.widgets.get("model_name").strip()

# Python-level overrides for stale widget values
_STALE_MODEL_NAMES = {"databricks-claude-sonnet-4-6", "databricks-claude-sonnet-4-5", "", None}
if MODEL_NAME in _STALE_MODEL_NAMES:
    print(f"[override] Stale MODEL_NAME '{MODEL_NAME}' -> 'databricks-gpt-oss-20b'")
    MODEL_NAME = "databricks-gpt-oss-20b"

print("[config] VS endpoint:", VS_ENDPOINT_NAME)
print("[config] VS index:", VS_INDEX_NAME)
print("[config] TOP_K:", TOP_K)
print("[config] MODEL_NAME:", MODEL_NAME)

# COMMAND ----------

OBS_CATEGORIES = [
    {
        "name": "Factoring / receivables sales / securitization",
        "query": (
            "factoring, receivables sale, receivables transfer, securitization, "
            "derecognition of receivables, continuing involvement, recourse, repurchase obligation"
        ),
    },
    {
        "name": "Supplier finance / reverse factoring",
        "query": (
            "supplier finance, reverse factoring, payable finance, supply chain finance, "
            "structured payables, debt-like working capital financing"
        ),
    },
    {
        "name": "Guarantees / indemnities / sureties",
        "query": (
            "guarantees given, indemnities, sureties, parent company guarantees, letters of comfort, "
            "letters of credit, performance bonds, support commitments"
        ),
    },
    {
        "name": "Purchase / capex / take-or-pay commitments",
        "query": (
            "purchase commitments, capital expenditure commitments, investment commitments, "
            "take-or-pay obligations, non-cancellable contracts, committed future payments"
        ),
    },
    {
        "name": "Unconsolidated entities / JV / affiliate support",
        "query": (
            "unconsolidated subsidiaries, joint ventures, associates, special purpose vehicles, "
            "funding commitments, support obligations, guarantees to affiliates"
        ),
    },
    {
        "name": "Contingent liabilities / litigation / tax / regulatory",
        "query": (
            "contingent liabilities, litigation, tax disputes, regulatory claims, warranty obligations, "
            "environmental liabilities, remediation obligations, pending claims"
        ),
    },
    {
        "name": "Derecognition / consolidation-perimeter risk",
        "query": (
            "derecognition judgment, transfer of risk, off balance sheet treatment, "
            "consolidation perimeter, deconsolidation, accounting judgment"
        ),
    },
]

def _vs_extract_rows(vs_response: dict) -> list[dict]:
    """Parse Vector Search response into list of dicts."""
    if not isinstance(vs_response, dict):
        return []
    result = vs_response.get("result") or vs_response.get("results") or vs_response
    manifest = vs_response.get("manifest") or (result or {}).get("manifest") or {}
    data = (result or {}).get("data_array") or (result or {}).get("data") or []
    cols = manifest.get("columns") or []
    names = [c.get("name") for c in cols] if cols else None
    out = []
    for r in data:
        if names and len(names) == len(r):
            out.append({k: v for k, v in zip(names, r)})
        else:
            out.append({"row": r})
    return out


def retrieve_chunks(query: str, top_k: int) -> list[dict]:
    """Retrieve chunks from Vector Search and join back to Delta for chunk_text_en."""
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
    cols = [
        "chunk_id",
        "document_id",
        "fiscal_year",
        "page_start",
        "page_end",
        "chunk_type",
        "section_hint",
        "source_path_dbfs",
        "file_name",
    ]
    resp = index.similarity_search(
        query_text=query,
        columns=cols,
        num_results=top_k,
    )
    hits = _vs_extract_rows(resp)
    chunk_ids = [h.get("chunk_id") for h in hits if h.get("chunk_id")]
    if not chunk_ids:
        return []

    # Join back to Delta for full English text
    df = (
        spark.table(CHUNKS_TABLE)
        .filter(F.col("chunk_id").isin(chunk_ids))
        .select(
            "chunk_id",
            "document_id",
            "fiscal_year",
            "file_name",
            "source_path_dbfs",
            "page_start",
            "page_end",
            "chunk_type",
            "section_hint",
            "chunk_text_en",
        )
    )
    by_id = {r["chunk_id"]: r.asDict() for r in df.collect()}

    ordered = []
    for h in hits:
        cid = h.get("chunk_id")
        if cid in by_id:
            ordered.append(by_id[cid])
    return ordered


def call_ai_query(prompt: str) -> str:
    """Call the LLM via ai_query SQL function. Raises RuntimeError on empty/NULL."""
    prompt_sql = json.dumps(prompt)
    sql = f"SELECT ai_query('{MODEL_NAME}', {prompt_sql}) AS out"
    print(f"[ai_query] model={MODEL_NAME}, prompt_chars={len(prompt)}")
    raw = spark.sql(sql).collect()[0]["out"]
    if not raw or not raw.strip():
        raise RuntimeError(
            f"ai_query('{MODEL_NAME}', ...) returned empty/NULL. "
            f"Prompt was {len(prompt)} chars (~{len(prompt)//4} tokens). "
            "If this exceeds the model context window, reduce top_k or max_prompt_chars."
        )
    return raw.strip()


def build_extraction_prompt(category_name: str, chunks: list[dict], max_prompt_chars: int = 12000) -> str:
    """Build a prompt asking the LLM to extract year-by-year findings for an OBS category."""
    header = (
        "You are a conservative financial analyst specializing in off-balance-sheet items.\n"
        "Analyze the following evidence chunks and extract findings related to: "
        f"{category_name}\n\n"
        "For each fiscal year where you find relevant information, output exactly one line in this format:\n"
        "YYYY: <brief finding>\n\n"
        "Rules:\n"
        "- Use ONLY the evidence provided. Do not invent facts.\n"
        "- If no information is found for a year, do not output a line for that year.\n"
        "- Keep each finding to 1-2 sentences maximum.\n"
        "- Focus on amounts, changes, and key disclosures.\n\n"
        "EVIDENCE:\n"
    )
    overhead_per_chunk = 120
    budget = max_prompt_chars - len(header)
    chars_per_chunk = max(200, budget // max(len(chunks), 1) - overhead_per_chunk)

    lines = []
    for i, ch in enumerate(chunks, start=1):
        cite = f"[C{i}] FY{ch.get('fiscal_year', '?')} p{ch.get('page_start', '?')}-{ch.get('page_end', '?')} {ch.get('file_name', '')}"
        en = (ch.get("chunk_text_en") or "").strip()[:chars_per_chunk]
        lines.append(cite + "\n" + en)

    prompt = header + "\n\n---\n\n".join(lines)
    print(f"[build_extraction_prompt] category='{category_name}', {len(chunks)} chunks, {len(prompt)} chars (~{len(prompt)//4} tokens est.)")
    return prompt


def parse_year_findings(llm_output: str) -> dict:
    """Parse 'YYYY: finding' lines from LLM output. Returns dict[int, str]. Only accepts years 2000-2030."""
    findings = {}
    for line in llm_output.splitlines():
        line = line.strip()
        m = re.match(r"^(20[0-3]\d)\s*:\s*(.+)$", line)
        if m:
            year = int(m.group(1))
            if 2000 <= year <= 2030:
                findings[year] = m.group(2).strip()
    return findings

# COMMAND ----------
# MAGIC %md
# MAGIC ### Run Off-Balance-Sheet Extraction

# COMMAND ----------

results = {}  # {category_name: {year: finding}}

for idx, cat in enumerate(OBS_CATEGORIES, start=1):
    cat_name = cat["name"]
    cat_query = cat["query"]
    print(f"[{idx}/{len(OBS_CATEGORIES)}] Processing: {cat_name}")

    try:
        chunks = retrieve_chunks(cat_query, TOP_K)
        print(f"[{idx}/{len(OBS_CATEGORIES)}] Retrieved {len(chunks)} chunks for '{cat_name}'")

        if not chunks:
            print(f"[{idx}/{len(OBS_CATEGORIES)}] No chunks found for '{cat_name}', skipping LLM call")
            results[cat_name] = {}
            continue

        prompt = build_extraction_prompt(cat_name, chunks)
        llm_output = call_ai_query(prompt)
        findings = parse_year_findings(llm_output)
        print(f"[{idx}/{len(OBS_CATEGORIES)}] Parsed {len(findings)} year-findings for '{cat_name}'")

        results[cat_name] = findings

    except Exception as e:
        print(f"[{idx}/{len(OBS_CATEGORIES)}] ERROR for '{cat_name}': {e}")
        try:
            log_pipeline_error(
                ERRORS_TABLE,
                stage="obs_extraction",
                error=e,
                extra={"category": cat_name},
            )
        except Exception:
            pass
        results[cat_name] = {}

print(f"\n[done] Extraction complete for {len(results)} categories.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Off-Balance-Sheet Items by Fiscal Year

# COMMAND ----------

# Collect all years that appear in any category
all_years = set()
for findings in results.values():
    all_years.update(findings.keys())

if not all_years:
    print("[pivot] No findings extracted across any category. The table will be empty.")
    all_years = set(FISCAL_YEARS)

all_years = sorted(all_years)

# Build the pivot table
rows = []
for cat_name in [c["name"] for c in OBS_CATEGORIES]:
    findings = results.get(cat_name, {})
    row = {"Category": cat_name}
    for yr in all_years:
        row[str(yr)] = findings.get(yr, "")
    rows.append(row)

df_table = pd.DataFrame(rows)
df_table = df_table.set_index("Category")

# Display via Spark for Databricks notebook rendering
try:
    schema_fields = [T.StructField("Category", T.StringType(), nullable=False)]
    for yr in all_years:
        schema_fields.append(T.StructField(str(yr), T.StringType(), nullable=True))
    spark_schema = T.StructType(schema_fields)

    spark_rows = []
    for cat_name in [c["name"] for c in OBS_CATEGORIES]:
        findings = results.get(cat_name, {})
        r = [cat_name] + [findings.get(yr, "") for yr in all_years]
        spark_rows.append(r)

    spark_df = spark.createDataFrame(spark_rows, schema=spark_schema)
    display(spark_df)
except Exception as e:
    print(f"[pivot] Could not display Spark DataFrame: {e}")

# Also print as text for log readability
print(df_table.to_string())

# COMMAND ----------
# MAGIC %md
# MAGIC ### Narrative Summary

# COMMAND ----------

# Build compact text representation of findings for the narrative prompt
narrative_lines = []
for cat_name in [c["name"] for c in OBS_CATEGORIES]:
    findings = results.get(cat_name, {})
    if not findings:
        narrative_lines.append(f"{cat_name}: No data found.")
        continue
    for yr in sorted(findings.keys()):
        narrative_lines.append(f"{cat_name} | {yr}: {findings[yr]}")

data_text = "\n".join(narrative_lines)

if not any(results.get(c["name"], {}) for c in OBS_CATEGORIES):
    print("[narrative] No findings to summarize. Skipping narrative generation.")
else:
    try:
        narrative_prompt = (
            "You are a senior financial analyst.\n"
            "Below is a summary of off-balance-sheet items extracted from annual reports "
            "across multiple fiscal years.\n\n"
            "DATA:\n"
            f"{data_text}\n\n"
            "Write 2-3 paragraphs analyzing:\n"
            "1. Key trends over time for each category\n"
            "2. Any material changes or notable shifts\n"
            "3. Potential risks or concerns for stakeholders\n\n"
            "Be concise and factual. Do not invent information beyond what is provided."
        )
        narrative = call_ai_query(narrative_prompt)
        print("=== Narrative Summary ===\n")
        print(narrative)
    except Exception as e:
        print(f"[narrative] ERROR generating narrative: {e}")
        try:
            log_pipeline_error(
                ERRORS_TABLE,
                stage="obs_narrative",
                error=e,
            )
        except Exception:
            pass
