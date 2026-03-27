# Databricks notebook source
# NOTEBOOK FILE: 06_retrieval_eval.py
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 06 Retrieval Evaluation (Benchmark Questions)
# MAGIC
# MAGIC Runs a small benchmark suite of realistic English financial-statement questions.
# MAGIC Stores retrieval results (and optional answer drafts) into `retrieval_eval_runs`.

# COMMAND ----------
# MAGIC %pip install -U databricks-vectorsearch

# COMMAND ----------

import uuid
import json
from datetime import datetime, timezone
import os
import requests
from pyspark.sql import types as T
from pyspark.sql import functions as F
from databricks.vector_search.client import VectorSearchClient

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
dbutils.widgets.text("top_k", "8")
dbutils.widgets.text("filters", "")  # e.g. {"fiscal_year >= 2020": "..."} is index-dependent; keep empty for pilot

# Optional: generate a draft answer for each query.
dbutils.widgets.dropdown("answer_mode", "anthropic_messages", ["anthropic_messages", "ai_query", "disabled"])
dbutils.widgets.text("answer_model_name", "databricks-claude-sonnet-4-6")  # change if not enabled in workspace

VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()
VS_INDEX_NAME = dbutils.widgets.get("vs_index_name").strip()
TOP_K = int(dbutils.widgets.get("top_k"))
FILTERS_RAW = dbutils.widgets.get("filters").strip() or None
ANSWER_MODE = dbutils.widgets.get("answer_mode").strip()
ANSWER_MODEL = dbutils.widgets.get("answer_model_name").strip() or None

# If you previously used the template endpoint name, auto-switch to the configured default.
_ensure_text_widget("vs_endpoint_name", DEFAULT_VS_ENDPOINT, override_if={"", "vs_fin_agent"})
VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()

# If an old run left answers disabled / unset, switch back to enabled defaults.
_ensure_text_widget("answer_model_name", "databricks-claude-sonnet-4-6", override_if={""})
try:
    if dbutils.widgets.get("answer_mode") in ("", "disabled"):
        dbutils.widgets.remove("answer_mode")
        dbutils.widgets.dropdown("answer_mode", "anthropic_messages", ["anthropic_messages", "ai_query", "disabled"])
except Exception:
    pass
ANSWER_MODE = dbutils.widgets.get("answer_mode").strip()
ANSWER_MODEL = dbutils.widgets.get("answer_model_name").strip() or None

print("VS endpoint:", VS_ENDPOINT_NAME)
print("VS index:", VS_INDEX_NAME)
print("TOP_K:", TOP_K)
print("ANSWER_MODE:", ANSWER_MODE)
print("ANSWER_MODEL:", ANSWER_MODEL or "(unset)")

# COMMAND ----------

BENCHMARK_QUESTIONS = [
    "Summarize the company’s debt maturity profile and any refinancing risk disclosed.",
    "What covenants are disclosed for the main financing facilities, and were there any covenant breaches or waivers?",
    "Describe lease liabilities and major lease commitments (IFRS 16), including maturity if disclosed.",
    "Is there any discussion of factoring, securitization, or off-balance-sheet financing exposure? Provide details and risks.",
    "What goodwill balances are reported and what impairment tests or key assumptions are disclosed?",
    "Did the auditor include any emphasis-of-matter, key audit matters, or going-concern language? Summarize the most important items.",
    "Describe acquisition-related disclosures (business combinations) and how they impacted debt, goodwill, and cash flows.",
    "Summarize liquidity risk management and any disclosures about cash, credit lines, and headroom.",
    "Identify any major provisions and contingent liabilities (litigation, restructuring, guarantees) and how they evolved.",
    "Explain any significant changes in accounting policies or estimates that materially affect comparability across years.",
]


def _vs_extract_rows(vs_response: dict) -> list[dict]:
    """
    Normalize similarity_search responses into a list of dicts.
    The Vector Search API returns a 'manifest' + 'data_array' pattern.
    """
    if not vs_response:
        return []
    result = vs_response.get("result") or vs_response.get("results") or vs_response
    if not isinstance(result, dict):
        return []
    data = result.get("data_array") or result.get("data") or []
    manifest = vs_response.get("manifest") or result.get("manifest") or {}
    cols = manifest.get("columns") or []
    col_names = [c.get("name") for c in cols] if cols else None
    rows = []
    for r in data:
        if col_names and len(col_names) == len(r):
            rows.append({k: v for k, v in zip(col_names, r)})
        else:
            rows.append({"row": r})
    return rows


def _draft_answer_with_ai_query(model_name: str, question: str, retrieved_chunks: list[dict]) -> str:
    # Conservative, citation-driven prompt; keep short for eval runs.
    evidence_lines = []
    for i, ch in enumerate(retrieved_chunks[:8], start=1):
        cite = f"[C{i}] FY{ch.get('fiscal_year')} p{ch.get('page_start')}-{ch.get('page_end')}"
        evidence_lines.append(cite + " " + (ch.get("chunk_text_en") or "")[:1200])
    prompt = (
        "You are a financial statement analyst. Answer in English and do not invent facts.\n"
        "Use ONLY the evidence below. If evidence is insufficient, say so.\n"
        "Cite every important claim using [C#].\n\n"
        f"QUESTION:\n{question}\n\n"
        "EVIDENCE:\n" + "\n\n".join(evidence_lines)
    )
    # ai_query(endpoint, request_string)
    prompt_sql = json.dumps(prompt)
    sql = f"SELECT ai_query('{model_name}', {prompt_sql}) AS answer"
    out = spark.sql(sql).collect()[0]["answer"]
    return out


def _draft_answer_with_anthropic_messages(model_name: str, question: str, retrieved_chunks: list[dict]) -> str:
    evidence_lines = []
    for i, ch in enumerate(retrieved_chunks[:8], start=1):
        cite = f"[C{i}] FY{ch.get('fiscal_year')} p{ch.get('page_start')}-{ch.get('page_end')}"
        evidence_lines.append(cite + " " + (ch.get("chunk_text_en") or "")[:1200])
    prompt = (
        "You are a financial statement analyst. Answer in English and do not invent facts.\n"
        "Use ONLY the evidence below. If evidence is insufficient, say so.\n"
        "Cite every important claim using [C#].\n\n"
        f"QUESTION:\n{question}\n\n"
        "EVIDENCE:\n" + "\n\n".join(evidence_lines)
    )

    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    host = ctx.browserHostName().get()
    token = None
    try:
        token = ctx.apiToken().get()
    except Exception:
        token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError("Unable to obtain a Databricks token for Anthropic proxy calls.")

    url = f"https://{host}/serving-endpoints/anthropic/v1/messages"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": model_name,
        "max_tokens": 650,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic proxy call failed: {resp.status_code} {resp.text[:2000]}")
    data = resp.json()
    blocks = data.get("content") or []
    texts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("text"):
            texts.append(b["text"])
    return "".join(texts).strip()


# COMMAND ----------

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)

run_ts = datetime.now(timezone.utc)

columns = [
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

rows_to_write = []
for q in BENCHMARK_QUESTIONS:
    vs_resp = index.similarity_search(
        query_text=q,
        columns=columns,
        num_results=TOP_K,
        filters=FILTERS_RAW,
    )
    retrieved = _vs_extract_rows(vs_resp)

    # Join back to the Delta table for full texts (kept out of the vector index).
    chunk_ids = [r.get("chunk_id") for r in retrieved if r.get("chunk_id")]
    chunk_texts = {}
    if chunk_ids:
        df = (
            spark.table(CHUNKS_TABLE)
            .filter(F.col("chunk_id").isin(chunk_ids))
            .select("chunk_id", "chunk_text_en", "chunk_text_fr", "fiscal_year", "page_start", "page_end")
        )
        chunk_texts = {r["chunk_id"]: r.asDict() for r in df.collect()}

    retrieved_enriched = []
    for r in retrieved:
        cid = r.get("chunk_id")
        extra = chunk_texts.get(cid) if cid else None
        out = dict(r)
        if extra:
            out["chunk_text_en"] = extra.get("chunk_text_en")
            out["chunk_text_fr"] = extra.get("chunk_text_fr")
        retrieved_enriched.append(out)

    answer = None
    if ANSWER_MODE != "disabled" and ANSWER_MODEL:
        try:
            if ANSWER_MODE == "anthropic_messages":
                answer = _draft_answer_with_anthropic_messages(ANSWER_MODEL, q, retrieved_enriched)
            else:
                answer = _draft_answer_with_ai_query(ANSWER_MODEL, q, retrieved_enriched)
        except Exception as e:
            log_pipeline_error(ERRORS_TABLE, stage="retrieval_eval_answer", error=e, extra={"question": q})
            answer = None

    rows_to_write.append(
        {
            "run_id": str(uuid.uuid4()),
            "run_ts": run_ts,
            "query_text": q,
            "top_k": TOP_K,
            "filters": FILTERS_RAW,
            "retrieved_json": json.dumps(
                {"vector_search_raw": vs_resp, "retrieved": retrieved_enriched},
                ensure_ascii=True,
            ),
            "answer_en": answer,
            "model_name": ANSWER_MODEL if ANSWER_MODE != "disabled" else None,
            "notes": None,
        }
    )

eval_schema = T.StructType(
    [
        T.StructField("run_id", T.StringType(), nullable=False),
        T.StructField("run_ts", T.TimestampType(), nullable=False),
        T.StructField("query_text", T.StringType(), nullable=False),
        T.StructField("top_k", T.IntegerType(), nullable=False),
        T.StructField("filters", T.StringType(), nullable=True),
        T.StructField("retrieved_json", T.StringType(), nullable=True),
        T.StructField("answer_en", T.StringType(), nullable=True),
        T.StructField("model_name", T.StringType(), nullable=True),
        T.StructField("notes", T.StringType(), nullable=True),
    ]
)

spark.createDataFrame(rows_to_write, schema=eval_schema).write.mode("append").saveAsTable(RETRIEVAL_EVAL_TABLE)

display(spark.table(RETRIEVAL_EVAL_TABLE).orderBy(F.col("run_ts").desc()).limit(20))
