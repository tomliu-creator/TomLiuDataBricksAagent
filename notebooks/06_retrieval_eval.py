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

import json
import os
import uuid
from datetime import datetime, timezone

import requests
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F
from pyspark.sql import types as T


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


def _ensure_dropdown_widget(name: str, default: str, choices: list[str]):
    try:
        cur = dbutils.widgets.get(name)
        if cur not in choices:
            dbutils.widgets.remove(name)
            dbutils.widgets.dropdown(name, default, choices)
    except Exception:
        dbutils.widgets.dropdown(name, default, choices)


dbutils.widgets.text("vs_endpoint_name", DEFAULT_VS_ENDPOINT)
dbutils.widgets.text("vs_index_name", DEFAULT_VS_INDEX)
dbutils.widgets.text("top_k", "8")
dbutils.widgets.text("filters", "")  # optional Vector Search filters

# Optional: generate a draft answer for each query.
_ensure_dropdown_widget("answer_mode", "ai_query", ["ai_query", "anthropic_messages", "disabled"])
_ensure_text_widget("answer_model_name", "databricks-gpt-oss-20b")
_ensure_text_widget("anthropic_proxy_endpoint_name", "anthropic")

_ensure_text_widget("vs_endpoint_name", DEFAULT_VS_ENDPOINT, override_if={"", "vs_fin_agent"})

VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()
VS_INDEX_NAME = dbutils.widgets.get("vs_index_name").strip()
TOP_K = int(dbutils.widgets.get("top_k"))
FILTERS_RAW = dbutils.widgets.get("filters").strip() or None
ANSWER_MODE = dbutils.widgets.get("answer_mode").strip()
ANSWER_MODEL = dbutils.widgets.get("answer_model_name").strip() or None
ANTHROPIC_PROXY_ENDPOINT = dbutils.widgets.get("anthropic_proxy_endpoint_name").strip()

# Handle stale widget values from older runs.
_STALE_MODEL_NAMES = {"databricks-claude-sonnet-4-6", "databricks-claude-sonnet-4-5", "", None}
if ANSWER_MODEL in _STALE_MODEL_NAMES:
    print(f"[override] Stale ANSWER_MODEL '{ANSWER_MODEL}' -> 'databricks-gpt-oss-20b'")
    ANSWER_MODEL = "databricks-gpt-oss-20b"
if ANSWER_MODE == "anthropic_messages":
    print("[override] ANSWER_MODE 'anthropic_messages' -> 'ai_query' (proxy not configured)")
    ANSWER_MODE = "ai_query"

print("VS endpoint:", VS_ENDPOINT_NAME)
print("VS index:", VS_INDEX_NAME)
print("TOP_K:", TOP_K)
print("ANSWER_MODE:", ANSWER_MODE)
print("ANSWER_MODEL:", ANSWER_MODEL or "(unset)")
print("ANTHROPIC_PROXY_ENDPOINT:", ANTHROPIC_PROXY_ENDPOINT)


BENCHMARK_QUESTIONS = [
    "Summarize the company’s debt maturity profile and any refinancing risk disclosed.",
    "What covenants are disclosed for the main financing facilities, and were there any covenant breaches, cures, resets, or waivers?",
    "Describe lease liabilities and major lease commitments (IFRS 16), including maturity if disclosed.",
    "Is there any discussion of factoring, receivables sales, securitization, or off-balance-sheet financing exposure? Provide details, accounting treatment, retained risk, and credit implications.",
    "Is there any supplier finance, reverse factoring, payable financing, or similar working-capital arrangement? Explain the size, structure, and debt-like risk.",
    "Identify all guarantees, indemnities, letters of comfort, letters of credit, performance bonds, or similar support arrangements. Summarize size, beneficiaries, triggers, and risk.",
    "Are there any unconsolidated entities, joint ventures, associates, or special-purpose vehicles that create funding commitments, guarantees, or contingent obligations for the group?",
    "What non-cancellable purchase commitments, take-or-pay agreements, capex commitments, or other contractual obligations are disclosed outside reported borrowings?",
    "Summarize liquidity risk management and any disclosures about cash, credit lines, headroom, and dependence on receivables monetization or short-term facilities.",
    "Identify major provisions and contingent liabilities, including litigation, restructuring, tax, environmental, warranty, and guarantee-related exposures, and explain how they evolved.",
    "What goodwill balances are reported and what impairment tests or key assumptions are disclosed?",
    "Did the auditor include any emphasis-of-matter, key audit matters, or going-concern language? Summarize the most important items.",
    "Describe acquisition-related disclosures (business combinations) and how they impacted debt, goodwill, contingent consideration, guarantees, and cash flows.",
    "Explain any significant changes in accounting policies, estimates, or consolidation perimeter that materially affect comparability or could shift obligations on or off the balance sheet."
]

def _vs_extract_rows(vs_response: dict) -> list[dict]:
    if not isinstance(vs_response, dict):
        return []
    result = vs_response.get("result") or vs_response.get("results") or vs_response
    if not isinstance(result, dict):
        return []
    manifest = vs_response.get("manifest") or result.get("manifest") or {}
    data = result.get("data_array") or result.get("data") or []
    cols = manifest.get("columns") or []
    names = [c.get("name") for c in cols] if cols else None
    out = []
    for r in data:
        if names and isinstance(r, list) and len(names) == len(r):
            out.append({k: v for k, v in zip(names, r)})
        else:
            out.append({"row": r})
    return out


def _draft_answer_with_ai_query(
    model_name: str,
    question: str,
    retrieved_chunks: list[dict],
    max_prompt_chars: int = 12000,
) -> str:
    header = (
        "You are a financial statement analyst. Answer in English and do not invent facts.\n"
        "Use ONLY the evidence below. If evidence is insufficient, say so.\n"
        "Cite every important claim using [C#].\n\n"
        f"QUESTION:\n{question}\n\n"
        "EVIDENCE:\n"
    )
    chunks = retrieved_chunks[:8]
    budget = max_prompt_chars - len(header)
    chars_per_chunk = max(200, budget // max(len(chunks), 1) - 120)

    evidence_lines = []
    for i, ch in enumerate(chunks, start=1):
        cite = f"[C{i}] FY{ch.get('fiscal_year')} p{ch.get('page_start')}-{ch.get('page_end')}"
        evidence_lines.append(cite + " " + ((ch.get("chunk_text_en") or "")[:chars_per_chunk]))
    prompt = header + "\n\n".join(evidence_lines)

    prompt_sql = json.dumps(prompt)
    sql = f"SELECT ai_query('{model_name}', {prompt_sql}) AS answer"
    out = spark.sql(sql).collect()[0]["answer"]
    if not out or not out.strip():
        raise RuntimeError(
            f"ai_query('{model_name}', ...) returned empty/NULL. Prompt was {len(prompt)} chars (~{len(prompt)//4} tokens). "
            "If this exceeds the model context window, reduce top_k or max_prompt_chars."
        )
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

    if not ANTHROPIC_PROXY_ENDPOINT:
        raise RuntimeError("Anthropic proxy endpoint name is empty.")

    url = f"https://{host}/serving-endpoints/{ANTHROPIC_PROXY_ENDPOINT}/v1/messages"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": model_name,
        "max_tokens": 650,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 404:
        raise RuntimeError(f"Anthropic proxy endpoint '{ANTHROPIC_PROXY_ENDPOINT}' not found (404).")
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic proxy call failed: {resp.status_code} {resp.text[:2000]}")
    data = resp.json()
    blocks = data.get("content") or []
    texts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("text"):
            texts.append(b["text"])
    return "".join(texts).strip()


vsc = VectorSearchClient(disable_notice=True)
index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Pipeline Health Check

# COMMAND ----------

_total = spark.table(CHUNKS_TABLE).count()
_has_en = spark.table(CHUNKS_TABLE).filter(
    F.col("chunk_text_en").isNotNull() & (F.length(F.col("chunk_text_en")) > 0)
).count()
_idx_done = spark.table(CHUNKS_TABLE).filter(F.col("index_status") == "done").count()
print(f"[health] Total chunks: {_total}")
print(f"[health] Chunks with English text: {_has_en} ({100*_has_en//_total if _total else 0}%)")
print(f"[health] Missing English: {_total - _has_en}")
print(f"[health] Index status=done: {_idx_done}")
try:
    _idx_info = index.describe()
    _status = _idx_info.get("status", {}).get("detailed_state", _idx_info.get("status", {}))
    _num_indexed = _idx_info.get("status", {}).get("indexed_row_count", "?")
    print(f"[health] VS index state: {_status}")
    print(f"[health] VS indexed rows: {_num_indexed}")
except Exception as _e:
    print(f"[health] Could not describe index: {_e}")

# COMMAND ----------

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
for qi, q in enumerate(BENCHMARK_QUESTIONS, start=1):
    print(f"[{qi}/{len(BENCHMARK_QUESTIONS)}] {q[:80]}...")
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
            print(f"[ERROR] Q{qi}: {type(e).__name__}: {e}")
            log_pipeline_error(ERRORS_TABLE, stage="retrieval_eval_answer", error=e, extra={"question": q})
            answer = None

    rows_to_write.append(
        {
            "run_id": str(uuid.uuid4()),
            "run_ts": run_ts,
            "query_text": q,
            "top_k": TOP_K,
            "filters": FILTERS_RAW,
            "retrieved_json": json.dumps({"vector_search_raw": vs_resp, "retrieved": retrieved_enriched}, ensure_ascii=True),
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

n_ok = sum(1 for r in rows_to_write if r["answer_en"])
n_fail = len(rows_to_write) - n_ok
print(f"\n=== Run complete: {n_ok}/{len(rows_to_write)} answers generated, {n_fail} failed ===\n")

display(
    spark.table(RETRIEVAL_EVAL_TABLE)
    .filter(F.col("run_ts") == run_ts)
    .orderBy(F.col("query_text"))
)

