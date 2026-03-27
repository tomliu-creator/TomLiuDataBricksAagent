# Databricks notebook source
# NOTEBOOK FILE: 07_llm_query_demo.py
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 07 LLM Query Demo (English Q -> English A with French citations)
# MAGIC
# MAGIC Flow:
# MAGIC 1) English user question
# MAGIC 2) Retrieve top-K chunks via Vector Search over `chunk_text_en`
# MAGIC 3) Generate an English answer using an LLM
# MAGIC 4) Cite the original French chunks/pages in the output

# COMMAND ----------
# MAGIC %pip install -U databricks-vectorsearch

# COMMAND ----------

import json
import os
from datetime import datetime, timezone

import requests
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F

# If you restarted Python or ran cells out of order, helpers from `_utils` may be missing.
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

dbutils.widgets.text("question", "What are the key liquidity risks and debt maturities disclosed?")
dbutils.widgets.text("vs_endpoint_name", DEFAULT_VS_ENDPOINT)
dbutils.widgets.text("vs_index_name", DEFAULT_VS_INDEX)
dbutils.widgets.text("top_k", "8")
dbutils.widgets.text("filters", "")  # optional Vector Search filters

dbutils.widgets.dropdown("llm_mode", "ai_query", ["ai_query", "anthropic_messages"])
dbutils.widgets.text("model_name", "databricks-gpt-oss-20b")
dbutils.widgets.text("anthropic_proxy_endpoint_name", "anthropic")  # set if your workspace uses a different proxy endpoint
dbutils.widgets.text("temperature", "0.1")
dbutils.widgets.text("max_tokens", "900")

QUESTION = dbutils.widgets.get("question").strip()
_ensure_text_widget("vs_endpoint_name", DEFAULT_VS_ENDPOINT, override_if={"", "vs_fin_agent"})
_ensure_text_widget("vs_index_name", DEFAULT_VS_INDEX, override_if={""})
VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()
VS_INDEX_NAME = dbutils.widgets.get("vs_index_name").strip()
TOP_K = int(dbutils.widgets.get("top_k"))
FILTERS_RAW = dbutils.widgets.get("filters").strip() or None

LLM_MODE = dbutils.widgets.get("llm_mode").strip()
MODEL_NAME = dbutils.widgets.get("model_name").strip()
ANTHROPIC_PROXY_ENDPOINT = dbutils.widgets.get("anthropic_proxy_endpoint_name").strip()
TEMPERATURE = float(dbutils.widgets.get("temperature"))
MAX_TOKENS = int(dbutils.widgets.get("max_tokens"))

print("QUESTION:", QUESTION)
print("VS endpoint:", VS_ENDPOINT_NAME)
print("VS index:", VS_INDEX_NAME)
print("LLM_MODE:", LLM_MODE)
print("MODEL_NAME:", MODEL_NAME)
print("ANTHROPIC_PROXY_ENDPOINT:", ANTHROPIC_PROXY_ENDPOINT)

# COMMAND ----------


def _vs_extract_rows(vs_response: dict) -> list[dict]:
    if not isinstance(vs_response, dict):
        return []

    # Vector Search responses commonly look like:
    # { "manifest": { "columns": [...] }, "result": { "data_array": [...] } }
    # but some older examples have manifest nested under result.
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


def retrieve_chunks(question: str, top_k: int) -> list[dict]:
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
        query_text=question,
        columns=cols,
        num_results=top_k,
        filters=FILTERS_RAW,
    )
    hits = _vs_extract_rows(resp)
    chunk_ids = [h.get("chunk_id") for h in hits if h.get("chunk_id")]
    if not chunk_ids:
        return []

    # Join back to Delta for full French+English text and audit fields.
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
            "chunk_text_fr",
        )
    )
    by_id = {r["chunk_id"]: r.asDict() for r in df.collect()}

    ordered = []
    for h in hits:
        cid = h.get("chunk_id")
        if cid in by_id:
            ordered.append(by_id[cid])
    return ordered


def get_retrieval_diagnostics() -> dict:
    total_chunks = spark.table(CHUNKS_TABLE).count()
    translated_chunks = (
        spark.table(CHUNKS_TABLE)
        .filter(F.col("chunk_text_en").isNotNull() & (F.length(F.col("chunk_text_en")) > 0))
        .count()
    )
    indexed_pending = (
        spark.table(CHUNKS_TABLE)
        .filter(F.col("index_status") != "done")
        .count()
    )
    return {
        "chunks_total": total_chunks,
        "chunks_with_english": translated_chunks,
        "chunks_index_not_done": indexed_pending,
    }


def build_prompt(question: str, chunks: list[dict]) -> str:
    lines = []
    for i, ch in enumerate(chunks, start=1):
        cite = f"[C{i}] FY{ch['fiscal_year']} p{ch['page_start']}-{ch['page_end']} {ch['file_name']}"
        en = (ch.get("chunk_text_en") or "").strip()
        fr = (ch.get("chunk_text_fr") or "").strip()
        # Keep prompt size bounded; the Delta table retains full text for audit.
        en = en[:1600]
        fr = fr[:900]
        lines.append(
            cite
            + "\nEN:\n"
            + en
            + "\n\nFR (original):\n"
            + fr
        )

    return (
        "You are a conservative financial statement analyst.\n"
        "Answer in English. Do not invent facts.\n"
        "Use ONLY the evidence provided. If the evidence is insufficient, say so and list what is missing.\n"
        "Cite every important claim using the citation tags [C1], [C2], ...\n"
        "Do not cite sources you did not use.\n\n"
        f"QUESTION:\n{question}\n\n"
        "EVIDENCE:\n"
        + "\n\n---\n\n".join(lines)
    )


class _AnthropicProxyNotFound(RuntimeError):
    pass


def call_llm_anthropic_messages(prompt: str) -> str:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    host = ctx.browserHostName().get()
    token = None
    try:
        token = ctx.apiToken().get()
    except Exception:
        token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError("Unable to obtain a Databricks token. Set env var DATABRICKS_TOKEN or enable context apiToken().")

    if not ANTHROPIC_PROXY_ENDPOINT:
        raise _AnthropicProxyNotFound("Anthropic proxy endpoint name is empty.")

    url = f"https://{host}/serving-endpoints/{ANTHROPIC_PROXY_ENDPOINT}/v1/messages"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 404:
        raise _AnthropicProxyNotFound(
            f"Anthropic proxy endpoint '{ANTHROPIC_PROXY_ENDPOINT}' not found (404)."
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic proxy call failed: {resp.status_code} {resp.text[:2000]}")
    data = resp.json()
    blocks = data.get("content") or []
    texts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("text"):
            texts.append(b["text"])
    return "".join(texts).strip()


def call_llm_ai_query(prompt: str) -> str:
    # ai_query(endpoint, request_string, modelParameters => named_struct(...))
    prompt_sql = json.dumps(prompt)
    sql = (
        f"SELECT ai_query('{MODEL_NAME}', {prompt_sql}, "
        f"modelParameters => named_struct('max_tokens', {MAX_TOKENS}, 'temperature', {TEMPERATURE})) AS out"
    )
    return spark.sql(sql).collect()[0]["out"]


# COMMAND ----------

retrieved = retrieve_chunks(QUESTION, TOP_K)
print("Retrieved chunks:", len(retrieved))
if not retrieved:
    diagnostics = get_retrieval_diagnostics()
    raise RuntimeError(
        "No chunks retrieved. "
        f"Diagnostics: {diagnostics}. "
        "Most likely causes: the Delta Sync index has not been synced after creation, "
        "the index name/endpoint is pointing to the wrong index, or the index is still empty. "
        "Rerun 05_vector_index and confirm it runs a triggered sync on the same endpoint/index."
    )

prompt = build_prompt(QUESTION, retrieved)

answer = None
try:
    if LLM_MODE == "anthropic_messages":
        try:
            answer = call_llm_anthropic_messages(prompt)
        except _AnthropicProxyNotFound as e:
            print("Anthropic proxy unavailable:", str(e))
            print("Falling back to ai_query with model_name =", MODEL_NAME)
            answer = call_llm_ai_query(prompt)
    else:
        answer = call_llm_ai_query(prompt)
except Exception as e:
    # Don't mask the root cause if `_utils` wasn't loaded in this Python session.
    try:
        log_pipeline_error(ERRORS_TABLE, stage="llm_query", error=e, extra={"llm_mode": LLM_MODE, "model": MODEL_NAME})
    except Exception:
        print("LLM call failed (unable to log to pipeline_errors):", repr(e))
    raise

print("\n=== Answer (English) ===\n")
print(answer)

print("\n=== Citations (French evidence pointers) ===\n")
for i, ch in enumerate(retrieved, start=1):
    print(
        f"[C{i}] FY{ch['fiscal_year']} p{ch['page_start']}-{ch['page_end']} | {ch['file_name']} | {ch['source_path_dbfs']}"
    )
