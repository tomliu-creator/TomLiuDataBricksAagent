# Databricks notebook source
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
# MAGIC %pip install -U databricks-vectorsearch anthropic

# COMMAND ----------

import json
import os
from datetime import datetime, timezone

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

dbutils.widgets.text("question", "What are the key liquidity risks and debt maturities disclosed?")
dbutils.widgets.text("vs_endpoint_name", DEFAULT_VS_ENDPOINT)
dbutils.widgets.text("vs_index_name", DEFAULT_VS_INDEX)
dbutils.widgets.text("top_k", "8")
dbutils.widgets.text("filters", "")  # optional Vector Search filters

dbutils.widgets.dropdown("llm_mode", "anthropic_messages", ["anthropic_messages", "ai_query"])
dbutils.widgets.text("model_name", "databricks-claude-sonnet-4-6")
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
TEMPERATURE = float(dbutils.widgets.get("temperature"))
MAX_TOKENS = int(dbutils.widgets.get("max_tokens"))

print("QUESTION:", QUESTION)
print("VS endpoint:", VS_ENDPOINT_NAME)
print("VS index:", VS_INDEX_NAME)
print("LLM_MODE:", LLM_MODE)
print("MODEL_NAME:", MODEL_NAME)

# COMMAND ----------


def _vs_extract_rows(vs_response: dict) -> list[dict]:
    result = vs_response.get("result") or vs_response.get("results") or vs_response
    data = (result or {}).get("data_array") or []
    cols = ((result or {}).get("manifest") or {}).get("columns") or []
    names = [c.get("name") for c in cols] if cols else None
    out = []
    for r in data:
        if names and len(names) == len(r):
            out.append({k: v for k, v in zip(names, r)})
        else:
            out.append({"row": r})
    return out


def retrieve_chunks(question: str, top_k: int) -> list[dict]:
    vsc = VectorSearchClient()
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


def call_llm_anthropic_messages(prompt: str) -> str:
    import anthropic

    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    host = ctx.browserHostName().get()
    token = None
    try:
        token = ctx.apiToken().get()
    except Exception:
        token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError("Unable to obtain a Databricks token. Set env var DATABRICKS_TOKEN or enable context apiToken().")

    client = anthropic.Anthropic(
        api_key="unused",
        base_url=f"https://{host}/serving-endpoints/anthropic",
        default_headers={"Authorization": f"Bearer {token}"},
    )

    msg = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    # Anthropic SDK returns structured content blocks
    return "".join([blk.text for blk in msg.content if getattr(blk, "text", None)])


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
    raise RuntimeError("No chunks retrieved. Check that the vector index exists and contains translated chunks.")

prompt = build_prompt(QUESTION, retrieved)

answer = None
try:
    if LLM_MODE == "anthropic_messages":
        answer = call_llm_anthropic_messages(prompt)
    else:
        answer = call_llm_ai_query(prompt)
except Exception as e:
    log_pipeline_error(ERRORS_TABLE, stage="llm_query", error=e, extra={"llm_mode": LLM_MODE, "model": MODEL_NAME})
    raise

print("\n=== Answer (English) ===\n")
print(answer)

print("\n=== Citations (French evidence pointers) ===\n")
for i, ch in enumerate(retrieved, start=1):
    print(
        f"[C{i}] FY{ch['fiscal_year']} p{ch['page_start']}-{ch['page_end']} | {ch['file_name']} | {ch['source_path_dbfs']}"
    )
