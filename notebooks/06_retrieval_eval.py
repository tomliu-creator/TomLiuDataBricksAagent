# Databricks notebook source
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
from databricks.vector_search.client import VectorSearchClient

dbutils.widgets.text("vs_endpoint_name", DEFAULT_VS_ENDPOINT)
dbutils.widgets.text("vs_index_name", DEFAULT_VS_INDEX)
dbutils.widgets.text("top_k", "8")
dbutils.widgets.text("filters", "")  # e.g. {"fiscal_year >= 2020": "..."} is index-dependent; keep empty for pilot

# Optional: generate a draft answer for each query via ai_query(model, prompt).
dbutils.widgets.text("answer_model_name", "")  # e.g. databricks-claude-sonnet-4-6 (if enabled)

VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()
VS_INDEX_NAME = dbutils.widgets.get("vs_index_name").strip()
TOP_K = int(dbutils.widgets.get("top_k"))
FILTERS_RAW = dbutils.widgets.get("filters").strip() or None
ANSWER_MODEL = dbutils.widgets.get("answer_model_name").strip() or None

print("VS endpoint:", VS_ENDPOINT_NAME)
print("VS index:", VS_INDEX_NAME)
print("TOP_K:", TOP_K)
print("ANSWER_MODEL:", ANSWER_MODEL or "(disabled)")

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
    manifest = result.get("manifest") or {}
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
    out = spark.sql(
        f\"\"\"SELECT ai_query('{model_name}', {json.dumps(prompt)}) AS answer\"\"\"
    ).collect()[0]["answer"]
    return out


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
    if ANSWER_MODEL:
        try:
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
            "model_name": ANSWER_MODEL,
            "notes": None,
        }
    )

spark.createDataFrame(rows_to_write).write.mode("append").saveAsTable(RETRIEVAL_EVAL_TABLE)

display(spark.table(RETRIEVAL_EVAL_TABLE).orderBy(F.col("run_ts").desc()).limit(20))
