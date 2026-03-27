# Databricks notebook source
# NOTEBOOK FILE: 04_chunk_translate.py
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 04 Chunk + Translate (French evidence, English retrieval)
# MAGIC
# MAGIC - Builds `document_chunks` from parsed pages (French).
# MAGIC - Translates `chunk_text_fr` to English using `ai_translate(chunk_text_fr, 'en')`.
# MAGIC
# MAGIC Retrieval later runs over `chunk_text_en`, while citations always point back to
# MAGIC `chunk_text_fr` plus `(fiscal_year, page_start, page_end, source_path_dbfs)`.

# COMMAND ----------

import hashlib
import re

MAX_DOCS = None  # for debugging

# Chunking parameters (page-aware; stable citations)
MIN_CHARS = 2500
MAX_CHARS = 6000
MAX_PAGES = 5
OVERLAP_PAGES = 1

# Translation batching
TRANSLATE_LIMIT = None  # set to an int to limit translated chunks per run


def _sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _pick_section_hint(text: str) -> str | None:
    if not text:
        return None
    # Heuristic: first heading-like line (ALL CAPS / numbered section) within first ~60 lines.
    lines = [ln.strip() for ln in text.splitlines()[:60] if ln.strip()]
    for ln in lines:
        if re.match(r"^\d+(\.\d+)*\s+.{5,}$", ln):
            return ln[:200]
        if re.match(r"^[A-Z0-9][A-Z0-9\s\-\–\—'’(),.:/]{10,}$", ln) and len(ln) <= 140:
            return ln[:200]
    return None


def build_chunks_for_doc(pages: list[dict]) -> list[dict]:
    chunks = []
    n = len(pages)
    i = 0
    chunk_idx = 0
    while i < n:
        start = i
        end = i - 1
        chars = 0

        while end + 1 < n and (end - start + 1) < MAX_PAGES:
            nxt = pages[end + 1]
            nxt_chars = len(nxt["page_text_fr"] or "")
            if chars < MIN_CHARS:
                end += 1
                chars += nxt_chars
                continue
            if chars + nxt_chars <= MAX_CHARS:
                end += 1
                chars += nxt_chars
                continue
            break

        if end < start:
            end = start

        chunk_pages = pages[start : end + 1]
        page_start = chunk_pages[0]["page_num"]
        page_end = chunk_pages[-1]["page_num"]

        # Page markers make debugging + audit easier without losing original text.
        parts = []
        for p in chunk_pages:
            parts.append(f"[PAGE {p['page_num']}]\\n{p['page_text_fr'] or ''}".strip())
        chunk_text_fr = "\\n\\n".join(parts).strip()

        chunk_type = "sliding_pages"
        chunk_id = _sha256_hex(f"{pages[0]['document_id']}||{page_start}||{page_end}||{chunk_idx}||{chunk_type}")
        section_hint = _pick_section_hint(chunk_text_fr)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": pages[0]["document_id"],
                "fiscal_year": pages[0]["fiscal_year"],
                "file_name": pages[0]["file_name"],
                "source_path_dbfs": pages[0]["source_path_dbfs"],
                "source_path_local": pages[0]["source_path_local"],
                "chunk_index": chunk_idx,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_type": chunk_type,
                "section_hint": section_hint,
                "chunk_text_fr": chunk_text_fr,
                "chunk_sha_fr": _sha256_hex(chunk_text_fr),
                "chunk_char_len_fr": len(chunk_text_fr),
            }
        )

        chunk_idx += 1
        i = (end + 1) - OVERLAP_PAGES
        if i <= start:
            i = end + 1

    return chunks


# COMMAND ----------

pending_docs = (
    spark.table(INVENTORY_TABLE)
    .filter((F.col("source_name") == SOURCE_NAME) & (F.col("is_present") == True))
    .filter(F.col("parse_status") == "done")
    .filter(F.col("chunk_status").isin(["pending", "error"]))
    .select("document_id", "fiscal_year", "file_name", "file_path_dbfs", "file_path_local")
    .orderBy("fiscal_year", "file_name")
)

pending_doc_count = pending_docs.count()
print("Documents pending chunking:", pending_doc_count)
if pending_doc_count == 0:
    show_validation_snapshot(CATALOG, SCHEMA)
else:
    display(pending_docs)

# COMMAND ----------

docs = pending_docs.limit(MAX_DOCS).collect() if (MAX_DOCS and pending_doc_count > 0) else pending_docs.collect()

# Rebuild strategy: remove any existing chunks for documents we're about to chunk.
# This prevents stale chunks lingering after PDF changes or partial previous runs.
if docs:
    doc_ids_to_rechunk = [(d["document_id"],) for d in docs]
    spark.createDataFrame(doc_ids_to_rechunk, ["document_id"]).createOrReplaceTempView("docs_to_rechunk")
    spark.sql(f"""
    DELETE FROM {CHUNKS_TABLE}
    WHERE document_id IN (SELECT document_id FROM docs_to_rechunk)
    """)

all_chunks = []
ok_docs = []
failed_docs = []  # (document_id, chunk_error)

for d in docs:
    doc_id = d["document_id"]
    try:
        pages_df = (
            spark.table(PAGES_TABLE)
            .filter(F.col("document_id") == doc_id)
            .select(
                "document_id",
                "fiscal_year",
                "file_name",
                "source_path_dbfs",
                "source_path_local",
                "page_num",
                "page_text_fr",
            )
            .orderBy("page_num")
        )
        pages = [r.asDict() for r in pages_df.collect()]
        if not pages:
            raise ValueError("No pages found for document_id (expected parse stage to populate document_pages).")

        all_chunks.extend(build_chunks_for_doc(pages))
        ok_docs.append(doc_id)
    except Exception as e:
        failed_docs.append((doc_id, str(e)[:4000]))
        log_pipeline_error(ERRORS_TABLE, stage="chunk_build", document_id=doc_id, error=e)

print("Chunked OK docs:", len(ok_docs))
print("Chunked failed docs:", len(failed_docs))

if pending_doc_count > 0 and len(all_chunks) == 0:
    raise RuntimeError("No chunks were created. Check document_pages content and chunking parameters.")

# COMMAND ----------

if all_chunks:
    df_new_chunks = (
        spark.createDataFrame(all_chunks)
        .withColumn("chunk_text_en", F.lit(None).cast("string"))
        .withColumn("chunk_char_len_en", F.lit(None).cast("int"))
        .withColumn("chunk_ts", F.current_timestamp())
    )
    df_new_chunks.createOrReplaceTempView("new_chunks")

    # MERGE with change detection on chunk_sha_fr to reset translation+index only when needed.
    spark.sql(f"""
    MERGE INTO {CHUNKS_TABLE} t
    USING new_chunks s
    ON t.chunk_id = s.chunk_id
    WHEN MATCHED AND (t.chunk_sha_fr IS NULL OR t.chunk_sha_fr <> s.chunk_sha_fr) THEN UPDATE SET
      t.document_id = s.document_id,
      t.fiscal_year = s.fiscal_year,
      t.file_name = s.file_name,
      t.source_path_dbfs = s.source_path_dbfs,
      t.source_path_local = s.source_path_local,
      t.chunk_index = s.chunk_index,
      t.page_start = s.page_start,
      t.page_end = s.page_end,
      t.chunk_type = s.chunk_type,
      t.section_hint = s.section_hint,
      t.chunk_text_fr = s.chunk_text_fr,
      t.chunk_sha_fr = s.chunk_sha_fr,
      t.chunk_text_en = null,
      t.chunk_char_len_fr = s.chunk_char_len_fr,
      t.chunk_char_len_en = null,
      t.chunk_ts = current_timestamp(),
      t.translate_status = 'pending',
      t.translate_ts = null,
      t.translate_error = null,
      t.index_status = 'pending',
      t.index_ts = null,
      t.index_error = null
    WHEN MATCHED THEN UPDATE SET
      t.document_id = s.document_id,
      t.fiscal_year = s.fiscal_year,
      t.file_name = s.file_name,
      t.source_path_dbfs = s.source_path_dbfs,
      t.source_path_local = s.source_path_local,
      t.chunk_index = s.chunk_index,
      t.page_start = s.page_start,
      t.page_end = s.page_end,
      t.chunk_type = s.chunk_type,
      t.section_hint = s.section_hint,
      t.chunk_text_fr = s.chunk_text_fr,
      t.chunk_sha_fr = s.chunk_sha_fr,
      t.chunk_char_len_fr = s.chunk_char_len_fr,
      t.chunk_ts = current_timestamp()
    WHEN NOT MATCHED THEN INSERT (
      chunk_id, document_id, fiscal_year, file_name, source_path_dbfs, source_path_local,
      chunk_index, page_start, page_end, chunk_type, section_hint,
      chunk_text_fr, chunk_sha_fr, chunk_text_en, chunk_char_len_fr, chunk_char_len_en,
      chunk_ts, translate_status, translate_ts, translate_error, index_status, index_ts, index_error
    ) VALUES (
      s.chunk_id, s.document_id, s.fiscal_year, s.file_name, s.source_path_dbfs, s.source_path_local,
      s.chunk_index, s.page_start, s.page_end, s.chunk_type, s.section_hint,
      s.chunk_text_fr, s.chunk_sha_fr, null, s.chunk_char_len_fr, null,
      current_timestamp(), 'pending', null, null, 'pending', null, null
    )
    """)

    # Update inventory chunk status
    if ok_docs:
        ok_df = spark.createDataFrame([(x,) for x in ok_docs], ["document_id"])
        ok_df.createOrReplaceTempView("chunk_ok_docs")
        spark.sql(f"""
        MERGE INTO {INVENTORY_TABLE} t
        USING chunk_ok_docs s
        ON t.document_id = s.document_id
        WHEN MATCHED THEN UPDATE SET
          t.chunk_status = 'done',
          t.chunk_ts = current_timestamp(),
          t.chunk_error = null
        """)

    if failed_docs:
        failed_df = spark.createDataFrame(failed_docs, ["document_id", "chunk_error"])
        failed_df.createOrReplaceTempView("chunk_failed_docs")
        spark.sql(f"""
        MERGE INTO {INVENTORY_TABLE} t
        USING chunk_failed_docs s
        ON t.document_id = s.document_id
        WHEN MATCHED THEN UPDATE SET
          t.chunk_status = 'error',
          t.chunk_ts = current_timestamp(),
          t.chunk_error = s.chunk_error
        """)

# COMMAND ----------

# Translation step: translate only pending/error chunks without English text.
chunks_to_translate = (
    spark.table(CHUNKS_TABLE)
    .filter((F.col("translate_status").isin(["pending", "error"])) | F.col("chunk_text_en").isNull())
    .filter(F.col("chunk_text_fr").isNotNull() & (F.length(F.col("chunk_text_fr")) > 0))
    .select("chunk_id", "chunk_text_fr", "chunk_sha_fr")
)

if TRANSLATE_LIMIT:
    chunks_to_translate = chunks_to_translate.limit(int(TRANSLATE_LIMIT))

translate_count = chunks_to_translate.count()
print("Chunks to translate:", translate_count)
if translate_count == 0:
    show_validation_snapshot(CATALOG, SCHEMA)
else:
    try:
        # ai_translate syntax: ai_translate(content, to_lang)  (public preview)
        df_translated = (
            chunks_to_translate.withColumn(
                "chunk_text_en",
                F.expr("ai_translate(substring(chunk_text_fr, 1, 6500), 'en')"),
            )
            .withColumn("chunk_char_len_en", F.length(F.col("chunk_text_en")))
            .withColumn("translate_status", F.lit("done"))
            .withColumn("translate_ts", F.current_timestamp())
            .withColumn("translate_error", F.lit(None).cast("string"))
        )
        df_translated.createOrReplaceTempView("translated_chunks")

        spark.sql(f"""
        MERGE INTO {CHUNKS_TABLE} t
        USING translated_chunks s
        ON t.chunk_id = s.chunk_id
        WHEN MATCHED THEN UPDATE SET
          t.chunk_text_en = s.chunk_text_en,
          t.chunk_char_len_en = s.chunk_char_len_en,
          t.translate_status = s.translate_status,
          t.translate_ts = s.translate_ts,
          t.translate_error = s.translate_error,
          t.index_status = 'pending',
          t.index_ts = null,
          t.index_error = null
        """)
    except Exception as e:
        log_pipeline_error(ERRORS_TABLE, stage="chunk_translate", error=e)
        raise

# COMMAND ----------

# Validation views
display(
    spark.table(CHUNKS_TABLE)
    .groupBy("fiscal_year", "document_id")
    .agg(
        F.count("*").alias("chunks"),
        F.sum(F.when(F.col("chunk_text_en").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("translated_chunks"),
    )
    .orderBy("fiscal_year", "document_id")
)

show_validation_snapshot(CATALOG, SCHEMA)
