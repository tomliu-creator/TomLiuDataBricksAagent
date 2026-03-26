# Databricks notebook source
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 03 Parse Reports (PyMuPDF Baseline)
# MAGIC
# MAGIC Parses each pending PDF into page-level French text and stores it in `document_pages`.
# MAGIC
# MAGIC Key reliability points:
# MAGIC - Uses `file_path_local` (`/Volumes/...`) for Python PDF libraries
# MAGIC - Per-document error handling and logging to `pipeline_errors`
# MAGIC - Writes pages via MERGE on `(document_id, page_num)` so reruns do not duplicate

# COMMAND ----------
# MAGIC %pip install -U pymupdf

# COMMAND ----------

import fitz  # PyMuPDF

MAX_DOCS = None  # set to an int for debugging (e.g. 2)

pending = (
    spark.table(INVENTORY_TABLE)
    .filter((F.col("source_name") == SOURCE_NAME) & (F.col("is_present") == True))
    .filter(F.col("parse_status").isin(["pending", "error"]))
    .select(
        "document_id",
        "fiscal_year",
        "file_name",
        "file_path_dbfs",
        "file_path_local",
    )
    .orderBy("fiscal_year", "file_name")
)

pending_count = pending.count()
print("Pending documents to parse:", pending_count)
if pending_count == 0:
    show_validation_snapshot(CATALOG, SCHEMA)
    dbutils.notebook.exit("Nothing to parse.")

docs = pending.limit(MAX_DOCS).collect() if MAX_DOCS else pending.collect()
print("Will parse:", len(docs))

# COMMAND ----------

page_rows = []
ok_doc_ids = []
failed_docs = []  # list of (document_id, error_message)

for d in docs:
    doc_id = d["document_id"]
    local_path = d["file_path_local"] or uc_dbfs_to_local_path(d["file_path_dbfs"])
    try:
        pdf = fitz.open(local_path)
        for idx in range(pdf.page_count):
            page = pdf.load_page(idx)
            text = page.get_text("text") or ""
            page_rows.append(
                {
                    "document_id": doc_id,
                    "fiscal_year": int(d["fiscal_year"]) if d["fiscal_year"] is not None else None,
                    "file_name": d["file_name"],
                    "source_path_dbfs": d["file_path_dbfs"],
                    "source_path_local": local_path,
                    "page_num": idx + 1,
                    "page_text_fr": text,
                    "page_char_count": len(text),
                    "parse_method": "pymupdf_text",
                    "parse_ts": None,  # set after DF creation
                }
            )
        pdf.close()
        ok_doc_ids.append(doc_id)
    except Exception as e:
        failed_docs.append((doc_id, str(e)[:4000]))
        log_pipeline_error(
            ERRORS_TABLE,
            stage="parse_pdf",
            document_id=doc_id,
            source_path=d["file_path_dbfs"],
            error=e,
            extra={"file_path_local": local_path},
        )

print("Parsed OK docs:", len(ok_doc_ids))
print("Failed docs:", len(failed_docs))

if len(page_rows) == 0:
    raise RuntimeError("No pages were parsed. Check file paths and PyMuPDF availability.")

# COMMAND ----------

df_pages = spark.createDataFrame(page_rows).withColumn("parse_ts", F.current_timestamp())
df_pages.createOrReplaceTempView("new_pages")

# Remove stale pages for successfully parsed documents (handles PDFs whose page count changed).
if ok_doc_ids:
    ok_df_for_delete = spark.createDataFrame([(x,) for x in ok_doc_ids], ["document_id"])
    ok_df_for_delete.createOrReplaceTempView("docs_to_replace_pages")
    spark.sql(f"""
    DELETE FROM {PAGES_TABLE}
    WHERE document_id IN (SELECT document_id FROM docs_to_replace_pages)
    """)

spark.sql(f"""
MERGE INTO {PAGES_TABLE} t
USING new_pages s
ON t.document_id = s.document_id AND t.page_num = s.page_num
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

# COMMAND ----------

if ok_doc_ids:
    ok_df = spark.createDataFrame([(x,) for x in ok_doc_ids], ["document_id"])
    ok_df.createOrReplaceTempView("ok_docs")
    spark.sql(f"""
    MERGE INTO {INVENTORY_TABLE} t
    USING ok_docs s
    ON t.document_id = s.document_id
    WHEN MATCHED THEN UPDATE SET
      t.parse_status = 'done',
      t.parse_method = 'pymupdf_text',
      t.parse_ts = current_timestamp(),
      t.parse_error = null
    """)

if failed_docs:
    failed_df = spark.createDataFrame(failed_docs, ["document_id", "parse_error"])
    failed_df.createOrReplaceTempView("failed_docs")
    spark.sql(f"""
    MERGE INTO {INVENTORY_TABLE} t
    USING failed_docs s
    ON t.document_id = s.document_id
    WHEN MATCHED THEN UPDATE SET
      t.parse_status = 'error',
      t.parse_ts = current_timestamp(),
      t.parse_error = s.parse_error
    """)

# COMMAND ----------

# Validation: per-document page counts + stage snapshot
display(
    spark.table(PAGES_TABLE)
    .groupBy("fiscal_year", "document_id", "file_name")
    .agg(F.count("*").alias("pages"), F.sum("page_char_count").alias("chars"))
    .orderBy("fiscal_year", "file_name")
)

show_validation_snapshot(CATALOG, SCHEMA)
