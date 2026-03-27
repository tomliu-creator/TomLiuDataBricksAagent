# Databricks notebook source
# NOTEBOOK FILE: 02_inventory_ingest.py
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 02 Inventory Ingest (Idempotent)
# MAGIC
# MAGIC Reads PDFs from the UC volume and upserts into `document_inventory` with deterministic `document_id`.
# MAGIC
# MAGIC Fixes the root causes in the original setup notebook:
# MAGIC - `df_docs` is always MERGEd into `document_inventory` (no silent no-op)
# MAGIC - status columns are preserved across reruns unless the underlying file changed
# MAGIC - fiscal year is derived from the *parent folder* under `source=.../YYYY/`

# COMMAND ----------

REQUIRE_NONEMPTY = True

raw_glob = f"{SOURCE_DBFS_ROOT}/*/*.pdf"
print("Scanning:", raw_glob)

df_files = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .load(raw_glob)
    .select(
        F.when(F.col("path").startswith("/Volumes/"), F.concat(F.lit("dbfs:"), F.col("path")))
        .otherwise(F.col("path"))
        .alias("file_path_dbfs"),
        F.col("length").alias("file_size_bytes"),
        F.col("modificationTime").alias("modification_ts"),
    )
    .withColumn("file_name", F.element_at(F.split(F.col("file_path_dbfs"), "/"), -1))
)

if REQUIRE_NONEMPTY:
    assert_nonzero(df_files, "binaryFile scan")

# COMMAND ----------

df_docs = (
    df_files.transform(lambda d: add_fiscal_year_col(d, SOURCE_NAME, "file_path_dbfs"))
    .withColumn("source_name", F.lit(SOURCE_NAME))
    .withColumn("language", F.lit(LANG_SOURCE))
    .withColumn("file_path_local", F.regexp_replace(F.col("file_path_dbfs"), r"^dbfs:", ""))
    .withColumn("document_id", F.sha2(F.lower(F.col("file_path_dbfs")), 256))
    .withColumn(
        "file_fingerprint",
        F.sha2(
            F.concat_ws(
                "||",
                F.col("file_path_dbfs"),
                F.col("file_size_bytes").cast("string"),
                F.col("modification_ts").cast("string"),
            ),
            256,
        ),
    )
    .withColumn("is_present", F.lit(True))
    .withColumn("load_ts", F.current_timestamp())
)

bad_years = df_docs.filter(F.col("fiscal_year").isNull())
if bad_years.count() > 0:
    display(bad_years.select("file_path_dbfs", "file_name"))
    raise ValueError(
        "Some files do not have a parsable fiscal year folder (expected .../source=parts_holding_europe/YYYY/...pdf)."
    )

display(df_docs.orderBy("fiscal_year", "file_name"))

# COMMAND ----------

# Upsert into inventory, preserving statuses unless file changed.
df_upsert = (
    df_docs.select(
        "document_id",
        "source_name",
        "fiscal_year",
        "language",
        "file_path_dbfs",
        "file_path_local",
        "file_name",
        "file_size_bytes",
        "modification_ts",
        "file_fingerprint",
        "is_present",
        "load_ts",
    )
    .withColumn("parse_status", F.lit("pending"))
    .withColumn("parse_method", F.lit(None).cast("string"))
    .withColumn("parse_ts", F.lit(None).cast("timestamp"))
    .withColumn("parse_error", F.lit(None).cast("string"))
    .withColumn("chunk_status", F.lit("pending"))
    .withColumn("chunk_ts", F.lit(None).cast("timestamp"))
    .withColumn("chunk_error", F.lit(None).cast("string"))
    .withColumn("index_status", F.lit("pending"))
    .withColumn("index_ts", F.lit(None).cast("timestamp"))
    .withColumn("index_error", F.lit(None).cast("string"))
    .withColumn("notes", F.lit(None).cast("string"))
)

df_upsert.createOrReplaceTempView("new_inventory_rows")

spark.sql(f"""
MERGE INTO {INVENTORY_TABLE} t
USING new_inventory_rows s
ON t.document_id = s.document_id
WHEN MATCHED AND (t.file_fingerprint IS NULL OR t.file_fingerprint <> s.file_fingerprint) THEN UPDATE SET
  t.source_name        = s.source_name,
  t.fiscal_year        = s.fiscal_year,
  t.language           = s.language,
  t.file_path_dbfs     = s.file_path_dbfs,
  t.file_path_local    = s.file_path_local,
  t.file_name          = s.file_name,
  t.file_size_bytes    = s.file_size_bytes,
  t.modification_ts    = s.modification_ts,
  t.file_fingerprint   = s.file_fingerprint,
  t.is_present         = true,
  t.load_ts            = current_timestamp(),
  -- reset downstream stages on content change
  t.parse_status       = 'pending',
  t.parse_method       = null,
  t.parse_ts           = null,
  t.parse_error        = null,
  t.chunk_status       = 'pending',
  t.chunk_ts           = null,
  t.chunk_error        = null,
  t.index_status       = 'pending',
  t.index_ts           = null,
  t.index_error        = null
WHEN MATCHED THEN UPDATE SET
  t.source_name        = s.source_name,
  t.fiscal_year        = s.fiscal_year,
  t.language           = s.language,
  t.file_path_dbfs     = s.file_path_dbfs,
  t.file_path_local    = s.file_path_local,
  t.file_name          = s.file_name,
  t.file_size_bytes    = s.file_size_bytes,
  t.modification_ts    = s.modification_ts,
  t.file_fingerprint   = s.file_fingerprint,
  t.is_present         = true,
  t.load_ts            = current_timestamp()
WHEN NOT MATCHED THEN INSERT (
  document_id, source_name, fiscal_year, language,
  file_path_dbfs, file_path_local, file_name,
  file_size_bytes, modification_ts, file_fingerprint, is_present, load_ts,
  parse_status, parse_method, parse_ts, parse_error,
  chunk_status, chunk_ts, chunk_error,
  index_status, index_ts, index_error,
  notes
) VALUES (
  s.document_id, s.source_name, s.fiscal_year, s.language,
  s.file_path_dbfs, s.file_path_local, s.file_name,
  s.file_size_bytes, s.modification_ts, s.file_fingerprint, s.is_present, s.load_ts,
  'pending', null, null, null,
  'pending', null, null,
  'pending', null, null,
  null
)
""")

# Mark files not currently present as is_present=false (does not delete history).
spark.sql(f"""
UPDATE {INVENTORY_TABLE}
SET is_present = false
WHERE source_name = '{SOURCE_NAME}'
  AND file_path_dbfs NOT IN (SELECT file_path_dbfs FROM new_inventory_rows)
""")

# COMMAND ----------

display(
    spark.table(INVENTORY_TABLE)
    .filter(F.col("source_name") == SOURCE_NAME)
    .groupBy("fiscal_year", "parse_status", "chunk_status", "index_status", "is_present")
    .count()
    .orderBy("fiscal_year", "parse_status")
)

show_validation_snapshot(CATALOG, SCHEMA)
