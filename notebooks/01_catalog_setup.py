# Databricks notebook source
# NOTEBOOK FILE: 01_catalog_setup.py
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 01 Catalog / Schema / Volume Setup
# MAGIC
# MAGIC Creates (if needed) the Unity Catalog objects and the expected folder layout:
# MAGIC `Volumes/uc_cmifi_dev/fin_agent/annual_reports/source=parts_holding_europe/<year>/`

# COMMAND ----------

ensure_uc_objects(CATALOG, SCHEMA, VOLUME)

# COMMAND ----------

# Create source root + per-year folders (safe on reruns).
dbutils.fs.mkdirs(SOURCE_DBFS_ROOT)
for y in FISCAL_YEARS:
    dbutils.fs.mkdirs(f"{SOURCE_DBFS_ROOT}/{y}")

print("UC volume root (DBFS):", VOLUME_DBFS_ROOT)
print("UC source root (DBFS):", SOURCE_DBFS_ROOT)

# COMMAND ----------

# Quick listing to validate layout (will show empty year folders as entries).
display(dbutils.fs.ls(SOURCE_DBFS_ROOT))

# COMMAND ----------

# Create required Delta tables if they do not exist (DDL lives in sql/fin_agent_ddl.sql as well).
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {INVENTORY_TABLE} (
  document_id STRING,
  source_name STRING,
  fiscal_year INT,
  language STRING,
  file_path_dbfs STRING,
  file_path_local STRING,
  file_name STRING,
  file_size_bytes BIGINT,
  modification_ts TIMESTAMP,
  file_fingerprint STRING,
  is_present BOOLEAN,
  load_ts TIMESTAMP,
  parse_status STRING,
  parse_method STRING,
  parse_ts TIMESTAMP,
  parse_error STRING,
  chunk_status STRING,
  chunk_ts TIMESTAMP,
  chunk_error STRING,
  index_status STRING,
  index_ts TIMESTAMP,
  index_error STRING,
  notes STRING
)
USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {PAGES_TABLE} (
  document_id STRING,
  fiscal_year INT,
  file_name STRING,
  source_path_dbfs STRING,
  source_path_local STRING,
  page_num INT,
  page_text_fr STRING,
  page_char_count INT,
  parse_method STRING,
  parse_ts TIMESTAMP
)
USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
  chunk_id STRING,
  document_id STRING,
  fiscal_year INT,
  file_name STRING,
  source_path_dbfs STRING,
  source_path_local STRING,
  chunk_index INT,
  page_start INT,
  page_end INT,
  chunk_type STRING,
  section_hint STRING,
  chunk_text_fr STRING,
  chunk_sha_fr STRING,
  chunk_text_en STRING,
  chunk_char_len_fr INT,
  chunk_char_len_en INT,
  chunk_ts TIMESTAMP,
  translate_status STRING,
  translate_ts TIMESTAMP,
  translate_error STRING,
  index_status STRING,
  index_ts TIMESTAMP,
  index_error STRING
)
USING DELTA
""")

# Schema evolution: add new columns if this table existed from an earlier run.
try:
    spark.sql(f"ALTER TABLE {CHUNKS_TABLE} ADD COLUMNS (chunk_sha_fr STRING)")
except Exception:
    pass

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {ERRORS_TABLE} (
  error_ts TIMESTAMP,
  stage STRING,
  document_id STRING,
  chunk_id STRING,
  source_path STRING,
  error_type STRING,
  error_message STRING,
  stacktrace STRING,
  extra_json STRING
)
USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {RETRIEVAL_EVAL_TABLE} (
  run_id STRING,
  run_ts TIMESTAMP,
  query_text STRING,
  top_k INT,
  filters STRING,
  retrieved_json STRING,
  answer_en STRING,
  model_name STRING,
  notes STRING
)
USING DELTA
""")

# COMMAND ----------

show_validation_snapshot(CATALOG, SCHEMA)
