-- Unity Catalog / Delta DDL for the fin_agent pilot (Parts Holding Europe)
-- Intended to be run in Databricks SQL.

-- ---------------------------------------------------------------------------
-- Catalog + schema + volume
-- ---------------------------------------------------------------------------
CREATE CATALOG IF NOT EXISTS uc_cmifi_dev;
CREATE SCHEMA IF NOT EXISTS uc_cmifi_dev.fin_agent;
CREATE VOLUME IF NOT EXISTS uc_cmifi_dev.fin_agent.annual_reports;

-- ---------------------------------------------------------------------------
-- Inventory (idempotent MERGE target)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uc_cmifi_dev.fin_agent.document_inventory (
  document_id STRING,

  source_name STRING,                -- parts_holding_europe
  fiscal_year INT,
  language STRING,                   -- fr (source language)

  file_path_dbfs STRING,             -- dbfs:/Volumes/.../source=.../YYYY/file.pdf
  file_path_local STRING,            -- /Volumes/.../source=.../YYYY/file.pdf  (for Python libs)
  file_name STRING,
  file_size_bytes BIGINT,
  modification_ts TIMESTAMP,
  file_fingerprint STRING,           -- sha2(file_path_dbfs|size|mtime) to detect changes
  is_present BOOLEAN,                -- false if file missing from current volume listing

  load_ts TIMESTAMP,

  parse_status STRING,               -- pending|done|error|skipped
  parse_method STRING,
  parse_ts TIMESTAMP,
  parse_error STRING,

  chunk_status STRING,               -- pending|done|error
  chunk_ts TIMESTAMP,
  chunk_error STRING,

  index_status STRING,               -- pending|done|error
  index_ts TIMESTAMP,
  index_error STRING,

  notes STRING
)
USING DELTA;

-- ---------------------------------------------------------------------------
-- Parsed pages (French evidence layer)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uc_cmifi_dev.fin_agent.document_pages (
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
USING DELTA;

-- ---------------------------------------------------------------------------
-- Chunks (French evidence + English retrieval layer)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uc_cmifi_dev.fin_agent.document_chunks (
  chunk_id STRING,                   -- deterministic
  document_id STRING,
  fiscal_year INT,
  file_name STRING,
  source_path_dbfs STRING,
  source_path_local STRING,

  chunk_index INT,                   -- stable per doc for debugging
  page_start INT,
  page_end INT,
  chunk_type STRING,                 -- sliding_pages|section_aware (future)
  section_hint STRING,

  chunk_text_fr STRING,
  chunk_sha_fr STRING,               -- sha2(chunk_text_fr) to detect changes
  chunk_text_en STRING,
  chunk_char_len_fr INT,
  chunk_char_len_en INT,

  chunk_ts TIMESTAMP,

  translate_status STRING,           -- pending|done|error
  translate_ts TIMESTAMP,
  translate_error STRING,

  index_status STRING,               -- pending|done|error
  index_ts TIMESTAMP,
  index_error STRING
)
USING DELTA;

-- ---------------------------------------------------------------------------
-- Pipeline error log (per-stage diagnostics)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uc_cmifi_dev.fin_agent.pipeline_errors (
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
USING DELTA;

-- ---------------------------------------------------------------------------
-- Retrieval evaluation runs (bench questions + retrieved chunks + draft answer)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uc_cmifi_dev.fin_agent.retrieval_eval_runs (
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
USING DELTA;
