# Databricks notebook source
# This notebook is intended to be `%run` from the pipeline notebooks.

# COMMAND ----------

# Core UC locations (requested structure)
CATALOG = "uc_cmifi_dev"
SCHEMA = "fin_agent"
VOLUME = "annual_reports"

# Source identity for this pilot
SOURCE_NAME = "parts_holding_europe"
SOURCE_PREFIX = f"source={SOURCE_NAME}"
LANG_SOURCE = "fr"
LANG_RETRIEVAL = "en"

# Year folders to create/expect (some may legitimately be empty)
FISCAL_YEARS = list(range(2011, 2024))

# Databricks volume paths
VOLUME_DBFS_ROOT = f"dbfs:/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
VOLUME_FUSE_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SOURCE_DBFS_ROOT = f"{VOLUME_DBFS_ROOT}/{SOURCE_PREFIX}"
SOURCE_FUSE_ROOT = f"{VOLUME_FUSE_ROOT}/{SOURCE_PREFIX}"

# Tables
INVENTORY_TABLE = f"{CATALOG}.{SCHEMA}.document_inventory"
PAGES_TABLE = f"{CATALOG}.{SCHEMA}.document_pages"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.document_chunks"
ERRORS_TABLE = f"{CATALOG}.{SCHEMA}.pipeline_errors"
RETRIEVAL_EVAL_TABLE = f"{CATALOG}.{SCHEMA}.retrieval_eval_runs"

# Vector Search (names are intentionally parameterized; set in 05_vector_index/07_llm_query_demo)
# Workspace endpoint quota is often 1; reuse the existing endpoint if present.
DEFAULT_VS_ENDPOINT = "emd-default-vs"
DEFAULT_VS_INDEX = f"{CATALOG}.{SCHEMA}.vs_parts_holding_europe_chunks_en"

