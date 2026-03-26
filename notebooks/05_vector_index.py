# Databricks notebook source
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## 05 Vector Search Index (Delta Sync)
# MAGIC
# MAGIC Creates/updates a Delta Sync Vector Search index over `document_chunks`.
# MAGIC
# MAGIC Design choice for this pilot:
# MAGIC - Embedding + retrieval text column: `chunk_text_en`
# MAGIC - Evidence/citations: resolved from the Delta table (`chunk_text_fr`, page range, source path)

# COMMAND ----------
# MAGIC %pip install -U databricks-vectorsearch

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Parameterize names; do not hardcode brittle endpoints.
dbutils.widgets.text("vs_endpoint_name", DEFAULT_VS_ENDPOINT)
dbutils.widgets.text("vs_index_name", DEFAULT_VS_INDEX)
dbutils.widgets.text("embedding_model_endpoint_name", "")
dbutils.widgets.dropdown("pipeline_type", "TRIGGERED", ["TRIGGERED", "CONTINUOUS"])

VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()
VS_INDEX_NAME = dbutils.widgets.get("vs_index_name").strip()
EMBEDDING_MODEL_ENDPOINT = dbutils.widgets.get("embedding_model_endpoint_name").strip()
PIPELINE_TYPE = dbutils.widgets.get("pipeline_type").strip().upper()

print("VS endpoint:", VS_ENDPOINT_NAME)
print("VS index:", VS_INDEX_NAME)
print("Pipeline type:", PIPELINE_TYPE)
print("Embedding model endpoint:", EMBEDDING_MODEL_ENDPOINT or "(REQUIRED)")

if not EMBEDDING_MODEL_ENDPOINT:
    raise ValueError(
        "Set widget 'embedding_model_endpoint_name' to an English embedding model serving endpoint (multilingual not available here)."
    )

# COMMAND ----------

# Guardrail: ensure translations exist before indexing.
missing_en = (
    spark.table(CHUNKS_TABLE)
    .filter(F.col("chunk_text_en").isNull() | (F.length(F.col("chunk_text_en")) == 0))
    .count()
)
if missing_en > 0:
    raise ValueError(f"Found {missing_en} chunks with missing English translation; run 04_chunk_translate first.")

# Enable CDF for Delta Sync
spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

vsc = VectorSearchClient()

if not vsc.endpoint_exists(VS_ENDPOINT_NAME):
    print("Creating Vector Search endpoint:", VS_ENDPOINT_NAME)
    vsc.create_endpoint_and_wait(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
else:
    print("Vector Search endpoint exists:", VS_ENDPOINT_NAME)

columns_to_sync = [
    "document_id",
    "fiscal_year",
    "file_name",
    "source_path_dbfs",
    "page_start",
    "page_end",
    "chunk_type",
    "section_hint",
    "chunk_index",
]

if not vsc.index_exists(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME):
    print("Creating Delta Sync index:", VS_INDEX_NAME)
    vsc.create_delta_sync_index_and_wait(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        primary_key="chunk_id",
        source_table_name=CHUNKS_TABLE,
        pipeline_type=PIPELINE_TYPE,
        embedding_source_column="chunk_text_en",
        embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT,
        columns_to_sync=columns_to_sync,
    )
else:
    print("Index exists, syncing:", VS_INDEX_NAME)
    index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
    if PIPELINE_TYPE == "TRIGGERED":
        index.sync()
    index.wait_until_ready(verbose=True, wait_for_updates=(PIPELINE_TYPE == "TRIGGERED"))

# Mark chunks as indexed (requested/ready) for dashboarding.
spark.sql(f"""
UPDATE {CHUNKS_TABLE}
SET index_status = 'done',
    index_ts = current_timestamp(),
    index_error = null
WHERE index_status = 'pending'
  AND translate_status = 'done'
""")

spark.sql(f"""
UPDATE {INVENTORY_TABLE} t
SET index_status = 'done',
    index_ts = current_timestamp(),
    index_error = null
WHERE t.source_name = '{SOURCE_NAME}'
  AND t.parse_status = 'done'
  AND t.chunk_status = 'done'
  AND EXISTS (SELECT 1 FROM {CHUNKS_TABLE} c WHERE c.document_id = t.document_id)
  AND NOT EXISTS (SELECT 1 FROM {CHUNKS_TABLE} c WHERE c.document_id = t.document_id AND c.index_status <> 'done')
""")

show_validation_snapshot(CATALOG, SCHEMA)
