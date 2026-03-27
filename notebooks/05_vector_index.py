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

def _ensure_text_widget(name: str, default: str, override_if: set[str] | None = None):
    """
    Databricks widgets persist across reruns. If a widget already exists with an empty value,
    calling widgets.text(...) again won't update it. This helper forces a sane default.
    """
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


def _ensure_dropdown_widget(name: str, default: str, choices: list[str]):
    try:
        cur = dbutils.widgets.get(name)
        if cur not in choices:
            dbutils.widgets.remove(name)
            dbutils.widgets.dropdown(name, default, choices)
    except Exception:
        dbutils.widgets.dropdown(name, default, choices)


# Parameterize names; do not hardcode brittle endpoints.
_ensure_text_widget("vs_endpoint_name", DEFAULT_VS_ENDPOINT, override_if={"", "vs_fin_agent"})
_ensure_text_widget("vs_index_name", DEFAULT_VS_INDEX)
# Default to Databricks-hosted English embeddings if available in the workspace.
_ensure_text_widget("embedding_model_endpoint_name", "databricks-bge-large-en")
_ensure_dropdown_widget("pipeline_type", "TRIGGERED", ["TRIGGERED", "CONTINUOUS"])

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

# Re-read widgets here so manual widget edits in later cells are respected
# (Databricks notebooks often run cells out of order during debugging).
VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name").strip()
VS_INDEX_NAME = dbutils.widgets.get("vs_index_name").strip()
EMBEDDING_MODEL_ENDPOINT = dbutils.widgets.get("embedding_model_endpoint_name").strip()
PIPELINE_TYPE = dbutils.widgets.get("pipeline_type").strip().upper()

print("VS endpoint (effective):", VS_ENDPOINT_NAME)
print("VS index (effective):", VS_INDEX_NAME)
print("Embedding model (effective):", EMBEDDING_MODEL_ENDPOINT)

vsc = VectorSearchClient()

if not vsc.endpoint_exists(VS_ENDPOINT_NAME):
    existing = vsc.list_endpoints()
    if isinstance(existing, dict) and "endpoints" in existing:
        endpoints = existing.get("endpoints") or []
    elif isinstance(existing, list):
        endpoints = existing
    else:
        endpoints = []

    existing_names = []
    for e in endpoints:
        if isinstance(e, dict):
            n = e.get("name") or e.get("endpoint_name") or e.get("endpointName")
            if n:
                existing_names.append(n)
        elif isinstance(e, str) and e.strip():
            existing_names.append(e.strip())

    # De-dup + drop obvious garbage.
    existing_names = sorted({n for n in existing_names if n and n != "endpoints"})

    if existing_names:
        raise ValueError(
            f"Vector Search endpoint '{VS_ENDPOINT_NAME}' does not exist, but this workspace already has "
            f"endpoint(s) {existing_names} and the endpoint quota is exceeded (max 1). "
            f"Set the notebook widget 'vs_endpoint_name' to an existing endpoint (for example '{existing_names[0]}') "
            f"and rerun."
        )

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

created_index = False

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
    created_index = True
else:
    print("Index exists, syncing:", VS_INDEX_NAME)

index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
if PIPELINE_TYPE == "TRIGGERED":
    print("Running triggered sync:", VS_INDEX_NAME)
    index.sync()
index.wait_until_ready(verbose=True, wait_for_updates=(PIPELINE_TYPE == "TRIGGERED"))

if created_index:
    print("Created and synced Delta Sync index:", VS_INDEX_NAME)

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
