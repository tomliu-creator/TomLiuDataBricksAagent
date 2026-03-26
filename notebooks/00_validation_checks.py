# Databricks notebook source
# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %run ./_utils

# COMMAND ----------
# MAGIC %md
# MAGIC ## Validation Checks (Run Anytime)
# MAGIC
# MAGIC Minimal sanity checks after each stage:
# MAGIC - inventory counts + pending statuses
# MAGIC - pages row counts + per-doc/per-year pages
# MAGIC - chunks counts + translation/index statuses

# COMMAND ----------

show_validation_snapshot(CATALOG, SCHEMA)

# COMMAND ----------

display(
    spark.table(INVENTORY_TABLE)
    .filter(F.col("source_name") == SOURCE_NAME)
    .groupBy("fiscal_year", "parse_status", "chunk_status", "index_status", "is_present")
    .count()
    .orderBy("fiscal_year", "parse_status", "chunk_status")
)

# COMMAND ----------

display(
    spark.table(PAGES_TABLE)
    .groupBy("fiscal_year", "document_id", "file_name")
    .agg(F.count("*").alias("pages"), F.sum("page_char_count").alias("chars"))
    .orderBy("fiscal_year", "file_name")
)

# COMMAND ----------

display(
    spark.table(CHUNKS_TABLE)
    .groupBy("fiscal_year", "document_id")
    .agg(
        F.count("*").alias("chunks"),
        F.sum(F.when(F.col("translate_status") == "done", 1).otherwise(0)).alias("translated"),
        F.sum(F.when(F.col("index_status") == "done", 1).otherwise(0)).alias("indexed"),
    )
    .orderBy("fiscal_year", "document_id")
)

# COMMAND ----------

display(
    spark.table(ERRORS_TABLE)
    .orderBy(F.col("error_ts").desc())
    .limit(50)
)

