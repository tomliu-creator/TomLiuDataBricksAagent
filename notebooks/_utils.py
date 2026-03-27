# Databricks notebook source
# Shared helper utilities for the fin_agent Databricks pilot.

# COMMAND ----------

import json
import os
import re
import traceback
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql import types as T


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def uc_dbfs_to_local_path(path: str) -> str:
    """
    Normalize Databricks paths for Python libraries that require local filesystem paths.

    Supported:
      - dbfs:/Volumes/...  -> /Volumes/...
      - /Volumes/...       -> /Volumes/...
      - dbfs:/...          -> /dbfs/...
      - /dbfs/...          -> /dbfs/...
    """
    if path is None:
        return None
    if path.startswith("dbfs:/Volumes/"):
        return "/Volumes/" + path[len("dbfs:/Volumes/") :]
    if path.startswith("/Volumes/"):
        return path
    if path.startswith("dbfs:/"):
        return "/dbfs/" + path[len("dbfs:/") :]
    if path.startswith("/dbfs/"):
        return path
    # Last resort: return as-is (may still work for some runtimes)
    return path


def robust_extract_fiscal_year(file_path: str, source_name: str) -> int:
    """
    Extract a 4-digit year from the folder immediately under `source=<name>/`.
    Returns None if not parseable.
    """
    if not file_path:
        return None
    # Example: .../annual_reports/source=parts_holding_europe/2019/report.pdf
    m = re.search(rf"source={re.escape(source_name)}/(\d{{4}})(?:/|$)", file_path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def add_fiscal_year_col(df, source_name: str, col_name: str = "file_path"):
    # Prefer a strict folder-based extraction (avoids picking random years from inside filenames)
    return df.withColumn(
        "fiscal_year",
        F.expr(
            f"try_cast(nullif(regexp_extract({col_name}, 'source={source_name}/(\\\\d{{4}})(/|$)', 1), '') as int)"
        ),
    )


def ensure_uc_objects(catalog: str, schema: str, volume: str):
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")


def log_pipeline_error(
    table_fqn: str,
    stage: str,
    document_id: str = None,
    chunk_id: str = None,
    source_path: str = None,
    error: Exception = None,
    extra: dict | None = None,
):
    payload = {
        "error_ts": datetime.now(timezone.utc),
        "stage": stage,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "source_path": source_path,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "stacktrace": traceback.format_exc() if error else None,
        "extra_json": json.dumps(extra or {}, ensure_ascii=True),
    }
    spark.createDataFrame([payload]).write.mode("append").saveAsTable(table_fqn)


def show_validation_snapshot(catalog: str, schema: str):
    """
    Lightweight counts to run after each major stage.
    Uses display() if present.
    """
    def _safe_count(sql_text: str) -> int | None:
        try:
            return spark.sql(sql_text).collect()[0][0]
        except Exception:
            return None

    inv = f"{catalog}.{schema}.document_inventory"
    pages = f"{catalog}.{schema}.document_pages"
    chunks = f"{catalog}.{schema}.document_chunks"

    rows = [
        ("inventory_rows", _safe_count(f"SELECT count(*) FROM {inv}")),
        ("inventory_parse_pending", _safe_count(f"SELECT count(*) FROM {inv} WHERE parse_status = 'pending'")),
        ("pages_rows", _safe_count(f"SELECT count(*) FROM {pages}")),
        ("chunks_rows", _safe_count(f"SELECT count(*) FROM {chunks}")),
        ("chunks_translate_pending", _safe_count(f"SELECT count(*) FROM {chunks} WHERE translate_status = 'pending'")),
        ("chunks_index_pending", _safe_count(f"SELECT count(*) FROM {chunks} WHERE index_status = 'pending'")),
    ]
    schema = T.StructType(
        [
            T.StructField("metric", T.StringType(), nullable=False),
            T.StructField("value", T.LongType(), nullable=True),
        ]
    )
    # Explicit schema avoids Spark inference failures when all `value` entries are NULL.
    df = spark.createDataFrame(rows, schema=schema)
    try:
        display(df)
    except Exception:
        print(df.toPandas().to_string(index=False))


def assert_nonzero(df, what: str):
    c = df.count()
    if c == 0:
        raise ValueError(f"{what} produced 0 rows; stopping to prevent silent no-op.")
    return c
