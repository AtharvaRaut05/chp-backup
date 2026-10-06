"""Write files to the S3 data lake."""

import gzip
from datetime import datetime
from functools import cache
from pathlib import Path

import boto3
import pandas as pd


@cache
def _s3():
    return boto3.client("s3")


def raw_key(source: str, name: str, now: datetime) -> str:
    return f"raw/{source}/dt={now:%Y-%m-%d}/{name}.gz"


def put_raw(bucket: str, source: str, name: str, data: bytes, now: datetime) -> str:
    """Gzip and upload a raw file to raw/<source>/dt=<date>/<name>.gz. Returns the S3 key."""
    key = raw_key(source, name, now)
    _s3().put_object(Bucket=bucket, Key=key, Body=gzip.compress(data), ContentType="application/gzip")
    return key


def put_file(bucket: str, key: str, path: Path) -> str:
    """Upload a local file as-is (for downloads that are already compressed)."""
    _s3().upload_file(str(path), bucket, key)
    return key


def write_bronze(
    df: pd.DataFrame, bucket: str, database: str, table: str, partition_cols: list[str], mode: str = "append"
) -> None:
    """Write a DataFrame as Parquet under bronze/<table>/ and register it in the Glue catalog.

    mode="append" adds files; mode="overwrite_partitions" replaces the partitions present in df,
    which makes re-running a load safe.
    """
    if df.empty:
        return
    import awswrangler as wr  # provided by the AWS SDK for pandas Lambda layer

    wr.s3.to_parquet(
        df=df,
        path=f"s3://{bucket}/bronze/{table}/",
        dataset=True,
        mode=mode,
        database=database,
        table=table,
        partition_cols=partition_cols,
    )
