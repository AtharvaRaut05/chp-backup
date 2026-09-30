"""Write files to the S3 data lake."""

import gzip
from datetime import datetime
from functools import cache

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


def write_bronze(df: pd.DataFrame, bucket: str, database: str, table: str, partition_cols: list[str]) -> None:
    """Append a DataFrame as Parquet under bronze/<table>/ and register it in the Glue catalog."""
    if df.empty:
        return
    import awswrangler as wr  # provided by the AWS SDK for pandas Lambda layer

    wr.s3.to_parquet(
        df=df,
        path=f"s3://{bucket}/bronze/{table}/",
        dataset=True,
        mode="append",
        database=database,
        table=table,
        partition_cols=partition_cols,
    )
