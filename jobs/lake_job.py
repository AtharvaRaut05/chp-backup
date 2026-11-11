"""Shared plumbing for the daily job's Python steps: AWS session, lake bucket, bronze tables."""

from __future__ import annotations

import os

DATABASE = "traffic_bronze"


def session_and_bucket(profile: str = ""):
    import boto3

    session = boto3.Session(profile_name=profile or None, region_name="us-west-2")
    bucket = os.environ.get("LAKE_BUCKET") or next(
        o["OutputValue"] for o in session.client("cloudformation").describe_stacks(StackName="chp-backup")
        ["Stacks"][0]["Outputs"] if o["OutputKey"] == "LakeBucket")
    return session, bucket


def ensure_table(session, bucket: str, table: str, columns: dict[str, str]) -> None:
    """Create an empty table partitioned by dt, so dbt can read it before any rows exist."""
    import awswrangler as wr

    if not wr.catalog.does_table_exist(DATABASE, table, boto3_session=session):
        wr.catalog.create_parquet_table(database=DATABASE, table=table, path=f"s3://{bucket}/bronze/{table}/",
                                        columns_types=columns, partitions_types={"dt": "string"},
                                        boto3_session=session)


def query(session, sql: str, database: str = DATABASE):
    import awswrangler as wr

    return wr.athena.read_sql_query(sql, database=database, workgroup="traffic", boto3_session=session,
                                    ctas_approach=False)


def write(df, session, bucket: str, table: str, columns: dict[str, str], mode: str = "append") -> None:
    """Write rows (with a dt column) to bronze/<table>/ using the table's declared column types."""
    import awswrangler as wr

    if df.empty:
        return
    wr.s3.to_parquet(df=df[[*columns, "dt"]], path=f"s3://{bucket}/bronze/{table}/", dataset=True, mode=mode,
                     database=DATABASE, table=table, partition_cols=["dt"], boto3_session=session,
                     dtype=columns)


def get_secret(session, name: str) -> str | None:
    """A key saved in Parameter Store, or None if it hasn't been set up."""
    ssm = session.client("ssm")
    try:
        return ssm.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]
    except ssm.exceptions.ParameterNotFound:
        return None
