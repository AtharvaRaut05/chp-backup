"""Remove temporary tables a failed dbt run can leave behind in Athena.

dbt names them <something>__dbt_tmp. If one is left over, the next run stops with
TABLE_ALREADY_EXISTS, so the scheduled job clears them first.

    python jobs/drop_dbt_tmp.py traffic_silver
"""

import sys

import boto3
from botocore.exceptions import ClientError


def drop_tmp_tables(database: str, glue=None) -> list[str]:
    glue = glue or boto3.client("glue")
    names = [t["Name"]
             for page in glue.get_paginator("get_tables").paginate(DatabaseName=database)
             for t in page["TableList"] if t["Name"].endswith("__dbt_tmp")]
    for name in names:
        glue.delete_table(DatabaseName=database, Name=name)
    return names


def main() -> None:
    database = sys.argv[1] if len(sys.argv) > 1 else "traffic_silver"
    try:
        dropped = drop_tmp_tables(database)
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityNotFoundException":
            raise
        dropped = []  # first run: dbt hasn't created the database yet
    print(f"dropped {len(dropped)} leftover dbt tables: {dropped}" if dropped else "no leftover dbt tables")


if __name__ == "__main__":
    main()
