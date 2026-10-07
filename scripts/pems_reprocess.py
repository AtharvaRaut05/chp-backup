"""Re-parse PeMS station files already saved in S3 (raw/pems/...) and rewrite their bronze partitions.

Use after changing src/pems/parse.py, e.g. to keep ramp and connector sensors. Nothing is
downloaded from PeMS; it reads the copies the daily workflow saved. Safe to rerun: each
district-day partition is replaced, not appended.

    python scripts/pems_reprocess.py                    # every saved day
    python scripts/pems_reprocess.py --since 2026-10-01
"""

import argparse
import sys
import tempfile
from datetime import date
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.lake import write_bronze  # noqa: E402
from pems.parse import file_date, parse_station_5min  # noqa: E402

STACK = "chp-backup"
GLUE_DB = "traffic_bronze"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=date.fromisoformat, default=date.min)
    ap.add_argument("--profile", default="traffic")
    args = ap.parse_args()

    boto3.setup_default_session(profile_name=args.profile, region_name="us-west-2")
    s3 = boto3.client("s3")
    bucket = _lake_bucket()

    keys = [
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="raw/pems/")
        for obj in page.get("Contents", [])
        if "/type=station_5min/" in obj["Key"] and file_date(Path(obj["Key"]).name) >= args.since
    ]
    print(f"{len(keys)} station files to reprocess")

    with tempfile.TemporaryDirectory() as tmp:
        for i, key in enumerate(sorted(keys), 1):
            name = Path(key).name
            district, day = int(name[1:3]), file_date(name)
            path = Path(tmp) / name
            s3.download_file(bucket, key, str(path))
            df = parse_station_5min(path, district, day)
            write_bronze(df, bucket, GLUE_DB, "pems_station_5min", ["district", "dt"], mode="overwrite_partitions")
            path.unlink()
            print(f"[{i}/{len(keys)}] d{district:02d} {day}: {len(df):,} rows "
                  f"({', '.join(f'{t} {n:,}' for t, n in df['lane_type'].value_counts().items())})")


def _lake_bucket() -> str:
    outputs = boto3.client("cloudformation").describe_stacks(StackName=STACK)["Stacks"][0]["Outputs"]
    return next(o["OutputValue"] for o in outputs if o["OutputKey"] == "LakeBucket")


if __name__ == "__main__":
    main()
