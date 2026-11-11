"""Step 2 of the history backfill: load PeMS's historical CHP incident files into the lake.

For each day: download the statewide chp_incidents_day zip from the PeMS Clearinghouse, keep a
raw copy under raw/pems_chp/, and write the parsed incidents and notes to the bronze tables
hist_chp_incidents and hist_chp_details (one partition per day, replaced on a rerun). Skips
days already loaded unless --reload. Runs on your computer with your AWS login; the PeMS login
comes from Parameter Store, as for the daily workflow.

    python scripts/backfill_chp_history.py --start 2026-09-01 --end 2026-10-31
"""

import argparse
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import awswrangler as wr
import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pems.client import PemsClient  # noqa: E402
from pems.incidents import DETAIL_TYPES, INCIDENT_TYPES, file_name, parse_day_zip  # noqa: E402

STACK = "chp-backup"
DATABASE = "traffic_bronze"


def days_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=k) for k in range((end - start).days + 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--profile", default="traffic")
    ap.add_argument("--reload", action="store_true", help="load days again even if already loaded")
    args = ap.parse_args()
    session = boto3.Session(profile_name=args.profile, region_name="us-west-2")
    bucket = next(o["OutputValue"] for o in session.client("cloudformation").describe_stacks(StackName=STACK)
                  ["Stacks"][0]["Outputs"] if o["OutputKey"] == "LakeBucket")
    s3, ssm = session.client("s3"), session.client("ssm")
    user, password = (ssm.get_parameter(Name=f"/traffic/pems/{n}", WithDecryption=True)["Parameter"]["Value"]
                      for n in ("username", "password"))
    client = PemsClient(user, password)
    client.login()

    available = {}
    for year in sorted({d.year for d in days_between(args.start, args.end)}):
        available |= client.list_files(7, "chp_incidents_day", year)  # statewide files, listed under any district
    done = set()
    if not args.reload and wr.catalog.does_table_exist(DATABASE, "hist_chp_incidents", boto3_session=session):
        parts = wr.catalog.get_partitions(DATABASE, "hist_chp_incidents", boto3_session=session)
        done = {values[0] for values in parts.values()}

    loaded = missing = 0
    for day in days_between(args.start, args.end):
        name = file_name(day)
        if day.isoformat() in done:
            continue
        if name not in available:
            print(f"{day}: not on PeMS")
            missing += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = client.download(available[name], Path(tmp) / name)
            raw = path.read_bytes()
        s3.put_object(Bucket=bucket, Key=f"raw/pems_chp/dt={day}/{name}", Body=raw)
        incidents, details = parse_day_zip(raw, day)
        for df, table, types in ((incidents, "hist_chp_incidents", INCIDENT_TYPES),
                                 (details, "hist_chp_details", DETAIL_TYPES)):
            wr.s3.to_parquet(df=df, path=f"s3://{bucket}/bronze/{table}/", dataset=True,
                             mode="overwrite_partitions", database=DATABASE, table=table,
                             partition_cols=["dt"], dtype=types, boto3_session=session)
        print(f"{day}: {len(incidents):,} incidents, {len(details):,} notes")
        loaded += 1
    print(f"done: {loaded} days loaded, {len(done)} already there, {missing} not on PeMS")


if __name__ == "__main__":
    main()
