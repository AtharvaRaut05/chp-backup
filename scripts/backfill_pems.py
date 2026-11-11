"""Step 4 of the history backfill: download PeMS sensor data for the history days.

Lists every district-day in the range whose 5-minute sensor file isn't in the lake yet, splits the
list into batches and starts one run of the backfill workflow (AWS Step Functions) per batch. The
runs download one file at a time each, side by side, with the same function the daily workflow
uses, and keep going without your computer. Rerun the script to retry whatever is still missing.

    python scripts/backfill_pems.py --start 2026-01-01 --end 2026-09-28 --dry-run   # just count
    python scripts/backfill_pems.py --start 2026-01-01 --end 2026-09-28
    python scripts/backfill_pems.py --status                                         # progress
"""

import argparse
import json
import re
from datetime import date, datetime, timedelta, timezone

import boto3

STACK = "chp-backup"
DISTRICTS = (4, 7, 8, 11, 12)   # the CHP centers collected (see DispatchIds in template.yaml)
BATCH = 350                     # district-days per workflow run (keeps each run's history small)
_PART = re.compile(r"district=(\d+)/dt=(\d{4}-\d{2}-\d{2})/")


def todo(start: date, end: date, have: set[tuple[int, str]], districts=DISTRICTS) -> list[dict]:
    days = [start + timedelta(days=k) for k in range((end - start).days + 1)]
    return [{"district": d, "date": day.isoformat()} for day in days for d in districts
            if (d, day.isoformat()) not in have]


def batches(items: list, size: int = BATCH) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def loaded(s3, bucket: str) -> set[tuple[int, str]]:
    """(district, day) pairs with 5-minute sensor files already in the lake."""
    have = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="bronze/pems_station_5min/"):
        for obj in page.get("Contents", []):
            m = _PART.search(obj["Key"])
            if m:
                have.add((int(m.group(1)), m.group(2)))
    return have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--profile", default="traffic")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="show the backfill runs and how far they got")
    args = ap.parse_args()
    session = boto3.Session(profile_name=args.profile, region_name="us-west-2")
    outputs = {o["OutputKey"]: o["OutputValue"] for o in
               session.client("cloudformation").describe_stacks(StackName=STACK)["Stacks"][0]["Outputs"]}
    sf = session.client("stepfunctions")
    if args.status:
        for run in sf.list_executions(stateMachineArn=outputs["BackfillWorkflowArn"], maxResults=20)["executions"]:
            print(f"{run['name']}: {run['status']} (started {run['startDate']:%Y-%m-%d %H:%M})")
        have = loaded(session.client("s3"), outputs["LakeBucket"])
        print(f"district-days of sensor data in the lake: {len(have):,}")
        return
    if not (args.start and args.end):
        ap.error("--start and --end are required (or --status)")
    items = todo(args.start, args.end, loaded(session.client("s3"), outputs["LakeBucket"]))
    parts = batches(items)
    print(f"{len(items):,} district-days to download, in {len(parts)} runs of up to {BATCH}")
    if args.dry_run or not items:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for i, part in enumerate(parts, 1):
        run = sf.start_execution(stateMachineArn=outputs["BackfillWorkflowArn"], name=f"backfill-{stamp}-{i}",
                                 input=json.dumps({"todo": part}))
        print(f"  started run {i}: {part[0]['date']} to {part[-1]['date']} ({len(part)} files)  {run['executionArn']}")
    print("Check progress with: python scripts/backfill_pems.py --status")


if __name__ == "__main__":
    main()
