"""Turn the saved lane closure files into the bronze table caltrans_lane_closures.

The collector (src/caltrans/lane_closures.py) saves each district's feed every 5 minutes. This step
reads one UTC day of those files and keeps one row per closure for that day: when it was first and
last seen, and its latest details. A closure that spans several days has a row in each; the staging
model combines them. Runs daily before dbt; re-parses yesterday and today (today is still filling in)
and any older day not parsed yet.

    python jobs/lane_closures.py --profile traffic

Times come from the feed's epoch fields: planned start/end, and when crews radioed that the closure
actually started (code 1097), ended (1098) or was cancelled (1022).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from lake_job import ensure_table, session_and_bucket, write

TABLE = "caltrans_lane_closures"
RAW_PREFIX = "raw/caltrans_lcs/"
INDEFINITE_EPOCH = 32503708740  # the feed's "no end date" (2999-12-31 23:59)
_NAME = re.compile(r"D(\d{2})_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)\.json\.gz$")

# (column, path inside the "lcs" record)
FIELDS = {
    "closure_key": ("index",),
    "closure_id": ("closure", "closureID"),
    "route": ("location", "begin", "beginRoute"),
    "direction": ("location", "travelFlowDirection"),
    "begin_lat": ("location", "begin", "beginLatitude"),
    "begin_lon": ("location", "begin", "beginLongitude"),
    "end_lat": ("location", "end", "endLatitude"),
    "end_lon": ("location", "end", "endLongitude"),
    "begin_abs_pm": ("location", "begin", "beginMilepost"),
    "end_abs_pm": ("location", "end", "endMilepost"),
    "facility": ("closure", "facility"),
    "closure_type": ("closure", "typeOfClosure"),
    "work_type": ("closure", "typeOfWork"),
    "lanes_closed": ("closure", "lanesClosed"),
    "total_lanes": ("closure", "totalExistingLanes"),
    "planned_start": ("closure", "closureTimestamp", "closureStartEpoch"),
    "planned_end": ("closure", "closureTimestamp", "closureEndEpoch"),
    "started_at": ("closure", "code1097", "code1097Timestamp", "code1097Epoch"),
    "ended_at": ("closure", "code1098", "code1098Timestamp", "code1098Epoch"),
    "cancelled_at": ("closure", "code1022", "code1022Timestamp", "code1022Epoch"),
}
NUMERIC = ["begin_lat", "begin_lon", "end_lat", "end_lon", "begin_abs_pm", "end_abs_pm"]
TIMES = ["planned_start", "planned_end", "started_at", "ended_at", "cancelled_at"]
COLUMNS = {
    "closure_key": "string", "district": "int", "closure_id": "string", "route": "string", "direction": "string",
    **{c: "double" for c in NUMERIC},
    "facility": "string", "closure_type": "string", "work_type": "string", "lanes_closed": "string",
    "total_lanes": "string",
    **{c: "timestamp" for c in TIMES},
    "first_seen": "timestamp", "last_seen": "timestamp", "n_polls": "int",
}


def _get(record: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return None if record in ("", None) else record


def to_time(value) -> pd.Timestamp | None:
    """Epoch seconds (or milliseconds) -> UTC timestamp; blank or the 'indefinite' value -> None."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x > 1e11:  # milliseconds
        x /= 1000
    if x <= 0 or x >= INDEFINITE_EPOCH:
        return None
    return pd.Timestamp(x, unit="s", tz="UTC")


def parse_file(body: bytes, district: int, polled_at: pd.Timestamp) -> pd.DataFrame:
    """One snapshot of one district's feed -> one row per closure."""
    data = json.loads(body)
    records = data.get("data", []) if isinstance(data, dict) else data
    rows = []
    for r in records:
        lcs = r.get("lcs", r) if isinstance(r, dict) else None
        if not isinstance(lcs, dict):
            continue
        row = {col: _get(lcs, path) for col, path in FIELDS.items()}
        if row["closure_key"] is None:
            continue
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(FIELDS))
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TIMES:
        df[c] = pd.to_datetime(df[c].map(to_time), utc=True)
    df["district"] = district
    df["polled_at"] = polled_at
    return df


def summarize(snapshots: pd.DataFrame, day: date) -> pd.DataFrame:
    """One row per closure for the day: latest details plus first/last time it was in the feed."""
    if snapshots.empty:
        return pd.DataFrame(columns=[*COLUMNS, "dt"])
    snapshots = snapshots.sort_values("polled_at")
    g = snapshots.groupby("closure_key", sort=True)
    out = g.last()  # latest non-null value of each field
    out["first_seen"] = g["polled_at"].min()
    out["last_seen"] = g["polled_at"].max()
    out["n_polls"] = g.size()
    out["district"] = g["district"].last()
    out = out.reset_index()
    out["dt"] = day.isoformat()
    return out[[*COLUMNS, "dt"]]


# ---------- AWS ----------

def days_to_parse(s3, bucket: str, today: date) -> list[date]:
    def partitions(prefix):
        pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
        return {date.fromisoformat(p["Prefix"].rstrip("/").split("dt=")[1])
                for page in pages for p in page.get("CommonPrefixes", [])}

    raw, done = partitions(RAW_PREFIX), partitions(f"bronze/{TABLE}/")
    return sorted(d for d in raw if d not in done or d >= today - timedelta(days=1))


def read_day(s3, bucket: str, day: date) -> pd.DataFrame:
    frames = []
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{RAW_PREFIX}dt={day}/")
    for page in pages:
        for obj in page.get("Contents", []):
            m = _NAME.search(obj["Key"])
            if not m:
                continue
            polled_at = pd.Timestamp(datetime.strptime(m.group(2), "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc))
            body = gzip.decompress(s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
            try:
                frames.append(parse_file(body, int(m.group(1)), polled_at))
            except ValueError as e:
                print(f"WARNING skipping unreadable {obj['Key']}: {str(e)[:120]}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="")
    args = ap.parse_args()
    session, bucket = session_and_bucket(args.profile)
    ensure_table(session, bucket, TABLE, COLUMNS)
    s3 = session.client("s3")
    days = days_to_parse(s3, bucket, datetime.now(timezone.utc).date())
    total = 0
    for day in days:
        df = summarize(read_day(s3, bucket, day), day)
        write(df, session, bucket, TABLE, COLUMNS, mode="overwrite_partitions")
        total += len(df)
    print(f"lane closures: {len(days)} days parsed, {total:,} closure-day rows written")


if __name__ == "__main__":
    main()
