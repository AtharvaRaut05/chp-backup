"""Hourly rainfall for every place and day with CHP incidents -> bronze table weather_hourly.

Rain is one of the biggest drivers of crashes and backups, and it's known in real time, so it can be
a model input. Source: Open-Meteo (https://open-meteo.com), free for non-commercial use, no key.

Places are 0.25-degree grid cells (about 25 km), roughly the resolution of the weather models.
The job asks Athena which (cell, day) pairs have CHP incidents but no weather yet, so the first run
backfills everything and later runs fetch only the new day. It runs before dbt in the daily job; if
the weather service is down, it warns and the rest of the job carries on (rain features are then
missing for those days, and the next run fills them in).

    python jobs/weather.py              # uses the job's AWS role (or the default credentials)
    python jobs/weather.py --profile traffic

    python jobs/weather.py --profile traffic --refetch    # delete everything stored and fetch it again

Stored columns: grid_lat, grid_lon, ts (UTC, end of the hour), precipitation_mm (total over the hour
ending at ts), fetched_at (when it was downloaded), dt (Pacific day it was fetched for; partition).
Only hours that had ended when they were downloaded are kept: later hours in the response are
forecasts. The same hour can be fetched again by a later run; the staging model keeps the newest.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pandas as pd

DATABASE = "traffic_bronze"
TABLE = "weather_hourly"
CELLS_PER_REQUEST = 50
START = date(2026, 9, 28)            # first day of CHP collection
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"          # recent days (up to ~3 months back)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"    # older days (reanalysis, a few days' lag)
ARCHIVE_AFTER_DAYS = 60
# Open-Meteo counts each location in a request as a call and limits calls per minute (and per day).
# 50 locations every 6 seconds stays under the per-minute limit; a 429 waits out the minute and retries.
PAUSE_SECONDS = 6
COLUMNS = {"grid_lat": "double", "grid_lon": "double", "ts": "timestamp", "precipitation_mm": "double",
           "fetched_at": "timestamp"}


# ---------- what to fetch ----------

def plan(needed: pd.DataFrame, have: pd.DataFrame) -> list[tuple[date, list[tuple[float, float]]]]:
    """(day, cells) batches for every needed (cell, day) not already stored, at most CELLS_PER_REQUEST per batch."""
    key = ["grid_lat", "grid_lon", "day"]
    missing = needed[key].drop_duplicates().merge(have[key].drop_duplicates(), how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].sort_values(key)
    batches = []
    for day, g in missing.groupby("day"):
        cells = list(zip(g["grid_lat"], g["grid_lon"]))
        batches += [(pd.Timestamp(day).date(), cells[i:i + CELLS_PER_REQUEST])
                    for i in range(0, len(cells), CELLS_PER_REQUEST)]
    return batches


def request_url(day: date, cells: list[tuple[float, float]], today: date) -> str:
    """Hourly precipitation in UTC covering the Pacific day (07:00 or 08:00 UTC to the same next day)
    and the 24 hours before it, which the rain_last_24h feature looks back over."""
    base = ARCHIVE_URL if (today - day).days > ARCHIVE_AFTER_DAYS else FORECAST_URL
    params = {
        "latitude": ",".join(f"{lat:.2f}" for lat, _ in cells),
        "longitude": ",".join(f"{lon:.2f}" for _, lon in cells),
        "hourly": "precipitation",
        "start_date": (day - timedelta(days=1)).isoformat(),
        "end_date": (day + timedelta(days=1)).isoformat(),
        "timezone": "GMT",
        "timeformat": "unixtime",
    }
    return f"{base}?{urllib.parse.urlencode(params)}"


def parse(body: bytes, day: date, cells: list[tuple[float, float]], fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Open-Meteo returns one object for one location and a list for several, in request order.
    Each hourly precipitation value is the total over the PRECEDING hour, so its timestamp is the
    end of that hour. Hours that hadn't ended at fetched_at are forecasts and are dropped."""
    data = json.loads(body)
    data = data if isinstance(data, list) else [data]
    if len(data) != len(cells):
        raise ValueError(f"asked for {len(cells)} locations, got {len(data)}")
    frames = []
    for (lat, lon), loc in zip(cells, data):
        hourly = loc["hourly"]
        frames.append(pd.DataFrame({"grid_lat": lat, "grid_lon": lon,
                                    "ts": pd.to_datetime(hourly["time"], unit="s", utc=True),
                                    "precipitation_mm": pd.to_numeric(pd.Series(hourly["precipitation"]),
                                                                      errors="coerce")}))
    out = pd.concat(frames, ignore_index=True)
    out = out[out["ts"] <= fetched_at].copy()
    out["fetched_at"] = fetched_at
    out["dt"] = day.isoformat()
    return out


def get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "chp-backup/0.1 (student traffic research)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError as e:  # includes HTTP errors and dropped connections
            if attempt == retries:
                raise
            too_many = isinstance(e, urllib.error.HTTPError) and e.code == 429
            time.sleep(65 if too_many else 5 * (attempt + 1))


# ---------- AWS ----------

def needed_cells(session) -> pd.DataFrame:
    import awswrangler as wr

    return wr.athena.read_sql_query(f"""
        select distinct round(lat * 4) / 4 as grid_lat, round(lon * 4) / 4 as grid_lon,
               cast(date(log_time at time zone 'America/Los_Angeles') as varchar) as day
        from chp_incidents
        where lat is not null and lon is not null
          and dt >= '{START:%Y-%m-%d}'
          and date(log_time at time zone 'America/Los_Angeles') < date(current_timestamp at time zone 'America/Los_Angeles')
    """, database=DATABASE, workgroup="traffic", boto3_session=session, ctas_approach=False)


def stored_cells(session) -> pd.DataFrame:
    import awswrangler as wr

    empty = pd.DataFrame(columns=["grid_lat", "grid_lon", "day"])
    if not wr.catalog.does_table_exist(DATABASE, TABLE, boto3_session=session):
        return empty
    return wr.athena.read_sql_query(f"select distinct grid_lat, grid_lon, dt as day from {TABLE}",
                                    database=DATABASE, workgroup="traffic", boto3_session=session,
                                    ctas_approach=False)


def ensure_table(session, bucket: str) -> None:
    """Create the (empty) table on the first run, so dbt can read it even before any rows exist."""
    import awswrangler as wr

    if not wr.catalog.does_table_exist(DATABASE, TABLE, boto3_session=session):
        wr.catalog.create_parquet_table(database=DATABASE, table=TABLE, path=f"s3://{bucket}/bronze/{TABLE}/",
                                        columns_types=COLUMNS, partitions_types={"dt": "string"},
                                        boto3_session=session)


def drop_all(session, bucket: str) -> None:
    """Delete the stored weather (files and table), so the next fetch starts over."""
    import awswrangler as wr

    wr.s3.delete_objects(f"s3://{bucket}/bronze/{TABLE}/", boto3_session=session)
    wr.catalog.delete_table_if_exists(database=DATABASE, table=TABLE, boto3_session=session)


def write(df: pd.DataFrame, session, bucket: str) -> None:
    import awswrangler as wr

    wr.s3.to_parquet(df=df, path=f"s3://{bucket}/bronze/{TABLE}/", dataset=True, mode="append",
                     database=DATABASE, table=TABLE, partition_cols=["dt"], boto3_session=session,
                     dtype={"ts": "timestamp", "fetched_at": "timestamp"})


def main() -> None:
    import boto3

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="")
    ap.add_argument("--refetch", action="store_true", help="delete stored weather and fetch everything again")
    args = ap.parse_args()
    session = boto3.Session(profile_name=args.profile or None, region_name="us-west-2")
    bucket = os.environ.get("LAKE_BUCKET") or next(
        o["OutputValue"] for o in session.client("cloudformation").describe_stacks(StackName="chp-backup")
        ["Stacks"][0]["Outputs"] if o["OutputKey"] == "LakeBucket")
    if args.refetch:
        drop_all(session, bucket)
    ensure_table(session, bucket)
    batches = plan(needed_cells(session), stored_cells(session))
    today = date.today()
    rows, failed = 0, 0
    for i, (day, cells) in enumerate(batches):
        if i:
            time.sleep(PAUSE_SECONDS)
        try:
            df = parse(get(request_url(day, cells, today)), day, cells, pd.Timestamp.now(tz="UTC"))
        except (OSError, ValueError, KeyError) as e:
            failed += 1
            print(f"WARNING weather for {day} ({len(cells)} cells) not fetched: {str(e)[:200]}")
            continue
        write(df, session, bucket)
        rows += len(df)
    print(f"weather: {len(batches)} requests, {rows:,} hourly rows written, {failed} failed"
          + (" (failed days are fetched again on the next run)" if failed else ""))


if __name__ == "__main__":
    main()
