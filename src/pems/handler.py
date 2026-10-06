"""Lambda: download one district-day of PeMS station data and mark it paired.

Called by the daily workflow with {"district": 7, "date": "2026-09-29"}. Returns
status "paired", or "not_available" if PeMS hasn't published that day's file yet.
Errors are raised so the workflow can retry and record the failure.
"""

import json
import tempfile
from datetime import date
from functools import cache
from pathlib import Path

import boto3

from common.config import load
from common.lake import put_file, write_bronze
from pems import pairing
from pems.client import PemsClient
from pems.parse import file_date, parse_meta, parse_station_5min, station_file_name


def main(event, context):
    settings = load()
    district, day = int(event["district"]), date.fromisoformat(event["date"])
    client = PemsClient(*_credentials())
    client.login()

    name = station_file_name(district, day)
    url = client.list_files(district, "station_5min", day.year).get(name)
    if url is None:
        return _log({"district": district, "date": day.isoformat(), "status": "not_available"})

    with tempfile.TemporaryDirectory() as tmp:
        path = client.download(url, Path(tmp) / name)
        put_file(settings.bucket, f"raw/pems/district={district:02d}/type=station_5min/{name}", path)
        df = parse_station_5min(path, district, day)
        # Replacing the partition makes a retry after a partial failure safe.
        write_bronze(df, settings.bucket, settings.glue_db, "pems_station_5min", ["district", "dt"],
                     mode="overwrite_partitions")
        meta = _sync_meta(settings, client, district, day.year, Path(tmp))

    pairing.put(settings, district, day, "paired", rows=len(df))
    return _log({"district": district, "date": day.isoformat(), "status": "paired",
                 "rows": len(df), "meta": meta})


def _sync_meta(settings, client: PemsClient, district: int, year: int, tmp: Path) -> str | None:
    """Load the newest station-metadata file if it isn't loaded yet. Returns its name."""
    files = client.list_files(district, "meta", year) or client.list_files(district, "meta", year - 1)
    if not files:
        return None
    newest = max(files, key=file_date)
    if newest == pairing.loaded_meta(settings, district):
        return newest
    path = client.download(files[newest], tmp / newest)
    put_file(settings.bucket, f"raw/pems/district={district:02d}/type=meta/{newest}", path)
    write_bronze(parse_meta(path, district, file_date(newest)), settings.bucket, settings.glue_db,
                 "pems_stations", ["district", "dt"], mode="overwrite_partitions")
    pairing.mark_meta(settings, district, newest)
    return newest


@cache
def _credentials() -> tuple[str, str]:
    """PeMS login, stored as SecureString parameters (never in code or the template)."""
    ssm = boto3.client("ssm")
    values = [ssm.get_parameter(Name=f"/traffic/pems/{n}", WithDecryption=True)["Parameter"]["Value"]
              for n in ("username", "password")]
    return values[0], values[1]


def _log(summary: dict) -> dict:
    print(json.dumps(summary))
    return summary
