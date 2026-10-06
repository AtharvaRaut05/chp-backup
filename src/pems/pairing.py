"""Pairing status: whether each district-day of CHP incidents has matching PeMS traffic data.

Statuses:
  pending       incidents exist, PeMS not loaded yet (file not published, or download failed)
  paired        PeMS loaded; incidents for this district-day are usable for analysis
  no_incidents  no traffic incidents that day, so PeMS was not downloaded
  unpaired      PeMS still missing after the lookback window; incidents excluded from analysis

DynamoDB holds the current status (read by the orchestrator). Every change is also appended
to the bronze `pems_pairing` table so Athena/dbt can filter incidents to paired days.
"""

from datetime import date, datetime, timezone
from functools import cache

import boto3
import pandas as pd

from common.config import Settings
from common.lake import write_bronze

DONE = {"paired", "no_incidents", "unpaired"}


def key(district: int, day: date) -> str:
    return f"{district:02d}#{day.isoformat()}"


def get(settings: Settings, district: int, day: date) -> dict | None:
    return _table(settings.pairing_table).get_item(Key={"pk": key(district, day)}).get("Item")


def put(settings: Settings, district: int, day: date, status: str, **detail) -> None:
    now = datetime.now(timezone.utc)
    prev = get(settings, district, day) or {}
    attempts = int(prev.get("attempts", 0)) + (1 if status == "pending" and detail.get("error") else 0)
    item = {"pk": key(district, day), "status": status, "updated_at": now.isoformat(), "attempts": attempts,
            **{k: str(v) for k, v in detail.items()}}
    _table(settings.pairing_table).put_item(Item=item)

    row = pd.DataFrame([{"district": f"{district:02d}", "day": day.isoformat(), "status": status,
                         "attempts": attempts, "detail": str(detail) if detail else None,
                         "updated_at": now, "dt": now.strftime("%Y-%m-%d")}])
    row = row.astype({"district": "string", "day": "string", "status": "string", "attempts": "int64",
                      "detail": "string", "updated_at": "datetime64[ns, UTC]", "dt": "string"})
    write_bronze(row, settings.bucket, settings.glue_db, "pems_pairing", ["dt"])


def loaded_meta(settings: Settings, district: int) -> str | None:
    """Name of the newest station-metadata file already loaded for a district."""
    item = _table(settings.pairing_table).get_item(Key={"pk": f"meta#{district:02d}"}).get("Item")
    return item["file_name"] if item else None


def mark_meta(settings: Settings, district: int, file_name: str) -> None:
    _table(settings.pairing_table).put_item(Item={"pk": f"meta#{district:02d}", "file_name": file_name})


@cache
def _table(name: str):
    return boto3.resource("dynamodb").Table(name)
