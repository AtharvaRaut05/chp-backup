"""Lambda entry point: poll the CHP feed and write only new or changed incidents.

Steps: fetch -> parse -> compare with saved state -> (if changed) save raw, write Parquet
for changed incidents and detected closures -> save state -> ask for a live speed lookup at each
new freeway incident (tomtom/speed_lookup.py, run asynchronously). State is saved after the data, so a
failure part-way is retried on the next run (writes are at-least-once; dbt dedupes).
"""

import json
import os
import re
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

from chp.fetch import get_feed
from chp.parse import parse_snapshot
from chp.state import (
    State,
    closures_frame,
    diff,
    incident_hashes,
    load_state,
    save_state,
    snapshot_hash,
)
from common.config import TRAFFIC_TYPE_CODES, load
from common.lake import put_raw, write_bronze


def main(event, context):
    settings = load()
    body = get_feed(settings.feed_url)
    now = datetime.now(timezone.utc)  # after the download: when we actually saw this snapshot
    snap = parse_snapshot(body, now, settings.dispatch_ids)
    hashes = incident_hashes(snap)
    current = snapshot_hash(hashes)

    state = load_state(settings.state_table)
    feed_changed = current != state.snapshot_hash
    closing = any(int(v["missed"]) for v in state.incidents.values())  # absences still being counted
    if not feed_changed and not closing:
        return _log({"status": "unchanged", "open": len(hashes)})

    key = put_raw(settings.bucket, "chp", f"{now:%Y-%m-%dT%H-%M-%SZ}.xml", body, now) if feed_changed else None
    incidents, changed, closures = diff(state, hashes, now)

    tables = {
        "chp_incidents": snap.incidents[snap.incidents["log_id"].isin(changed)],
        "chp_details": snap.details[snap.details["log_id"].isin(changed)],
        "chp_units": snap.units[snap.units["log_id"].isin(changed)],
        "chp_closures": closures_frame(closures, now),
    }
    for table, df in tables.items():
        write_bronze(df, settings.bucket, settings.glue_db, table, ["dt"])

    try:
        save_state(settings.state_table, State(current, incidents, state.version))
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        # An overlapping run saved first. Its state wins; ours is dropped (dbt reads the latest version).
        return _log({"status": "superseded", "changed": len(changed)})
    looked_up = request_speed_lookups(new_freeway_incidents(snap, set(hashes) - set(state.incidents)))
    return _log({"status": "changed", "key": key, "open": len(incidents), "changed": len(changed),
                 "closed": len(closures), **{t: len(df) for t, df in tables.items()}, "speed_lookups": looked_up})


_FREEWAY = re.compile(r"\b(?:I|SR|US)-?\d{1,3}\b|\b[NSEW]B \d{1,3}\b")


def new_freeway_incidents(snap, new_ids: set[str]) -> list[dict]:
    """New traffic incidents (collisions, hazards, SigAlerts) on a freeway, with a location."""
    inc = snap.incidents[snap.incidents["log_id"].isin(new_ids)]
    text = (inc["location"].fillna("") + " " + inc["location_desc"].fillna("")).str.upper()
    keep = ((inc["log_type_code"].isin(TRAFFIC_TYPE_CODES) | (inc["log_type"] == "SIG Alert"))
            & text.str.contains(_FREEWAY) & inc["lat"].notna() & inc["lon"].notna())
    return [{"log_id": r.log_id, "lat": float(r.lat), "lon": float(r.lon)} for r in inc[keep].itertuples()]


def request_speed_lookups(incidents: list[dict]) -> int:
    """Hand new incidents to the speed lookup function without waiting for it. Never fails the poll."""
    function = os.environ.get("SPEED_LOOKUP_FUNCTION")
    if not function or not incidents:
        return 0
    try:
        boto3.client("lambda").invoke(FunctionName=function, InvocationType="Event",
                                      Payload=json.dumps({"incidents": incidents}).encode())
    except Exception as e:  # a lookup is nice to have; the poll's own data is already saved
        print(f"WARNING speed lookup not requested: {str(e)[:200]}")
        return 0
    return len(incidents)


def _log(summary: dict) -> dict:
    print(json.dumps(summary))
    return summary
