"""Lambda entry point: poll the CHP feed and write only new or changed incidents.

Steps: fetch -> parse -> compare with saved state -> (if changed) save raw, write Parquet
for changed incidents and detected closures -> save state. State is saved last, so a
failure part-way is retried on the next run (writes are at-least-once; dbt dedupes).
"""

import json
from datetime import datetime, timezone

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
from common.config import load
from common.lake import put_raw, write_bronze


def main(event, context):
    settings = load()
    now = datetime.now(timezone.utc)

    body = get_feed(settings.feed_url)
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

    save_state(settings.state_table, State(current, incidents, state.version))
    return _log({"status": "changed", "key": key, "open": len(incidents), "changed": len(changed),
                 "closed": len(closures), **{t: len(df) for t, df in tables.items()}})


def _log(summary: dict) -> dict:
    print(json.dumps(summary))
    return summary
