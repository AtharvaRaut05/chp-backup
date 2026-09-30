"""Lambda entry point: download the CHP feed, save the raw snapshot, write parsed Parquet tables."""

import json
from datetime import datetime, timezone

from chp.fetch import get_feed
from chp.parse import parse_snapshot
from common.config import load
from common.lake import put_raw, write_bronze


def main(event, context):
    settings = load()
    now = datetime.now(timezone.utc)

    body = get_feed(settings.feed_url)
    key = put_raw(settings.bucket, "chp", f"{now:%Y-%m-%dT%H-%M-%SZ}.xml", body, now)

    snap = parse_snapshot(body, now, settings.dispatch_ids)
    tables = {"chp_incidents": snap.incidents, "chp_details": snap.details, "chp_units": snap.units}
    for table, df in tables.items():
        write_bronze(df, settings.bucket, settings.glue_db, table, ["dt"])

    summary = {"bytes": len(body), "key": key, **{t: len(df) for t, df in tables.items()}}
    print(json.dumps(summary))
    return summary
