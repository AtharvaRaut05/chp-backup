"""Daily workflow steps around the PeMS download (run by Step Functions).

plan   -> decide which district-days need PeMS data:
          skip days already done, mark days with no traffic incidents as no_incidents,
          and return the rest (including earlier failures, retried every day).
fetch  -> pems.handler.main, once per district-day (retried by the workflow).
report -> record failures; after the lookback window, give up and mark the day unpaired
          so its incidents are excluded from analysis. Fails the run if anything failed,
          which triggers the alert email.
"""

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from common.config import DISPATCH_DISTRICT, TRAFFIC_TYPE_CODES, Settings, load
from pems import pairing

PACIFIC = ZoneInfo("America/Los_Angeles")


def plan(event, context):
    settings = load()
    today = datetime.now(PACIFIC).date()
    days = [today - timedelta(days=n) for n in range(1, settings.lookback_days + 1)]
    counts = incident_counts(settings, min(days), max(days))

    todo, skipped = [], 0
    for day in days:
        for district in settings.pems_districts:
            status = (pairing.get(settings, district, day) or {}).get("status")
            if status in pairing.DONE:
                continue
            if counts.get((district, day), 0) == 0:
                pairing.put(settings, district, day, "no_incidents")
                skipped += 1
            else:
                todo.append({"district": district, "date": day.isoformat()})

    return _log({"todo": todo, "skipped_no_incidents": skipped})


def report(event, context):
    settings = load()
    oldest_allowed = datetime.now(PACIFIC).date() - timedelta(days=settings.lookback_days)
    failed, gave_up = [], []

    for r in event.get("results", []):
        if r["status"] == "paired":
            continue
        district, day = int(r["district"]), date.fromisoformat(r["date"])
        if day <= oldest_allowed:
            pairing.put(settings, district, day, "unpaired", reason=r["status"])
            gave_up.append(r)
        else:
            error = r.get("error", {}).get("Cause", "")[:500] if r["status"] == "failed" else None
            pairing.put(settings, district, day, "pending", **({"error": error} if error else {}))
        if r["status"] == "failed":
            failed.append(r)

    summary = _log({"paired": sum(r["status"] == "paired" for r in event.get("results", [])),
                    "failed": len(failed), "gave_up": len(gave_up)})
    if failed:
        raise RuntimeError(f"PeMS download failed for {len(failed)} district-days: "
                           + ", ".join(f"d{r['district']:02d} {r['date']}" for r in failed))
    return summary


def incident_counts(settings: Settings, start: date, end: date) -> dict[tuple[int, date], int]:
    """Traffic incidents per (Caltrans district, Pacific date), from the CHP bronze table."""
    import awswrangler as wr  # provided by the AWS SDK for pandas Lambda layer

    codes = ", ".join(f"'{c}'" for c in TRAFFIC_TYPE_CODES)
    sql = f"""
        SELECT dispatch_id,
               CAST(date(log_time AT TIME ZONE 'America/Los_Angeles') AS varchar) AS day,
               count(DISTINCT log_id) AS n
        FROM chp_incidents
        WHERE dt BETWEEN '{start - timedelta(days=1)}' AND '{end + timedelta(days=1)}'
          AND lat IS NOT NULL
          AND (log_type_code IN ({codes}) OR log_type = 'SIG Alert')
        GROUP BY 1, 2
    """
    df = wr.athena.read_sql_query(sql, database=settings.glue_db, workgroup=settings.athena_workgroup,
                                  ctas_approach=False)
    counts: dict[tuple[int, date], int] = {}
    for dispatch_id, day, n in df.itertuples(index=False):
        district = DISPATCH_DISTRICT.get(dispatch_id)
        if district is not None:
            k = (district, date.fromisoformat(day))
            counts[k] = counts.get(k, 0) + int(n)
    return counts


def _log(summary: dict) -> dict:
    print(json.dumps(summary))
    return summary
