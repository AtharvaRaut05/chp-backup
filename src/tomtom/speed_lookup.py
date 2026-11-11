"""Live traffic speed where a crash was just reported (TomTom Traffic Flow, Flow Segment Data).

The CHP poller calls this (asynchronously) for each new traffic incident on a freeway. One request
returns the current and free-flow speed on the road segment nearest the point, so the model gets
"how slow is it there right now" at report time, even where PeMS sensors are missing, and without
waiting a day for PeMS. Zoom 10 only returns major roads, so the point snaps to the freeway, but it
can land on the opposite carriageway: the segment's heading is saved so dbt can check the direction.

Free tier: 20,000 segment requests a month, so a daily cap (DynamoDB counter) keeps it under that.
The key lives in Parameter Store (/traffic/tomtom/api_key); without it this does nothing.
Docs: https://docs.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from functools import cache

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError

from common.config import load
from common.lake import write_bronze

URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
KEY_PARAM = "/traffic/tomtom/api_key"
DAILY_CAP = int(os.environ.get("TOMTOM_DAILY_CAP", "600"))  # 600 x 31 days < 20,000 free requests a month
TABLE = "tomtom_flow"
COLUMNS = {
    "log_id": "string", "requested_at": "datetime64[ns, UTC]", "lat": "float64", "lon": "float64",
    "frc": "string", "current_speed_kmh": "float64", "free_flow_speed_kmh": "float64",
    "current_travel_time_s": "float64", "free_flow_travel_time_s": "float64", "confidence": "float64",
    "road_closure": "boolean", "segment_bearing_deg": "float64", "segment_km": "float64", "dt": "string",
}


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass heading from point 1 to point 2, degrees (0 = north, 90 = east)."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360


def km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def parse(body: dict, incident: dict, requested_at: datetime) -> dict:
    seg = body["flowSegmentData"]
    pts = [(c["latitude"], c["longitude"]) for c in seg.get("coordinates", {}).get("coordinate", [])]
    heading = bearing(*pts[0], *pts[-1]) if len(pts) >= 2 and pts[0] != pts[-1] else None
    length = sum(km(*a, *b) for a, b in zip(pts, pts[1:])) if len(pts) >= 2 else None
    return {
        "log_id": incident["log_id"], "requested_at": requested_at, "lat": incident["lat"], "lon": incident["lon"],
        "frc": seg.get("frc"), "current_speed_kmh": seg.get("currentSpeed"), "free_flow_speed_kmh": seg.get("freeFlowSpeed"),
        "current_travel_time_s": seg.get("currentTravelTime"), "free_flow_travel_time_s": seg.get("freeFlowTravelTime"),
        "confidence": seg.get("confidence"), "road_closure": seg.get("roadClosure"),
        "segment_bearing_deg": heading, "segment_km": length,
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "dt"])
    df["requested_at"] = pd.to_datetime(df["requested_at"], utc=True)
    df["dt"] = df["requested_at"].dt.strftime("%Y-%m-%d")
    return df.astype(COLUMNS)


def take_quota(table_name: str, day: str) -> bool:
    """Count one request against today's cap; False once the cap is reached."""
    try:
        _table(table_name).update_item(
            Key={"pk": f"tomtom_quota#{day}"}, UpdateExpression="ADD n :one",
            ConditionExpression="attribute_not_exists(n) OR n < :cap",
            ExpressionAttributeValues={":one": 1, ":cap": DAILY_CAP})
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def fetch(key: str, lat: float, lon: float) -> dict:
    resp = requests.get(URL, timeout=10, params={"key": key, "point": f"{lat:.6f},{lon:.6f}", "unit": "kmph"})
    resp.raise_for_status()
    return resp.json()


def main(event, context):
    incidents = event.get("incidents", [])
    key = api_key()
    if not key or not incidents:
        return _log({"status": "no key" if not key else "nothing to look up"})
    settings = load()
    rows, failed, capped = [], 0, 0
    for inc in incidents:
        now = datetime.now(timezone.utc)
        if not take_quota(settings.state_table, now.strftime("%Y-%m-%d")):
            capped = len(incidents) - len(rows) - failed
            break
        try:
            rows.append(parse(fetch(key, inc["lat"], inc["lon"]), inc, now))
        except (requests.RequestException, ValueError, KeyError, IndexError) as e:
            failed += 1
            print(f"WARNING lookup for {inc['log_id']} failed: {str(e)[:150]}")
    write_bronze(frame(rows), settings.bucket, settings.glue_db, TABLE, ["dt"])
    summary = _log({"status": "done", "looked_up": len(rows), "failed": failed, "over_daily_cap": capped})
    if failed and not rows:  # every lookup failed (bad key, service down): let the error alarm see it
        raise RuntimeError(f"all {failed} speed lookups failed")
    return summary


@cache
def api_key() -> str | None:
    ssm = boto3.client("ssm")
    try:
        return ssm.get_parameter(Name=KEY_PARAM, WithDecryption=True)["Parameter"]["Value"]
    except ssm.exceptions.ParameterNotFound:
        return None


@cache
def _table(name: str):
    return boto3.resource("dynamodb").Table(name)


def _log(summary: dict) -> dict:
    print(json.dumps(summary))
    return summary
