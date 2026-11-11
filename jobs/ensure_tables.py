"""Create any missing bronze table the dbt models read, empty, before dbt runs.

Each source's own step creates its table on its first run, but a step can fail first (service down,
no API key yet) and the TomTom table is otherwise only created by the Lambda's first write. dbt fails
on a missing source table, which would stop the whole daily rebuild.

    python jobs/ensure_tables.py --profile traffic
"""

import argparse

import lane_closures
import road_layout
from lake_job import ensure_table, session_and_bucket

# Same columns and types as src/tomtom/speed_lookup.py COLUMNS writes (tests check they match).
TOMTOM_COLUMNS = {"log_id": "string", "requested_at": "timestamp", "lat": "double", "lon": "double",
                  "frc": "string", "current_speed_kmh": "double", "free_flow_speed_kmh": "double",
                  "current_travel_time_s": "double", "free_flow_travel_time_s": "double", "confidence": "double",
                  "road_closure": "boolean", "segment_bearing_deg": "double", "segment_km": "double"}
# Same as src/pems/incidents.py INCIDENT_TYPES / DETAIL_TYPES (written by scripts/backfill_chp_history.py).
HIST_INCIDENT_COLUMNS = {"incident_id": "string", "cc_code": "string", "log_time": "timestamp",
                         "log_type_code": "string", "description": "string", "location": "string", "area": "string",
                         "lat": "double", "lon": "double", "district": "int", "freeway": "int", "direction": "string",
                         "abs_pm": "double", "duration_min": "double"}
HIST_DETAIL_COLUMNS = {"incident_id": "string", "detail_seq": "bigint", "detail_time": "timestamp",
                       "seen_at": "timestamp", "detail_text": "string"}
TABLES = {
    lane_closures.TABLE: lane_closures.COLUMNS,
    road_layout.TABLE: road_layout.COLUMNS,
    "tomtom_flow": TOMTOM_COLUMNS,
    "hist_chp_incidents": HIST_INCIDENT_COLUMNS,
    "hist_chp_details": HIST_DETAIL_COLUMNS,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="")
    args = ap.parse_args()
    session, bucket = session_and_bucket(args.profile)
    for table, columns in TABLES.items():
        ensure_table(session, bucket, table, columns)
    print(f"bronze tables present: {', '.join(TABLES)}")


if __name__ == "__main__":
    main()
