"""Parse PeMS's historical CHP incident files (Clearinghouse type chp_incidents_day).

One zip per day, statewide, holding two gzip CSVs with no header:
  all_text_chp_incident_day_YYYY_MM_DD.txt.gz      one row per incident
  all_text_chp_incident_det_day_YYYY_MM_DD.txt.gz  one row per dispatcher note or unit event

Times are Pacific local, to the minute. The live feed shows the same incidents and notes as they
happen; these files are CHP's record of them afterwards, so they cover any past day. Two
differences matter for features:
  - a note's time is when CHP stamped it, which can be before it was posted; "[Appended hh:mm:ss]"
    markers give the posting time when present, so seen_at uses the later of the two
  - a note can belong to an incident logged the day before (it is in that day's incident file)
"""

from __future__ import annotations

import gzip
import io
import re
import zipfile
from datetime import date

import pandas as pd

PACIFIC = "America/Los_Angeles"
INCIDENT_COLUMNS = ["incident_id", "cc_code", "incident_number", "log_time_local", "description", "location",
                    "area", "zoom_map", "tb_xy", "lat", "lon", "district", "county_fips", "city_fips",
                    "freeway", "direction", "state_pm", "abs_pm", "severity", "duration_min"]
# Column types of the bronze tables hist_chp_incidents / hist_chp_details (Glue/Athena names)
INCIDENT_TYPES = {"incident_id": "string", "cc_code": "string", "log_time": "timestamp", "log_type_code": "string",
                  "description": "string", "location": "string", "area": "string", "lat": "double", "lon": "double",
                  "district": "int", "freeway": "int", "direction": "string", "abs_pm": "double",
                  "duration_min": "double"}
DETAIL_TYPES = {"incident_id": "string", "detail_seq": "bigint", "detail_time": "timestamp", "seen_at": "timestamp",
                "detail_text": "string"}
_DETAIL_START = re.compile(r"^(\d+),(\d+),(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}),(.*)$")
_APPENDED = re.compile(r"\[Appended\s+(\d{2}:\d{2}:\d{2})\]")
_TYPE_CODE = re.compile(r"^(\w+)-")


def _pacific(local: pd.Series) -> pd.Series:
    """Pacific wall-clock times -> UTC. The repeated 1 AM hour when clocks fall back is read as
    daylight time (off by an hour for that one hour a year); the skipped spring hour moves forward."""
    return local.dt.tz_localize(PACIFIC, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")


def _members(raw: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return {n: gzip.decompress(z.read(n)).decode("utf-8", "replace") for n in z.namelist()}


def parse_incidents(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), header=None, names=INCIDENT_COLUMNS, dtype=str, keep_default_na=False)
    out = pd.DataFrame({
        "incident_id": df["incident_id"],
        "cc_code": df["cc_code"],
        "log_time": _pacific(pd.to_datetime(df["log_time_local"], format="%m/%d/%Y %H:%M:%S")),
        "log_type_code": df["description"].str.extract(_TYPE_CODE, expand=False),
        "description": df["description"],
        "location": df["location"],
        "area": df["area"],
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": pd.to_numeric(df["lon"], errors="coerce"),
        "district": pd.to_numeric(df["district"], errors="coerce").astype("Int64"),
        "freeway": pd.to_numeric(df["freeway"], errors="coerce").astype("Int64"),
        "direction": df["direction"].str.strip().replace("", None),
        "abs_pm": pd.to_numeric(df["abs_pm"], errors="coerce"),
        "duration_min": pd.to_numeric(df["duration_min"], errors="coerce"),
    })
    return out.dropna(subset=["log_time"]).drop_duplicates("incident_id", keep="last")


def parse_details(text: str) -> pd.DataFrame:
    """Notes can contain commas and line breaks: a line that doesn't start a new record continues
    the previous one."""
    rows = []
    for line in text.splitlines():
        m = _DETAIL_START.match(line)
        if m:
            rows.append(list(m.groups()))
        elif rows:
            rows[-1][3] += " " + line.strip()
    df = pd.DataFrame(rows, columns=["incident_id", "detail_id", "ts_local", "detail_text"])
    stamped = pd.to_datetime(df["ts_local"], format="%m/%d/%Y %H:%M:%S")
    appended = df["detail_text"].str.extract(_APPENDED, expand=False)
    # "[Appended hh:mm:ss]" is the time of day it was posted; past midnight it belongs to the next day
    posted = pd.to_datetime(stamped.dt.strftime("%Y-%m-%d") + " " + appended, errors="coerce")
    posted = posted.where(posted >= stamped - pd.Timedelta(hours=12), posted + pd.Timedelta(days=1))
    df["detail_time"] = _pacific(stamped)
    df["seen_at"] = _pacific(posted.where(posted > stamped, stamped).fillna(stamped))
    df["detail_seq"] = pd.to_numeric(df["detail_id"], errors="coerce").astype("Int64")
    return df[["incident_id", "detail_seq", "detail_time", "seen_at", "detail_text"]]


def parse_day_zip(raw: bytes, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(incidents, details) from one day's zip, each with dt = the file's day."""
    members = _members(raw)
    inc_text = next(t for n, t in members.items() if "_det_" not in n)
    det_text = next(t for n, t in members.items() if "_det_" in n)
    incidents, details = parse_incidents(inc_text), parse_details(det_text)
    incidents["dt"] = details["dt"] = day.isoformat()
    return incidents, details


def file_name(day: date) -> str:
    return f"all_text_chp_incidents_day_{day:%Y_%m_%d}.txt.zip"
