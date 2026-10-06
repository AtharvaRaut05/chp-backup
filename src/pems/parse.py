"""Parse PeMS Clearinghouse files into typed DataFrames.

station_5min: gzip CSV with no header. 12 summary columns, then 8 repeating per-lane groups
(52 columns total). Timestamps are Pacific local time. Only mainline (ML) rows are kept.
meta: tab-separated text with a header row, one row per station.
"""

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

PACIFIC = ZoneInfo("America/Los_Angeles")

SUMMARY_COLUMNS = [
    "ts_local", "station_id", "district", "freeway", "direction", "lane_type",
    "station_length", "samples", "pct_observed", "total_flow", "avg_occupancy", "avg_speed",
]
LANE_FIELDS = ["samples", "flow", "avg_occ", "avg_speed", "observed"]
ALL_COLUMNS = SUMMARY_COLUMNS + [f"lane{n}_{f}" for n in range(1, 9) for f in LANE_FIELDS]

STATION_TYPES = {
    "ts_local": pa.timestamp("s"),
    "station_id": pa.int64(),
    "district": pa.int64(),
    "freeway": pa.int64(),
    "direction": pa.string(),
    "lane_type": pa.string(),
    "station_length": pa.float64(),
    "samples": pa.int64(),
    "pct_observed": pa.float64(),
    "total_flow": pa.float64(),
    "avg_occupancy": pa.float64(),
    "avg_speed": pa.float64(),
}

META_COLUMNS = {
    "ID": "station_id", "Fwy": "freeway", "Dir": "direction", "District": "district",
    "County": "county", "City": "city", "State_PM": "state_pm", "Abs_PM": "abs_pm",
    "Latitude": "lat", "Longitude": "lon", "Length": "length", "Type": "type",
    "Lanes": "lanes", "Name": "name",
}


def parse_station_5min(path: Path, district: int, day: date) -> pd.DataFrame:
    table = pv.read_csv(
        pa.input_stream(str(path), compression="gzip"),
        read_options=pv.ReadOptions(column_names=ALL_COLUMNS),
        convert_options=pv.ConvertOptions(
            include_columns=SUMMARY_COLUMNS,
            column_types=STATION_TYPES,
            timestamp_parsers=["%m/%d/%Y %H:%M:%S"],
        ),
    )
    df = table.to_pandas()
    df = df[df["lane_type"] == "ML"].drop(columns=["lane_type", "district"])
    # The repeated 1 AM hour at the November DST change is ambiguous in local time; drop it.
    df["ts"] = df.pop("ts_local").dt.tz_localize(PACIFIC, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    df = df.dropna(subset=["ts"])
    df["district"] = f"{district:02d}"
    df["dt"] = day.isoformat()
    return df.astype({
        "ts": "datetime64[ns, UTC]", "station_id": "int64", "freeway": "Int64", "direction": "string",
        "samples": "Int64", "district": "string", "dt": "string",
    }).reset_index(drop=True)


def parse_meta(path: Path, district: int, file_day: date) -> pd.DataFrame:
    text_cols = {"City": "string", "Name": "string", "State_PM": "string"}  # State_PM can be "R12.3"
    df = pd.read_csv(path, sep="\t", usecols=list(META_COLUMNS), dtype=text_cols)
    df = df.rename(columns=META_COLUMNS).drop(columns=["district"])
    df["district"] = f"{district:02d}"
    df["dt"] = file_day.isoformat()
    return df.astype({
        "station_id": "int64", "freeway": "Int64", "direction": "string", "county": "Int64",
        "city": "string", "state_pm": "string", "type": "string", "lanes": "Int64", "name": "string",
        "district": "string", "dt": "string",
    })


def station_file_name(district: int, day: date) -> str:
    return f"d{district:02d}_text_station_5min_{day:%Y_%m_%d}.txt.gz"


def file_date(file_name: str) -> date:
    """'d07_text_meta_2026_08_25.txt' -> date(2026, 8, 25)."""
    y, m, d = file_name.split(".")[0].split("_")[-3:]
    return date(int(y), int(m), int(d))
