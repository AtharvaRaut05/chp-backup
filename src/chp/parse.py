"""Parse one CHP incident feed snapshot (sa.xml) into three DataFrames.

Pure function: bytes in, DataFrames out. No network or AWS calls, so it is fully testable.

Feed structure: State > Center > Dispatch > Log (incident) > LogDetails > details / units.
Every text value is wrapped in literal double quotes, and times are Pacific local time.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

PACIFIC = ZoneInfo("America/Los_Angeles")

# Leading "[27]" index and "[Appended, 16:38:43]" markers on each note.
_NOTE_PREFIX = re.compile(r"^\s*(?:\[(?:\d+|Appended, \d{2}:\d{2}:\d{2})\]\s*)+")
_NOTE_SEQ = re.compile(r"^\s*\[(\d+)\]")
_SHARED_SUFFIX = re.compile(r"(?:\s*\[Shared\])+\s*$")

INCIDENT_COLUMNS = {
    "log_id": "string",
    "center_id": "string",
    "dispatch_id": "string",
    "log_time": "datetime64[ns, UTC]",
    "log_type": "string",
    "log_type_code": "string",
    "log_type_desc": "string",
    "location": "string",
    "location_desc": "string",
    "area": "string",
    "lat": "float64",
    "lon": "float64",
}
DETAIL_COLUMNS = {
    "log_id": "string",
    "detail_seq": "Int64",
    "detail_time": "datetime64[ns, UTC]",
    "detail_text": "string",
    "detail_raw": "string",
}
UNIT_COLUMNS = {
    "log_id": "string",
    "unit_seq": "Int64",
    "unit_time": "datetime64[ns, UTC]",
    "unit_text": "string",
}
SNAPSHOT_COLUMNS = {"polled_at": "datetime64[ns, UTC]", "dt": "string"}


@dataclass
class Snapshot:
    incidents: pd.DataFrame
    details: pd.DataFrame
    units: pd.DataFrame


def parse_snapshot(xml: bytes, polled_at: datetime, dispatch_ids: tuple[str, ...]) -> Snapshot:
    """Parse a feed snapshot, keeping only incidents from the given Dispatch IDs."""
    root = ET.fromstring(xml)
    keep = set(dispatch_ids)
    incidents, details, units = [], [], []

    for center in root.iter("Center"):
        for dispatch in center.iter("Dispatch"):
            if dispatch.get("ID") not in keep:
                continue
            for log in dispatch.iter("Log"):
                log_id = log.get("ID")
                incidents.append(_incident_row(log, center.get("ID"), dispatch.get("ID")))

                for d in log.iter("details"):
                    raw = _text(d, "IncidentDetail")
                    if raw is None:  # empty <details/> element
                        continue
                    details.append({
                        "log_id": log_id,
                        "detail_seq": _note_seq(raw),
                        "detail_time": _parse_chp_time(_text(d, "DetailTime")),
                        "detail_text": _clean_note(raw),
                        "detail_raw": raw,
                    })

                for i, u in enumerate(log.iter("units")):
                    units.append({
                        "log_id": log_id,
                        "unit_seq": i,
                        "unit_time": _parse_chp_time(_text(u, "UnitTime")),
                        "unit_text": _text(u, "UnitDetail"),
                    })

    polled_utc = polled_at.astimezone(timezone.utc)
    return Snapshot(
        incidents=_frame(incidents, INCIDENT_COLUMNS, polled_utc),
        details=_frame(details, DETAIL_COLUMNS, polled_utc).drop_duplicates(
            subset=["log_id", "detail_seq", "detail_text"], ignore_index=True
        ),
        units=_frame(units, UNIT_COLUMNS, polled_utc),
    )


def _incident_row(log: ET.Element, center_id: str, dispatch_id: str) -> dict:
    log_type = _text(log, "LogType")
    code, desc = _split_log_type(log_type)
    latlon = _parse_latlon(_text(log, "LATLON"))
    return {
        "log_id": log.get("ID"),
        "center_id": center_id,
        "dispatch_id": dispatch_id,
        "log_time": _parse_chp_time(_text(log, "LogTime")),
        "log_type": log_type,
        "log_type_code": code,
        "log_type_desc": desc,
        "location": _text(log, "Location"),
        "location_desc": _text(log, "LocationDesc"),
        "area": _text(log, "Area"),
        "lat": latlon[0] if latlon else None,
        "lon": latlon[1] if latlon else None,
    }


def _frame(rows: list[dict], columns: dict, polled_at: datetime) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(columns))
    df["polled_at"] = polled_at
    df["dt"] = polled_at.strftime("%Y-%m-%d")
    return df.astype({**columns, **SNAPSHOT_COLUMNS})


def _text(el: ET.Element, tag: str) -> str | None:
    """Child text with the surrounding quotes stripped; None if missing or empty."""
    value = el.findtext(tag)
    if value is None:
        return None
    value = value.strip().strip('"').strip()
    return value or None


def _parse_chp_time(s: str | None) -> datetime | None:
    """'Sep 26 2026  4:28PM' (Pacific) -> UTC datetime.

    During the November DST fall-back hour, 1:00-1:59AM is ambiguous; fold=0 treats it
    as daylight time (the first occurrence).
    """
    if not s:
        return None
    local = datetime.strptime(" ".join(s.split()), "%b %d %Y %I:%M%p")
    return local.replace(tzinfo=PACIFIC).astimezone(timezone.utc)


def _parse_latlon(s: str | None) -> tuple[float, float] | None:
    """'38577615:121415468' -> (38.577615, -121.415468). '0:0' or malformed -> None."""
    if not s or ":" not in s:
        return None
    try:
        lat, lon = (int(part) / 1_000_000 for part in s.split(":", 1))
    except ValueError:
        return None
    if lat == 0 or lon == 0:
        return None
    return lat, -abs(lon)  # feed omits the minus sign; California is west longitude


def _split_log_type(s: str | None) -> tuple[str | None, str | None]:
    """'1183-Trfc Collision-Unkn Inj' -> ('1183', 'Trfc Collision-Unkn Inj')."""
    if not s:
        return None, None
    code, sep, desc = s.partition("-")
    return (code.strip(), desc.strip()) if sep else (None, s)


def _note_seq(raw: str) -> int | None:
    m = _NOTE_SEQ.match(raw)
    return int(m.group(1)) if m else None


def _clean_note(raw: str) -> str:
    """Drop index/appended prefixes and [Shared] suffixes; collapse whitespace.

    Tags that carry meaning, like [Notification] or [Rotation Request Comment], are kept.
    """
    text = _NOTE_PREFIX.sub("", raw)
    text = _SHARED_SUFFIX.sub("", text)
    return " ".join(text.split())
