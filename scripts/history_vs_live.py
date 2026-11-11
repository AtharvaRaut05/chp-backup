"""Step 3 of the history backfill: how far can the PeMS history stand in for what the poller saw?

On the days both exist, matches each live freeway traffic incident to its PeMS record (same
freeway and direction, logged within 2 minutes, within 0.5 km) and each live dispatcher note to
the same note in the history (same text). Reports:

  - how many live incidents have a PeMS record (coverage), and how far apart the log times are
  - for matched notes, minutes from the history's time (seen_at) to when our poller first saw the
    note. Positive = the history says it was known earlier than it really was: features built from
    history would peek ahead by that much unless it is added back.

    python scripts/history_vs_live.py --start 2026-09-29 --end 2026-10-31
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATCH_MINUTES, MATCH_KM = 2, 0.5
_MARKERS = re.compile(r"\[(?:\d+|Appended[ ,]+\d{2}:\d{2}:\d{2}|Shared)\]")


def normalize(text) -> str:
    """Note text for matching: the live feed and the PeMS file differ in punctuation (PeMS drops
    commas) and markers, so compare letters and digits only."""
    text = _MARKERS.sub(" ", str(text or "")).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def match_incidents(live: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Nearest-in-time PeMS record on the same freeway and direction, within the limits."""
    pairs = live.merge(hist, on=["freeway", "direction"], suffixes=("", "_h"))
    pairs["minutes_apart"] = (pairs["log_time_h"] - pairs["log_time"]).dt.total_seconds() / 60
    pairs["km_apart"] = km(pairs["lat"], pairs["lon"], pairs["lat_h"], pairs["lon_h"])
    pairs = pairs[(pairs["minutes_apart"].abs() <= MATCH_MINUTES) & (pairs["km_apart"] <= MATCH_KM)]
    return pairs.sort_values(["log_id", "km_apart"]).drop_duplicates("log_id")[
        ["log_id", "incident_id", "minutes_apart", "km_apart"]]


def note_lags(pairs: pd.DataFrame, live_notes: pd.DataFrame, hist_notes: pd.DataFrame) -> pd.DataFrame:
    hist_notes = hist_notes.assign(text=hist_notes["detail_text"].map(normalize))
    live_notes = live_notes.assign(text=live_notes["detail_text"].map(normalize))
    live_notes, hist_notes = live_notes[live_notes["text"] != ""], hist_notes[hist_notes["text"] != ""]
    m = (live_notes.merge(pairs[["log_id", "incident_id"]], on="log_id")
         .merge(hist_notes[["incident_id", "text", "seen_at", "detail_time"]], on=["incident_id", "text"]))
    m["lag_min"] = (m["first_seen_at"] - m["seen_at"]).dt.total_seconds() / 60
    return m.sort_values("lag_min").drop_duplicates(["log_id", "detail_seq", "text"])


def main() -> None:
    import awswrangler as wr
    import boto3

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--profile", default="traffic")
    args = ap.parse_args()
    session = boto3.Session(profile_name=args.profile, region_name="us-west-2")

    def q(sql):
        return wr.athena.read_sql_query(sql, database="traffic_silver", workgroup="traffic",
                                        boto3_session=session, ctas_approach=False)

    window = f"between date '{args.start}' and date '{args.end}'"
    live = q(f"""select log_id, log_time, freeway, direction, lat, lon from stg_chp_incidents
                 where is_traffic and freeway is not null and lat is not null and incident_day {window}""")
    hist = q(f"""select incident_id, log_time, freeway, direction, lat, lon
                 from traffic_bronze.hist_chp_incidents where dt {window.replace("date ", "")}""")
    for df in (live, hist):
        df["log_time"] = pd.to_datetime(df["log_time"], utc=True)
        df["freeway"] = pd.to_numeric(df["freeway"], errors="coerce")
    pairs = match_incidents(live, hist)
    print(f"live freeway traffic incidents: {len(live):,}; with a PeMS record: {len(pairs):,} "
          f"({len(pairs) / max(len(live), 1):.0%})")
    print("PeMS log time minus live log time (minutes):",
          pairs["minutes_apart"].describe().round(2).to_dict())

    live_notes = q(f"""select n.log_id, n.detail_seq, n.first_seen_at, n.detail_text
                       from stg_chp_notes n join stg_chp_incidents i on i.log_id = n.log_id
                       where i.is_traffic and i.freeway is not null and i.incident_day {window}""")
    live_notes = live_notes[live_notes["log_id"].isin(pairs["log_id"])]
    hist_notes = q(f"""select d.incident_id, d.detail_time, d.seen_at, d.detail_text
                       from traffic_bronze.hist_chp_details d
                       join traffic_bronze.hist_chp_incidents i on i.incident_id = d.incident_id
                       where i.dt {window.replace("date ", "")} and i.freeway is not null""")
    hist_notes = hist_notes[hist_notes["incident_id"].isin(pairs["incident_id"])]
    for df, cols in ((live_notes, ["first_seen_at"]), (hist_notes, ["detail_time", "seen_at"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True)
    m = note_lags(pairs, live_notes, hist_notes)
    print(f"\nlive notes on matched incidents: {len(live_notes):,}; found in the history: {len(m):,} "
          f"({len(m) / max(len(live_notes), 1):.0%})")
    print("minutes from the history's time to when our poller first saw the note:")
    print("  percentiles:", {p: round(float(np.percentile(m['lag_min'], p)), 1) for p in (5, 25, 50, 75, 90, 95, 99)})
    for limit in (1, 2, 5, 10):
        print(f"  seen more than {limit} min after the history's time: {(m['lag_min'] > limit).mean():.1%}")
    out = ROOT / "ml" / "output"
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "history_vs_live_notes.csv", index=False)
    pairs.to_csv(out / "history_vs_live_incidents.csv", index=False)
    print(f"\nsaved matches to {out}")


if __name__ == "__main__":
    main()
