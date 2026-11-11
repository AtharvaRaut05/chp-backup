"""PeMS historical CHP incident files: parsing, and matching them to what the live poller saw."""

import gzip
import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("scripts", "jobs")]

from history_vs_live import match_incidents, normalize, note_lags  # noqa: E402
from pems.incidents import file_name, parse_day_zip  # noqa: E402

INCIDENTS = """\
23878270,INHB,261101,10/14/2026 08:03:00,1183-Trfc Collision-Unkn Inj,Sr57 S / Lambert Rd,Santa Ana,,,33.925201,-117.880287,12,59,53980,57,S,20.742,10.2,,12
23878265,LAHB,261101,10/14/2026 08:02:00,SPINOUT,I210 E / Yarnell St Ofr,Newhall,,,34.320345,-118.483526,7,37,44000,210,E,R0.5,.5,,2
"""
DETAILS = """\
23878270,110010918,10/14/2026 08:04:00,[5] [Appended  08:05:09] [1] WHI TESLA HOV
23878270,110011285,10/14/2026 08:15:00,Unit Enroute
23878270,110011300,10/14/2026 08:20:00,[6] 2 VEHS BLKG #1 LN
CONT ON NEXT LINE [Shared]
23878265,110011301,10/14/2026 23:59:00,[2] [Appended  00:01:30] TC BLKG
"""


def day_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("all_text_chp_incident_day_2026_10_14.txt.gz", gzip.compress(INCIDENTS.encode()))
        z.writestr("all_text_chp_incident_det_day_2026_10_14.txt.gz", gzip.compress(DETAILS.encode()))
    return buf.getvalue()


def test_history_files_parse_to_utc_with_freeway_and_type():
    inc, det = parse_day_zip(day_zip(), date(2026, 10, 14))
    a = inc.set_index("incident_id").loc["23878270"]
    assert a["log_time"] == pd.Timestamp("2026-10-14 15:03", tz="UTC")       # 08:03 PDT
    assert a["freeway"] == 57 and a["direction"] == "S" and a["district"] == 12 and a["log_type_code"] == "1183"
    assert pd.isna(inc.set_index("incident_id").loc["23878265", "log_type_code"])  # "SPINOUT" has no code
    assert (inc["dt"] == "2026-10-14").all() and file_name(date(2026, 10, 14)) == "all_text_chp_incidents_day_2026_10_14.txt.zip"


def test_history_notes_use_the_posting_time_and_join_broken_lines():
    _, det = parse_day_zip(day_zip(), date(2026, 10, 14))
    assert len(det) == 4
    first = det.iloc[0]
    assert first["detail_time"] == pd.Timestamp("2026-10-14 15:04", tz="UTC")
    assert first["seen_at"] == pd.Timestamp("2026-10-14 15:05:09", tz="UTC")  # appended a minute later
    assert det.iloc[1]["seen_at"] == det.iloc[1]["detail_time"]                # no marker: the stamp
    assert det.iloc[2]["detail_text"].endswith("CONT ON NEXT LINE [Shared]")
    assert det.iloc[3]["seen_at"] == pd.Timestamp("2026-10-15 07:01:30", tz="UTC")  # posted after midnight


def test_note_text_matches_across_the_two_sources():
    assert normalize("[5] [Appended  08:05:09] [1] WHI TESLA, HOV") == normalize("WHI TESLA HOV") == "whi tesla hov"
    assert normalize("2 VEHS BLKG #1 LN [Shared]") == "2 vehs blkg 1 ln"


def test_live_incidents_match_their_history_record_and_note_lags():
    t = pd.Timestamp("2026-10-14 15:03", tz="UTC")
    live = pd.DataFrame({"log_id": ["L1", "L2"], "log_time": [t, t], "freeway": [57, 57], "direction": ["S", "N"],
                         "lat": [33.9252, 33.9252], "lon": [-117.8803, -117.8803]})
    hist = pd.DataFrame({"incident_id": ["H1", "H2"], "log_time": [t + pd.Timedelta(minutes=1), t],
                         "freeway": [57, 57], "direction": ["S", "S"], "lat": [33.9253, 34.5], "lon": [-117.8802, -117.88]})
    pairs = match_incidents(live, hist)
    assert pairs.to_dict("records")[0] | {"km_apart": 0} == {"log_id": "L1", "incident_id": "H1", "minutes_apart": 1.0, "km_apart": 0}
    assert len(pairs) == 1  # L2 is northbound: no record
    live_notes = pd.DataFrame({"log_id": ["L1"], "detail_seq": [1], "detail_text": ["WHI TESLA HOV"],
                               "first_seen_at": [t + pd.Timedelta(minutes=4)]})
    hist_notes = pd.DataFrame({"incident_id": ["H1"], "detail_text": ["[1] WHI TESLA HOV"],
                               "detail_time": [t + pd.Timedelta(minutes=1)], "seen_at": [t + pd.Timedelta(minutes=2)]})
    assert note_lags(pairs, live_notes, hist_notes)["lag_min"].tolist() == [2.0]


def test_sensor_backfill_lists_only_missing_district_days_in_batches():
    from backfill_pems import _PART, batches, todo

    have = {(7, "2026-03-02")}
    items = todo(date(2026, 3, 1), date(2026, 3, 2), have, districts=(7, 8))
    assert items == [{"district": 7, "date": "2026-03-01"}, {"district": 8, "date": "2026-03-01"},
                     {"district": 8, "date": "2026-03-02"}]
    assert [len(b) for b in batches(list(range(800)))] == [350, 350, 100]
    m = _PART.search("bronze/pems_station_5min/district=7/dt=2026-03-02/abc.snappy.parquet")
    assert (int(m.group(1)), m.group(2)) == (7, "2026-03-02")


def test_history_tables_created_empty_match_what_the_backfill_writes():
    import ensure_tables
    from pems.incidents import DETAIL_TYPES, INCIDENT_TYPES

    assert ensure_tables.HIST_INCIDENT_COLUMNS == INCIDENT_TYPES
    assert ensure_tables.HIST_DETAIL_COLUMNS == DETAIL_TYPES
