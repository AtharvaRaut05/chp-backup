"""New context sources: lane closures, live speed lookups, OpenStreetMap road layout, calendar seed."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import requests
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs"))

import ensure_tables  # noqa: E402
import lane_closures  # noqa: E402
import road_layout  # noqa: E402
from chp import handler  # noqa: E402
from chp.parse import parse_snapshot  # noqa: E402
from tomtom import speed_lookup  # noqa: E402

UTC = timezone.utc


# ---------- lane closures ----------

def lcs_record(index, lanes="2", started=None, ended=None, end_epoch="1791500000"):
    return {"lcs": {
        "index": index,
        "location": {"travelFlowDirection": "North",
                     "begin": {"beginRoute": "I-405", "beginLatitude": "34.0", "beginLongitude": "-118.4",
                               "beginMilepost": "40.1"},
                     "end": {"endLatitude": "34.02", "endLongitude": "-118.4", "endMilepost": ""}},
        "closure": {"closureID": "C1", "facility": "Freeway", "typeOfClosure": "Lane", "typeOfWork": "Paving",
                    "lanesClosed": lanes, "totalExistingLanes": "5",
                    "closureTimestamp": {"closureStartEpoch": "1791400000", "closureEndEpoch": end_epoch},
                    "code1097": {"isCode1097": "true" if started else "false",
                                 "code1097Timestamp": {"code1097Epoch": started or ""}},
                    "code1098": {"code1098Timestamp": {"code1098Epoch": ended or ""}},
                    "code1022": {"code1022Timestamp": {"code1022Epoch": ""}}}}}


def test_lane_closure_feed_is_flattened_with_times():
    body = json.dumps({"data": [lcs_record("a", started="1791410000000"), lcs_record("b", end_epoch="32503708740")]})
    df = lane_closures.parse_file(body.encode(), 7, pd.Timestamp("2026-10-08 12:00", tz="UTC"))
    a, b = df.set_index("closure_key").loc["a"], df.set_index("closure_key").loc["b"]
    assert a["route"] == "I-405" and a["begin_lat"] == 34.0 and pd.isna(a["end_abs_pm"]) and a["district"] == 7
    assert a["started_at"] == pd.Timestamp(1791410000, unit="s", tz="UTC")  # milliseconds read as such
    assert pd.isna(b["started_at"]) and pd.isna(b["planned_end"])          # never started; no end date


def test_lane_closure_day_keeps_first_and_last_sighting_and_latest_details():
    t = [pd.Timestamp(f"2026-10-08 {h}:00", tz="UTC") for h in (10, 11, 12)]
    snaps = pd.concat([
        lane_closures.parse_file(json.dumps({"data": [lcs_record("a", lanes="1")]}).encode(), 7, t[0]),
        lane_closures.parse_file(json.dumps({"data": [lcs_record("a", lanes="2", started="1791410000")]}).encode(), 7, t[1]),
        lane_closures.parse_file(json.dumps({"data": []}).encode(), 7, t[2]),
    ])
    out = lane_closures.summarize(snaps, datetime(2026, 10, 8).date())
    row = out.iloc[0]
    assert len(out) == 1 and row["first_seen"] == t[0] and row["last_seen"] == t[1] and row["n_polls"] == 2
    assert row["lanes_closed"] == "2" and row["started_at"] == pd.Timestamp(1791410000, unit="s", tz="UTC")
    assert list(out.columns) == [*lane_closures.COLUMNS, "dt"] and row["dt"] == "2026-10-08"


def test_lane_closure_days_to_parse_includes_unparsed_and_recent_days():
    class S3:
        def get_paginator(self, _):
            class P:
                def paginate(self, Bucket, Prefix, Delimiter):
                    days = ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"] if Prefix.startswith("raw") \
                        else ["2026-10-01", "2026-10-02", "2026-10-03"]
                    return [{"CommonPrefixes": [{"Prefix": f"{Prefix}dt={d}/"} for d in days]}]
            return P()

    days = lane_closures.days_to_parse(S3(), "b", datetime(2026, 10, 4).date())
    assert [str(d) for d in days] == ["2026-10-03", "2026-10-04"]


# ---------- live speed lookups ----------

def flow(heading_north=True):
    pts = [{"latitude": 34.0, "longitude": -118.0}, {"latitude": 34.01 if heading_north else 33.99, "longitude": -118.0}]
    return {"flowSegmentData": {"frc": "FRC0", "currentSpeed": 40, "freeFlowSpeed": 100, "currentTravelTime": 90,
                                "freeFlowTravelTime": 36, "confidence": 0.95, "roadClosure": False,
                                "coordinates": {"coordinate": pts}}}


def test_speed_lookup_keeps_speeds_and_segment_heading():
    now = datetime(2026, 10, 8, 12, tzinfo=UTC)
    row = speed_lookup.parse(flow(), {"log_id": "x", "lat": 34.0, "lon": -118.0}, now)
    assert row["current_speed_kmh"] == 40 and row["frc"] == "FRC0" and row["segment_bearing_deg"] == pytest.approx(0, abs=0.01)
    assert row["segment_km"] == pytest.approx(1.11, abs=0.01)
    assert speed_lookup.parse(flow(False), {"log_id": "y", "lat": 34.0, "lon": -118.0}, now)["segment_bearing_deg"] \
        == pytest.approx(180, abs=0.01)
    df = speed_lookup.frame([row])
    assert list(df.columns) == list(speed_lookup.COLUMNS) and df["dt"].iloc[0] == "2026-10-08"


def test_speed_lookup_table_matches_the_empty_table_the_job_creates():
    glue = {"string": "string", "datetime64[ns, UTC]": "timestamp", "float64": "double", "boolean": "boolean"}
    lambda_types = {c: glue[t] for c, t in speed_lookup.COLUMNS.items() if c != "dt"}
    assert lambda_types == ensure_tables.TOMTOM_COLUMNS


class FakeCounter:
    def __init__(self, cap):
        self.n, self.cap = 0, cap

    def update_item(self, **kw):
        if self.n >= self.cap:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        self.n += 1


@pytest.fixture
def lookup_env(monkeypatch):
    written = {}
    counter = FakeCounter(cap=2)
    monkeypatch.setattr(speed_lookup, "api_key", lambda: "k")
    monkeypatch.setattr(speed_lookup, "_table", lambda name: counter)
    monkeypatch.setattr(speed_lookup, "write_bronze", lambda df, *a: written.update(df=df))
    return written


def test_speed_lookups_stop_at_the_daily_cap(lookup_env, monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return flow()

    monkeypatch.setattr(speed_lookup.requests, "get", lambda *a, **k: Resp())
    incidents = [{"log_id": f"i{k}", "lat": 34.0, "lon": -118.0} for k in range(5)]
    out = speed_lookup.main({"incidents": incidents}, None)
    assert out["looked_up"] == 2 and out["over_daily_cap"] == 3 and len(lookup_env["df"]) == 2


def test_speed_lookups_all_failing_raise_for_the_alarm(lookup_env, monkeypatch):
    def get(*a, **k):
        raise requests.HTTPError("403 Forbidden")

    monkeypatch.setattr(speed_lookup.requests, "get", get)
    with pytest.raises(RuntimeError):
        speed_lookup.main({"incidents": [{"log_id": "a", "lat": 34.0, "lon": -118.0}]}, None)


def test_speed_lookup_without_a_key_does_nothing(monkeypatch):
    monkeypatch.setattr(speed_lookup, "api_key", lambda: None)
    assert speed_lookup.main({"incidents": [{"log_id": "a", "lat": 1, "lon": 2}]}, None)["status"] == "no key"


def test_poller_asks_for_lookups_only_for_new_freeway_traffic_incidents():
    snap = parse_snapshot((Path(__file__).parent / "fixtures" / "sa_sample.xml").read_bytes(),
                          datetime(2026, 10, 1, tzinfo=UTC), ("LACC", "OCCC", "INCC", "BCCC", "VTCC"))
    inc = snap.incidents
    picked = handler.new_freeway_incidents(snap, set(inc["log_id"]))
    ids = {p["log_id"] for p in picked}
    assert picked and all(p["lat"] and p["lon"] for p in picked)
    chosen = inc[inc["log_id"].isin(ids)]
    traffic = chosen["log_type_code"].isin(["1179", "1180", "1181", "1182", "1183", "20001", "20002", "1125", "1125A"])
    assert (traffic | (chosen["log_type"] == "SIG Alert")).all()
    text = (chosen["location"].fillna("") + " " + chosen["location_desc"].fillna("")).str.upper()
    assert text.str.contains(r"\b(?:I|SR|US)-?\d{1,3}\b|\b[NSEW]B \d{1,3}\b").all()
    assert len(ids) < inc["log_id"].nunique()  # surface-street and non-traffic incidents are skipped
    assert handler.new_freeway_incidents(snap, set()) == []  # nothing new, nothing looked up


def test_poller_never_fails_because_a_lookup_could_not_be_requested(monkeypatch):
    class Boom:
        def invoke(self, **kw):
            raise RuntimeError("throttled")

    monkeypatch.setenv("SPEED_LOOKUP_FUNCTION", "fn")
    monkeypatch.setattr(handler.boto3, "client", lambda name: Boom())
    assert handler.request_speed_lookups([{"log_id": "a", "lat": 1.0, "lon": 2.0}]) == 0


# ---------- road layout ----------

def _way(id, nodes, lats, lon, tags):
    return {"type": "way", "id": id, "nodes": nodes, "geometry": [{"lat": a, "lon": lon} for a in lats], "tags": tags}


@pytest.fixture
def network():
    """I-5 northbound: 4 lanes, dropping to 3 at 34.015; an on-ramp joins at 34.005 and an off-ramp
    leaves at 34.020. Southbound runs alongside, 4 lanes, no ramps."""
    lat = [34.0 + 0.005 * k for k in range(7)]
    return road_layout.Network([
        _way(1, [1, 2, 3, 4], lat[:4], -118.0, {"highway": "motorway", "ref": "I 5", "lanes": "4", "maxspeed": "65 mph"}),
        _way(2, [4, 5, 6, 7], lat[3:], -118.0, {"highway": "motorway", "ref": "I 5;CA 1", "lanes": "3"}),
        _way(3, list(range(17, 10, -1)), lat[::-1], -117.9996, {"highway": "motorway", "ref": "I 5", "lanes": "4"}),
        {"type": "way", "id": 9, "nodes": [5, 99], "tags": {"highway": "motorway_link"},
         "geometry": [{"lat": lat[4], "lon": -118.0}, {"lat": lat[4], "lon": -118.002}]},
        {"type": "way", "id": 8, "nodes": [98, 2], "tags": {"highway": "motorway_link"},
         "geometry": [{"lat": lat[1], "lon": -118.002}, {"lat": lat[1], "lon": -118.0}]},
    ])


def test_road_layout_northbound_sees_the_lane_drop_and_ramps(network):
    out = road_layout.layout(network, 34.011, -117.9999, 5, "N")
    assert out["road_heading_deg"] == 0 and out["road_lanes"] == 4 and out["maxspeed_mph"] == 65
    assert out["lane_drop_ahead"] == 1 and out["exits_ahead_1mi"] == 1 and out["merges_behind_1mi"] == 1
    assert out["dist_next_exit_mi"] == pytest.approx(0.62, abs=0.02)   # 34.011 -> 34.020
    assert out["dist_prev_merge_mi"] == pytest.approx(0.41, abs=0.02)  # 34.005 -> 34.011
    assert out["curve_deg"] == 0


def test_road_layout_southbound_takes_the_other_carriageway(network):
    out = road_layout.layout(network, 34.011, -117.9999, 5, "S")
    assert out["road_heading_deg"] == 180 and out["lane_drop_ahead"] == 0 and out["exits_ahead_1mi"] == 0
    assert out["dist_next_exit_mi"] == road_layout.CAP_MI


def test_road_layout_needs_the_same_freeway_nearby(network):
    assert road_layout.layout(network, 34.011, -117.9999, 405, "N")["match_m"] is None
    assert road_layout.layout(network, 34.2, -118.0, 5, "N")["match_m"] is None


def test_road_turn_is_the_smaller_angle():
    assert road_layout.turn(350, 10) == 20 and road_layout.turn(90, 270) == 180


# ---------- calendar seed ----------

def test_calendar_seed_marks_long_weekends_and_getaway_days():
    seed = pd.read_csv(Path(__file__).resolve().parents[1] / "transform" / "seeds" / "calendar_days.csv")
    kinds = seed.groupby("day")["kind"].apply(set)
    assert kinds["2026-11-26"] >= {"holiday", "long_weekend"}      # Thanksgiving
    assert kinds["2026-11-25"] == {"getaway_day"}                    # the Wednesday before
    assert kinds["2026-10-10"] == {"long_weekend"}                   # Saturday before Columbus Day
    assert kinds["2026-11-02"] == {"dst_week"}                       # Monday after the clocks fall back
    assert "2026-10-14" not in kinds                                 # an ordinary Wednesday
    assert set(seed["kind"]) == {"holiday", "long_weekend", "getaway_day", "winter_break", "dst_week", "special_day"}
    assert not seed.duplicated(["day", "kind"]).any()
