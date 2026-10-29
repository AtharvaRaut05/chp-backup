from datetime import datetime, timezone
from pathlib import Path

import pytest

from chp.parse import (
    _clean_note,
    _note_seq,
    _parse_chp_time,
    _parse_latlon,
    _split_log_type,
    parse_snapshot,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sa_small.xml"
POLLED_AT = datetime(2026, 9, 26, 23, 48, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def snap():
    return parse_snapshot(FIXTURE.read_bytes(), POLLED_AT, ("LACC", "VTCC", "GGCC"))


# --- helpers -------------------------------------------------------------

def test_latlon_converts_to_decimal_degrees():
    assert _parse_latlon("33873376:118266002") == (33.873376, -118.266002)


@pytest.mark.parametrize("value", [None, "", "0:0", "abc:def", "123"])
def test_latlon_missing_or_bad_returns_none(value):
    assert _parse_latlon(value) is None


@pytest.mark.parametrize("local, expected_utc", [
    ("Sep 26 2026  4:28PM", datetime(2026, 9, 26, 23, 28, tzinfo=timezone.utc)),  # PDT, UTC-7
    ("Dec  1 2026 12:05PM", datetime(2026, 12, 1, 20, 5, tzinfo=timezone.utc)),   # PST, UTC-8
    ("Nov  1 2026  1:30AM", datetime(2026, 11, 1, 8, 30, tzinfo=timezone.utc)),   # ambiguous -> PDT
])
def test_chp_time_converts_pacific_to_utc(local, expected_utc):
    assert _parse_chp_time(local) == expected_utc


def test_log_type_split():
    assert _split_log_type("1183-Trfc Collision-Unkn Inj") == ("1183", "Trfc Collision-Unkn Inj")
    assert _split_log_type("SIG Alert") == (None, "SIG Alert")


@pytest.mark.parametrize("raw, clean", [
    ("[6] VEH IS AT AVALON [Shared]", "VEH IS AT AVALON"),
    ("[1] [1] SOLO TC - BLOCKING CD/#1[Shared]", "SOLO TC - BLOCKING CD/#1"),
    ("[22] [Appended, 16:38:23] [3] RP ADVS BLK SD", "RP ADVS BLK SD"),
    ("[12] 98-50 HOV AND 1 BLOCKED OFF  / REQ SIG", "98-50 HOV AND 1 BLOCKED OFF / REQ SIG"),
    ("[30] [Notification] [CHP]-ABN HANDLE  [Shared]", "[Notification] [CHP]-ABN HANDLE"),
    ("[17] [Rotation Request Comment] 1039 Extreme Tow \n [Shared]", "[Rotation Request Comment] 1039 Extreme Tow"),
])
def test_clean_note(raw, clean):
    assert _clean_note(raw) == clean


def test_note_seq_is_first_index():
    assert _note_seq("[22] [Appended, 16:38:23] [3] TEXT") == 22
    assert _note_seq("NO INDEX") is None


# --- full snapshot ----------------------------------------------------------

def test_only_requested_dispatch_centers(snap):
    assert set(snap.incidents["dispatch_id"]) == {"LACC", "VTCC", "GGCC"}
    assert "260926SA0830" not in set(snap.incidents["log_id"])


def test_dispatch_is_not_the_same_as_center(snap):
    vt = snap.incidents.set_index("log_id").loc["260926VT0275"]
    assert (vt["center_id"], vt["dispatch_id"]) == ("GGHB", "VTCC")


def test_row_counts(snap):
    assert len(snap.incidents) == 5
    assert len(snap.details) == 10   # empty <details/> skipped
    assert len(snap.units) == 5


def test_zero_latlon_and_empty_fields_become_null(snap):
    silver = snap.incidents.set_index("log_id").loc["260925GG2816"]
    assert silver[["lat", "lon", "area"]].isna().all()


def test_duplicate_unit_events_are_kept(snap):
    # Two different units can be assigned at the same minute; both rows must survive.
    assigned = snap.units[(snap.units.log_id == "260926LA2139") & (snap.units.unit_text == "Unit Assigned")]
    assert len(assigned) == 2


def test_every_child_row_has_a_parent(snap):
    ids = set(snap.incidents["log_id"])
    assert set(snap.details["log_id"]) <= ids
    assert set(snap.units["log_id"]) <= ids


def test_snapshot_columns(snap):
    for df in (snap.incidents, snap.details, snap.units):
        assert (df["polled_at"] == POLLED_AT).all()
        assert (df["dt"] == "2026-09-26").all()


def test_types_are_stable(snap):
    assert str(snap.incidents["log_time"].dtype) == "datetime64[ns, UTC]"
    assert str(snap.details["detail_seq"].dtype) == "Int64"
    assert str(snap.incidents["lat"].dtype) == "float64"


def test_no_matching_incidents_still_has_columns():
    empty = parse_snapshot(FIXTURE.read_bytes(), POLLED_AT, ("NONE",))
    assert len(empty.incidents) == 0
    assert "log_time" in empty.incidents.columns


# --- feeds that must not be read as "no incidents" ---------------------------

@pytest.mark.parametrize("body", [
    b"<State></State>",
    b"<html><body><p>Service unavailable</p></body></html>",
    b'<?xml version="1.0"?><Error><Code>AccessDenied</Code></Error>',
])
def test_error_pages_are_rejected(body):
    from chp.parse import BadFeed
    with pytest.raises(BadFeed):
        parse_snapshot(body, POLLED_AT, ("LACC",))


@pytest.mark.parametrize("value, expected", [
    ("Sep 26 2026  4:28PM", datetime(2026, 9, 26, 23, 28, tzinfo=timezone.utc)),
    ("Sep 26 2026 4:28 PM", datetime(2026, 9, 26, 23, 28, tzinfo=timezone.utc)),
    ("Jan 5 2026 9:01AM", datetime(2026, 1, 5, 17, 1, tzinfo=timezone.utc)),  # PST
])
def test_time_formats(value, expected):
    assert _parse_chp_time(value) == expected


def test_unreadable_time_is_empty_not_a_crash():
    assert _parse_chp_time("sometime soon") is None


# ---------- download ----------

def test_cut_off_download_is_retried_and_then_accepted(monkeypatch):
    import requests

    from chp import fetch

    good = b"<State><Center/></State>"
    bodies = [good[:15], good]  # first response ends mid-document, second is complete

    class Resp:
        status_code = 200

        def __init__(self, body):
            self.content = body

        def raise_for_status(self):
            pass

    monkeypatch.setattr(fetch.requests, "get", lambda url, **kw: Resp(bodies.pop(0)))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    assert fetch.get_feed("http://x") == good


def test_feed_that_stays_cut_off_raises(monkeypatch):
    import pytest

    from chp import fetch

    class Resp:
        status_code = 200
        content = b"<State><Center>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(fetch.requests, "get", lambda url, **kw: Resp())
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with pytest.raises(fetch.FeedError, match="incomplete XML"):
        fetch.get_feed("http://x")
