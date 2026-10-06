from datetime import datetime, timedelta, timezone
from pathlib import Path

from chp.parse import parse_snapshot
from chp.state import (
    CLOSE_AFTER_MISSES,
    State,
    closures_frame,
    diff,
    incident_hashes,
    snapshot_hash,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sa_small.xml"
T0 = datetime(2026, 9, 30, 5, 0, tzinfo=timezone.utc)
SOCAL = ("LACC", "VTCC")


def snap_at(t, xml=None):
    return parse_snapshot(xml or FIXTURE.read_bytes(), t, SOCAL)


def test_hashes_ignore_poll_time():
    assert incident_hashes(snap_at(T0)) == incident_hashes(snap_at(T0 + timedelta(minutes=5)))


def test_hash_changes_when_a_note_is_added():
    xml = FIXTURE.read_text().replace(
        '<IncidentDetail>"[6] VEH IS AT AVALON [Shared]"</IncidentDetail>',
        '<IncidentDetail>"[6] VEH IS AT AVALON [Shared]"</IncidentDetail>'
        '</details><details><DetailTime>"Sep 26 2026  4:50PM"</DetailTime>'
        '<IncidentDetail>"[9] BLKG #2 LN"</IncidentDetail>',
    ).encode()
    before, after = incident_hashes(snap_at(T0)), incident_hashes(snap_at(T0, xml))
    assert before["260926LA2139"] != after["260926LA2139"]
    assert before["260926LA2056"] == after["260926LA2056"]
    assert snapshot_hash(before) != snapshot_hash(after)


def test_first_run_marks_everything_new():
    hashes = incident_hashes(snap_at(T0))
    incidents, changed, closures = diff(State(), hashes, T0)
    assert changed == set(hashes) and closures == []
    assert all(v["first_seen"] == T0.isoformat() for v in incidents.values())


def test_unchanged_incident_is_not_rewritten_but_last_seen_moves():
    hashes = incident_hashes(snap_at(T0))
    incidents, _, _ = diff(State(), hashes, T0)
    t1 = T0 + timedelta(minutes=1)
    incidents, changed, _ = diff(State(None, incidents), hashes, t1)
    assert changed == set()
    assert all(v["first_seen"] == T0.isoformat() and v["last_seen"] == t1.isoformat() for v in incidents.values())


def test_incident_closes_after_consecutive_misses():
    hashes = incident_hashes(snap_at(T0))
    incidents, _, _ = diff(State(), hashes, T0)
    remaining = {k: v for k, v in hashes.items() if k != "260926LA2139"}

    for i in range(1, CLOSE_AFTER_MISSES + 1):
        incidents, _, closures = diff(State(None, incidents), remaining, T0 + timedelta(minutes=i))
        if i < CLOSE_AFTER_MISSES:
            assert closures == [] and incidents["260926LA2139"]["missed"] == i

    assert [c["log_id"] for c in closures] == ["260926LA2139"]
    assert closures[0]["last_seen"] == T0.isoformat()
    assert "260926LA2139" not in incidents


def test_reappearing_incident_resets_its_miss_count():
    hashes = incident_hashes(snap_at(T0))
    incidents, _, _ = diff(State(), hashes, T0)
    gone = {k: v for k, v in hashes.items() if k != "260926LA2139"}
    incidents, _, _ = diff(State(None, incidents), gone, T0 + timedelta(minutes=1))
    incidents, changed, _ = diff(State(None, incidents), hashes, T0 + timedelta(minutes=2))
    assert incidents["260926LA2139"]["missed"] == 0
    assert "260926LA2139" not in changed  # same content as before, nothing new to write


def test_closures_frame_types():
    df = closures_frame([{"log_id": "x", "first_seen": T0.isoformat(), "last_seen": T0.isoformat(),
                          "closed_detected_at": T0.isoformat()}], T0)
    assert str(df["closed_detected_at"].dtype) == "datetime64[ns, UTC]"
    assert list(df["dt"]) == ["2026-09-30"]
    assert len(closures_frame([], T0)) == 0
