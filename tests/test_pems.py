from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from common.config import load
from pems import client as client_mod
from pems import orchestrate
from pems.client import PemsClient, PemsError
from pems.parse import file_date, parse_meta, parse_station_5min, station_file_name

FIXTURES = Path(__file__).parent / "fixtures"


# --- parsing (fixtures are real rows from the Clearinghouse) ----------------

def test_station_file_keeps_mainline_and_converts_to_utc():
    df = parse_station_5min(FIXTURES / "pems_station_5min_sample.txt.gz", 7, date(2026, 9, 29))
    assert list(df["station_id"]) == [715898]  # the two on-ramp (OR) rows are dropped
    row = df.iloc[0]
    assert row["ts"] == datetime(2026, 9, 29, 7, 0, tzinfo=timezone.utc)  # midnight PDT
    assert (row["total_flow"], row["avg_speed"], row["pct_observed"]) == (202, 70.4, 0)
    assert (row["district"], row["dt"]) == ("07", "2026-09-29")


def test_meta_file_columns():
    df = parse_meta(FIXTURES / "pems_meta_sample.txt", 7, date(2026, 8, 25))
    assert list(df["station_id"]) == [715898, 715900]
    assert df.iloc[0][["freeway", "direction", "lat", "lon", "lanes"]].tolist() == [5, "N", 33.880069, -118.021261, 3]
    assert str(df["state_pm"].dtype) == "string"


def test_file_names_and_dates():
    assert station_file_name(7, date(2026, 9, 29)) == "d07_text_station_5min_2026_09_29.txt.gz"
    assert file_date("d07_text_meta_2026_08_25.txt") == date(2026, 8, 25)


# --- client (fake HTTP session) --------------------------------------------

class FakeResponse:
    def __init__(self, text="", json_data=None, content_type="text/html"):
        self.text, self._json, self.headers = text, json_data, {"Content-Type": content_type}

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_login_checks_for_logout_link(monkeypatch):
    c = PemsClient("u", "p")
    monkeypatch.setattr(c._session, "post", lambda *a, **k: FakeResponse("Welcome, Atharva ... Logout"))
    c.login()
    monkeypatch.setattr(c._session, "post", lambda *a, **k: FakeResponse("Forgot your password?"))
    with pytest.raises(PemsError):
        c.login()


def test_list_files_flattens_months(monkeypatch):
    c = PemsClient("u", "p")
    listing = {"data": {"September": [{"file_name": "d07_text_station_5min_2026_09_29.txt.gz",
                                       "url": "/?download=530752&dnode=Clearinghouse"}]}}
    monkeypatch.setattr(c._session, "get", lambda *a, **k: FakeResponse(json_data=listing))
    assert c.list_files(7, "station_5min", 2026) == {
        "d07_text_station_5min_2026_09_29.txt.gz": "https://pems.dot.ca.gov/?download=530752&dnode=Clearinghouse"}
    monkeypatch.setattr(c._session, "get", lambda *a, **k: FakeResponse(json_data={"data": []}))
    assert c.list_files(7, "meta", 2026) == {}


# --- orchestration (fake pairing store and incident counts) ----------------

@pytest.fixture
def store(monkeypatch):
    statuses = {}
    monkeypatch.setattr(orchestrate.pairing, "get",
                        lambda s, d, day: {"status": statuses[(d, day)]} if (d, day) in statuses else None)
    monkeypatch.setattr(orchestrate.pairing, "put",
                        lambda s, d, day, status, **kw: statuses.__setitem__((d, day), status))
    monkeypatch.setenv("DISPATCH_IDS", "LACC,OCCC")  # districts 7 and 12
    monkeypatch.setenv("LOOKBACK_DAYS", "3")
    return statuses


def _today():
    return datetime.now(orchestrate.PACIFIC).date()


def test_plan_skips_days_without_incidents_and_days_already_done(store, monkeypatch):
    y = _today() - timedelta(days=1)
    store[(7, y - timedelta(days=1))] = "paired"
    monkeypatch.setattr(orchestrate, "incident_counts", lambda s, a, b: {(7, y): 12, (7, y - timedelta(days=2)): 3})

    result = orchestrate.plan({}, None)

    assert result["todo"] == [{"district": 7, "date": y.isoformat()},
                              {"district": 7, "date": (y - timedelta(days=2)).isoformat()}]
    assert store[(12, y)] == "no_incidents"  # Orange County had none -> no download
    assert result["skipped_no_incidents"] == 3  # all three Orange County days


def test_report_retries_recent_failures_and_gives_up_on_old_ones(store):
    recent, old = _today() - timedelta(days=1), _today() - timedelta(days=3)
    results = [
        {"district": 7, "date": recent.isoformat(), "status": "paired"},
        {"district": 12, "date": recent.isoformat(), "status": "not_available"},
        {"district": 7, "date": old.isoformat(), "status": "not_available"},
    ]
    assert orchestrate.report({"results": results}, None)["gave_up"] == 1
    assert store[(12, recent)] == "pending"
    assert store[(7, old)] == "unpaired"  # its CHP incidents get excluded from analysis


def test_report_fails_the_run_when_a_download_failed(store):
    results = [{"district": 7, "date": (_today() - timedelta(days=1)).isoformat(), "status": "failed",
                "error": {"Error": "PemsError", "Cause": "timeout"}}]
    with pytest.raises(RuntimeError, match="d07"):
        orchestrate.report({"results": results}, None)


def test_districts_follow_dispatch_ids(monkeypatch):
    monkeypatch.setenv("DISPATCH_IDS", "LACC,VTCC,GGCC,HMCC")
    assert load().pems_districts == [4, 7]  # HMCC is district 1, which PeMS doesn't cover
