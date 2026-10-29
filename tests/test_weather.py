import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs"))
import weather  # noqa: E402


def test_plan_fetches_only_missing_cell_days_in_batches(monkeypatch):
    monkeypatch.setattr(weather, "CELLS_PER_REQUEST", 2)
    needed = pd.DataFrame({"grid_lat": [34.0, 34.0, 34.25, 34.5, 34.0],
                           "grid_lon": [-118.0, -118.0, -118.25, -118.5, -118.0],
                           "day": ["2026-10-01", "2026-10-01", "2026-10-01", "2026-10-01", "2026-10-02"]})
    have = pd.DataFrame({"grid_lat": [34.0], "grid_lon": [-118.0], "day": ["2026-10-02"]})
    batches = weather.plan(needed, have)
    assert [(d, len(c)) for d, c in batches] == [(date(2026, 10, 1), 2), (date(2026, 10, 1), 1)]
    assert sorted(c for _, cells in batches for c in cells) == [(34.0, -118.0), (34.25, -118.25), (34.5, -118.5)]


def test_request_uses_archive_only_for_old_days():
    cells = [(34.0, -118.25), (33.75, -117.5)]
    recent = weather.request_url(date(2026, 10, 20), cells, today=date(2026, 10, 27))
    old = weather.request_url(date(2026, 6, 1), cells, today=date(2026, 10, 27))
    assert recent.startswith(weather.FORECAST_URL) and old.startswith(weather.ARCHIVE_URL)
    # the day before too, for the 24-hour lookback of incidents early in the day
    assert "latitude=34.00%2C33.75" in recent and "start_date=2026-10-19" in recent and "end_date=2026-10-21" in recent


def test_parse_keeps_hours_that_had_ended_and_their_end_time():
    t0 = 1791936000
    body = json.dumps([
        {"hourly": {"time": [t0, t0 + 3600, t0 + 7200], "precipitation": [0.0, 1.2, 9.9]}},
        {"hourly": {"time": [t0, t0 + 3600, t0 + 7200], "precipitation": [None, 0.4, 9.9]}},
    ]).encode()
    fetched = pd.Timestamp(t0 + 3600 + 600, unit="s", tz="UTC")  # 10 minutes into the third hour
    df = weather.parse(body, date(2026, 10, 1), [(34.0, -118.0), (33.75, -117.5)], fetched)
    assert len(df) == 4 and set(df["dt"]) == {"2026-10-01"}  # the unfinished hour (a forecast) is dropped
    assert 9.9 not in df["precipitation_mm"].tolist()
    first = df[(df["grid_lat"] == 34.0)].sort_values("ts")
    # Open-Meteo's value at time T is the total over the hour BEFORE T: T is already the end of the hour
    assert first["ts"].iloc[0] == pd.Timestamp(t0, unit="s", tz="UTC")
    assert first["precipitation_mm"].tolist() == [0.0, 1.2]
    assert df["precipitation_mm"].isna().sum() == 1 and (df["fetched_at"] == fetched).all()


def test_parse_handles_a_single_location_and_rejects_a_mismatch():
    one = json.dumps({"hourly": {"time": [1791936000], "precipitation": [0.5]}}).encode()
    now = pd.Timestamp("2026-12-01", tz="UTC")
    assert len(weather.parse(one, date(2026, 10, 1), [(34.0, -118.0)], now)) == 1
    try:
        weather.parse(one, date(2026, 10, 1), [(34.0, -118.0), (34.25, -118.0)], now)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a mismatch error")


def test_rate_limited_request_waits_and_retries(monkeypatch):
    import io
    import urllib.error

    calls, sleeps = [], []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        return Resp(b"ok")

    monkeypatch.setattr(weather.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(weather.time, "sleep", sleeps.append)
    assert weather.get("https://example.test/x") == b"ok"
    assert sleeps == [65]  # waited out the per-minute limit, then tried again
