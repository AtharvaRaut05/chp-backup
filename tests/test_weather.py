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
    assert "latitude=34.00%2C33.75" in recent and "start_date=2026-10-20" in recent and "end_date=2026-10-21" in recent


def test_parse_stamps_each_value_at_the_end_of_its_hour():
    body = json.dumps([
        {"hourly": {"time": [1791936000, 1791939600], "precipitation": [0.0, 1.2]}},
        {"hourly": {"time": [1791936000, 1791939600], "precipitation": [None, 0.4]}},
    ]).encode()
    df = weather.parse(body, date(2026, 10, 1), [(34.0, -118.0), (33.75, -117.5)])
    assert len(df) == 4 and set(df["dt"]) == {"2026-10-01"}
    first = df[(df["grid_lat"] == 34.0)].sort_values("ts")
    assert first["ts"].iloc[0] == pd.Timestamp(1791936000 + 3600, unit="s", tz="UTC")
    assert first["precipitation_mm"].tolist() == [0.0, 1.2]
    assert df["precipitation_mm"].isna().sum() == 1


def test_parse_handles_a_single_location_and_rejects_a_mismatch():
    one = json.dumps({"hourly": {"time": [1791936000], "precipitation": [0.5]}}).encode()
    assert len(weather.parse(one, date(2026, 10, 1), [(34.0, -118.0)])) == 1
    try:
        weather.parse(one, date(2026, 10, 1), [(34.0, -118.0), (34.25, -118.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected a mismatch error")
