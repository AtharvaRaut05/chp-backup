import gzip
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chp import handler
from chp.state import State
from common import lake

FIXTURE = Path(__file__).parent / "fixtures" / "sa_small.xml"


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


@pytest.fixture
def env(monkeypatch):
    """Run the handler against fakes: fixture feed, in-memory S3, tables and state."""
    s3, written, saved = FakeS3(), {}, {"state": State()}
    monkeypatch.setattr(lake, "_s3", lambda: s3)
    monkeypatch.setattr(handler, "get_feed", lambda url: FIXTURE.read_bytes())
    monkeypatch.setattr(handler, "write_bronze", lambda df, b, db, table, p: written.update({table: df}))
    monkeypatch.setattr(handler, "load_state", lambda name: saved["state"])

    def save(name, state):
        saved["state"] = State(state.snapshot_hash, state.incidents, state.version + 1)

    monkeypatch.setattr(handler, "save_state", save)
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("DISPATCH_IDS", "LACC,VTCC")
    return s3, written, saved


def test_raw_key_is_partitioned_by_date():
    now = datetime(2026, 9, 30, 4, 5, 6, tzinfo=timezone.utc)
    assert lake.raw_key("chp", "x.xml", now) == "raw/chp/dt=2026-09-30/x.xml.gz"


def test_first_run_writes_everything(env):
    s3, written, saved = env
    result = handler.main({}, None)

    assert result["status"] == "changed"
    assert gzip.decompress(s3.puts[0]["Body"]) == FIXTURE.read_bytes()
    assert len(written["chp_incidents"]) == 4  # 3 LACC + 1 VTCC
    assert len(written["chp_closures"]) == 0
    assert saved["state"].version == 1


def test_identical_feed_writes_nothing(env):
    s3, written, saved = env
    handler.main({}, None)
    s3.puts.clear()
    written.clear()

    result = handler.main({}, None)

    assert result["status"] == "unchanged"
    assert s3.puts == [] and written == {}
    assert saved["state"].version == 1


def test_dispatch_ids_come_from_environment(env, monkeypatch):
    _, written, _ = env
    monkeypatch.setenv("DISPATCH_IDS", "VTCC")
    handler.main({}, None)
    assert set(written["chp_incidents"]["dispatch_id"]) == {"VTCC"}


def test_closure_completes_even_if_feed_stops_changing(env, monkeypatch):
    _, written, saved = env
    handler.main({}, None)
    xml = FIXTURE.read_text()
    start = xml.index('<Log ID = "260926LA2139">')
    smaller = (xml[:start] + xml[xml.index("</Log>", start) + 6:]).encode()
    monkeypatch.setattr(handler, "get_feed", lambda url: smaller)

    for _ in range(3):  # same smaller feed every poll
        written.clear()
        handler.main({}, None)

    assert list(written["chp_closures"]["log_id"]) == ["260926LA2139"]
    assert written["chp_incidents"].empty
    assert handler.main({}, None)["status"] == "unchanged"
