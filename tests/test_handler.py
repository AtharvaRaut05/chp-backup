import gzip
from datetime import datetime, timezone
from pathlib import Path

from chp import handler
from common import lake

FIXTURE = Path(__file__).parent / "fixtures" / "sa_small.xml"


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


def test_raw_key_is_partitioned_by_date():
    now = datetime(2026, 9, 30, 4, 5, 6, tzinfo=timezone.utc)
    assert lake.raw_key("chp", "x.xml", now) == "raw/chp/dt=2026-09-30/x.xml.gz"


def test_handler_saves_raw_and_writes_tables(monkeypatch):
    feed = FIXTURE.read_bytes()
    s3, written = FakeS3(), {}
    monkeypatch.setattr(lake, "_s3", lambda: s3)
    monkeypatch.setattr(handler, "get_feed", lambda url: feed)
    monkeypatch.setattr(handler, "write_bronze", lambda df, bucket, db, table, parts: written.update({table: df}))
    monkeypatch.setenv("BUCKET", "test-bucket")

    result = handler.main({}, None)

    assert s3.puts[0]["Bucket"] == "test-bucket"
    assert s3.puts[0]["Key"].startswith("raw/chp/dt=")
    assert gzip.decompress(s3.puts[0]["Body"]) == feed
    assert set(written) == {"chp_incidents", "chp_details", "chp_units"}
    assert result["chp_incidents"] == len(written["chp_incidents"]) == 4  # 3 LACC + 1 VTCC
