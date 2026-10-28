import json

import pytest
import requests

from caltrans import travel_times
from common import lake


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


class Resp:
    def __init__(self, body, status=200):
        self.content, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(lake, "_s3", lambda: fake)
    monkeypatch.setenv("BUCKET", "test-bucket")
    return fake


def test_saves_one_raw_file_per_district(s3, monkeypatch):
    urls = []

    def get(url, **kw):
        urls.append(url)
        return Resp(json.dumps({"data": [{"tt": 1}]}).encode())

    monkeypatch.setattr(travel_times.requests, "get", get)
    out = travel_times.main({}, None)
    assert out == {"saved": 4, "failed": {}}
    assert "https://cwwp2.dot.ca.gov/data/d8/tt/ttStatusD08.json" in urls
    keys = [p["Key"] for p in s3.puts]
    assert all(k.startswith("raw/caltrans_tt/dt=") and k.endswith(".json.gz") for k in keys)
    assert {k.split("/")[-1][:3] for k in keys} == {"D08", "D11", "D12", "D03"}


def test_one_bad_district_does_not_stop_the_others(s3, monkeypatch):
    def get(url, **kw):
        if "d11" in url:
            return Resp(b"<html>maintenance</html>")  # error page with HTTP 200
        if "d3/" in url:
            return Resp(b"", status=503)
        return Resp(b"{}")

    monkeypatch.setattr(travel_times.requests, "get", get)
    out = travel_times.main({}, None)
    assert out["saved"] == 2 and set(out["failed"]) == {11, 3}


def test_all_districts_failing_raises_for_the_alarm(s3, monkeypatch):
    def get(url, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(travel_times.requests, "get", get)
    with pytest.raises(RuntimeError):
        travel_times.main({}, None)
