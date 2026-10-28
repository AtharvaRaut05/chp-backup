"""Save Caltrans' real-time travel-time files every 5 minutes.

Caltrans publishes the current travel time on selected freeway segments, refreshed every 5 minutes,
for districts 3, 8, 11 and 12 (https://cwwp2.dot.ca.gov/documentation/tt/tt.htm). PeMS gives this
project traffic a day late; these files are available within minutes, so with enough history they
can stand in for "traffic right before the crash" in a model that runs live.

Files are saved raw (gzipped, one per district per run) under raw/caltrans_tt/. They are parsed once
there's enough history to model with, so nothing about their layout is assumed here beyond JSON.
"""

import json
from datetime import datetime, timezone

import requests

from chp.fetch import USER_AGENT
from common.config import load
from common.lake import put_raw

DISTRICTS = (8, 11, 12, 3)
URL = "https://cwwp2.dot.ca.gov/data/d{d}/tt/ttStatusD{d:02d}.json"


def fetch(district: int, timeout: int = 20) -> bytes:
    resp = requests.get(URL.format(d=district), timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    json.loads(resp.content)  # reject an error page served with HTTP 200
    return resp.content


def main(event, context):
    settings = load()
    now = datetime.now(timezone.utc)
    saved, failed = [], {}
    for d in DISTRICTS:
        try:
            body = fetch(d)
        except (requests.RequestException, ValueError) as e:
            failed[d] = str(e)[:200]
            continue
        saved.append(put_raw(settings.bucket, "caltrans_tt", f"D{d:02d}_{now:%Y-%m-%dT%H-%M-%SZ}.json", body, now))
    summary = {"saved": len(saved), "failed": failed}
    print(json.dumps(summary))
    if not saved:  # every district failed: let the error alarm see it
        raise RuntimeError(f"no travel-time files saved: {failed}")
    return summary
