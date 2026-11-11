"""Save Caltrans CWWP2 feeds (https://cwwp2.dot.ca.gov) raw, one gzipped file per district per run."""

import json
from datetime import datetime, timezone

import requests

from chp.fetch import USER_AGENT
from common.config import load
from common.lake import put_raw


def fetch(url: str, timeout: int = 20) -> bytes:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    json.loads(resp.content)  # reject an error page served with HTTP 200
    return resp.content


def save_districts(source: str, url: str, districts: tuple[int, ...]) -> dict:
    """Download url (formatted with d=district) for each district and save it under raw/<source>/.
    One failing district doesn't stop the others; if all fail, raise so the error alarm sees it."""
    settings = load()
    now = datetime.now(timezone.utc)
    saved, failed = [], {}
    for d in districts:
        try:
            body = fetch(url.format(d=d))
        except (requests.RequestException, ValueError) as e:
            failed[d] = str(e)[:200]
            continue
        saved.append(put_raw(settings.bucket, source, f"D{d:02d}_{now:%Y-%m-%dT%H-%M-%SZ}.json", body, now))
    summary = {"saved": len(saved), "failed": failed}
    print(json.dumps(summary))
    if not saved:
        raise RuntimeError(f"no {source} files saved: {failed}")
    return summary
