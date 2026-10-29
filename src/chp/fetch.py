"""Download the CHP incident feed."""

import time
import xml.etree.ElementTree as ET

import requests

USER_AGENT = "chp-backup/0.1 (student traffic research; contact: atharvaraut05@ucla.edu)"


class FeedError(Exception):
    """The feed could not be downloaded or did not look like XML."""


def get_feed(url: str, timeout: int = 15, retries: int = 2) -> bytes:
    """Return the raw feed bytes, retrying on timeouts, server errors and cut-off downloads.

    The feed is a few MB of XML. Sometimes the connection drops part-way and the server still
    answered 200, so the body ends mid-document; that is checked here (and retried) rather than
    failing later in parsing.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code >= 500:
                raise FeedError(f"server error {resp.status_code}")
            resp.raise_for_status()
            body = resp.content
            if not body.lstrip().startswith(b"<"):
                raise FeedError("response is not XML")
            try:
                ET.fromstring(body)
            except ET.ParseError as e:
                raise FeedError(f"incomplete XML ({len(body):,} bytes): {e}") from e
            return body
        except (requests.Timeout, requests.ConnectionError, FeedError) as e:
            if attempt == retries:
                raise FeedError(f"failed after {retries + 1} attempts: {e}") from e
            time.sleep(2 ** (attempt + 1))  # 2s, then 4s; worst case ~51s, inside the 120s Lambda timeout
