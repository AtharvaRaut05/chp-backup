"""Download the CHP incident feed."""

import os
import time
import xml.etree.ElementTree as ET

import requests

USER_AGENT = "chp-backup/0.1 (student traffic research; contact: atharvaraut05@ucla.edu)"
# Ask caches along the way for a fresh copy, not one they kept.
HEADERS = {"User-Agent": USER_AGENT, "Cache-Control": "no-cache", "Pragma": "no-cache"}
# Polls in a row that failed in this Lambda container before it is replaced (see get_feed).
RECYCLE_AFTER = 3
_failed_polls = 0


class FeedError(Exception):
    """The feed could not be downloaded or did not look like XML."""


def get_feed(url: str, timeout: int = 15, retries: int = 2) -> bytes:
    """Return the raw feed bytes, retrying on timeouts, server errors and cut-off downloads.

    The feed is a few hundred KB of XML. Sometimes the body ends mid-document even though the
    server answered 200; that is checked here (and retried) rather than failing later in parsing.

    Retries ask for an uncached copy (a throwaway query parameter plus no-cache headers). On
    Nov 2, 2026 one Lambda container got the same cut-off copy (exactly 172,032 bytes) on every
    attempt for about 12 hours while a fresh container got the full feed at once: a bad copy held
    somewhere between this container and CHP. So after RECYCLE_AFTER failed polls in a row, the
    container exits and Lambda starts the next poll in a new one.
    """
    global _failed_polls
    for attempt in range(retries + 1):
        try:
            params = {"_": str(time.time_ns() // 1_000_000)} if attempt else None
            resp = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
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
            _failed_polls = 0
            return body
        except (requests.Timeout, requests.ConnectionError, FeedError) as e:
            if attempt == retries:
                _failed_polls += 1
                if _failed_polls >= RECYCLE_AFTER:
                    print(f"[ERROR] FeedError: {_failed_polls} polls in a row failed in this container: {e}; "
                          "exiting so the next poll starts in a new one", flush=True)
                    os._exit(1)
                raise FeedError(f"failed after {retries + 1} attempts: {e}") from e
            time.sleep(2 ** (attempt + 1))  # 2s, then 4s; worst case ~51s, inside the 120s Lambda timeout
