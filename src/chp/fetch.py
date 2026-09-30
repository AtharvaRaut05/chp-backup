"""Download the CHP incident feed."""

import time

import requests

USER_AGENT = "chp-backup/0.1 (student traffic research; contact: atharvaraut05@ucla.edu)"


class FeedError(Exception):
    """The feed could not be downloaded or did not look like XML."""


def get_feed(url: str, timeout: int = 20, retries: int = 2) -> bytes:
    """Return the raw feed bytes, retrying on timeouts and server errors."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code >= 500:
                raise FeedError(f"server error {resp.status_code}")
            resp.raise_for_status()
            body = resp.content
            if not body.lstrip().startswith(b"<"):
                raise FeedError("response is not XML")
            return body
        except (requests.Timeout, requests.ConnectionError, FeedError) as e:
            if attempt == retries:
                raise FeedError(f"failed after {retries + 1} attempts: {e}") from e
            time.sleep(2 ** (attempt + 1))  # wait 2s, then 4s