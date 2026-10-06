"""Talk to the PeMS Clearinghouse the same way its web page does.

PeMS has no official API. These requests were confirmed against the live site:
  login:  POST https://pems.dot.ca.gov/  (form fields: username, password, login)
  list:   GET  /?srq=clearinghouse&district_id=7&geotag=null&yy=2026&type=station_5min&returnformat=text
          -> {"data": {"September": [{"file_name": ..., "file_id": ..., "url": "/?download=...", "bytes": ...}]}}
  file:   GET  the listed url (gzip for station_5min, tab-separated text for meta)
"""

import time
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE = "https://pems.dot.ca.gov/"
USER_AGENT = "chp-backup/0.1 (student traffic research)"


class PemsError(Exception):
    pass


class PemsClient:
    def __init__(self, username: str, password: str, timeout: int = 120):
        self._creds = {"username": username, "password": password}
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    def login(self) -> None:
        resp = self._session.post(
            BASE, data={"redirect": "", **self._creds, "login": "Login"}, timeout=self._timeout
        )
        resp.raise_for_status()
        if "Logout" not in resp.text:
            raise PemsError("PeMS login failed: check the username and password in Parameter Store")

    def list_files(self, district: int, file_type: str, year: int) -> dict[str, str]:
        """Return {file_name: download_url} for one district, file type and year."""
        resp = self._session.get(BASE, timeout=self._timeout, params={
            "srq": "clearinghouse", "district_id": district, "geotag": "null",
            "yy": year, "type": file_type, "returnformat": "text",
        })
        resp.raise_for_status()
        data = resp.json().get("data") or {}  # PeMS returns [] when nothing exists
        months = data.values() if isinstance(data, dict) else []
        return {f["file_name"]: urljoin(BASE, f["url"]) for files in months for f in files}

    def download(self, url: str, dest: Path) -> Path:
        with self._session.get(url, stream=True, timeout=self._timeout) as resp:
            resp.raise_for_status()
            if "text/html" in resp.headers.get("Content-Type", ""):
                raise PemsError(f"got a web page instead of a file from {url} (session expired?)")
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        time.sleep(2)  # be polite between downloads
        return dest
