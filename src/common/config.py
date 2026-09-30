"""Project settings. Read from environment variables on AWS, defaults locally."""

import os
from dataclasses import dataclass

# CHP communication centers covering Southern California (PeMS districts 7, 8, 11, 12).
# Verify against your feed with scripts/list_dispatch.py.
SOCAL_DISPATCH = (
    "LACC",  # Los Angeles
    "VTCC",  # Ventura
    "OCCC",  # Orange County
    "INCC",  # Inland (San Bernardino / Riverside)
    "BCCC",  # Border (San Diego)
    "BSCC",  # Barstow
    "ICCC",  # Indio
    "ECCC",  # El Centro
)


@dataclass(frozen=True)
class Settings:
    feed_url: str
    bucket: str
    glue_db: str
    dispatch_ids: tuple[str, ...]


def load() -> Settings:
    return Settings(
        feed_url=os.environ.get("FEED_URL", "https://media.chp.ca.gov/sa_xml/sa.xml"),
        bucket=os.environ.get("BUCKET", "local-test"),
        glue_db=os.environ.get("GLUE_DB", "traffic_bronze"),
        dispatch_ids=SOCAL_DISPATCH,
    )
