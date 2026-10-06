"""Project settings. Read from environment variables on AWS, defaults locally."""

import os
from dataclasses import dataclass

# CHP communication centers covering Southern California (PeMS districts 7, 8, 11, 12).
# To expand coverage, set DISPATCH_IDS (comma-separated) in template.yaml instead of editing this.
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

# Caltrans district each CHP dispatch center mostly covers. Approximate: a few centers
# cross district lines, which only affects the "did this district have incidents" check.
# PeMS has no detector data for districts 1, 2 and 9.
DISPATCH_DISTRICT = {
    "LACC": 7, "VTCC": 7, "OCCC": 12, "INCC": 8, "BSCC": 8, "ICCC": 8, "BCCC": 11, "ECCC": 11,
    "GGCC": 4, "MYCC": 5, "SLCC": 5, "SACC": 3, "CHCC": 3, "TKCC": 3, "STCC": 10, "MRCC": 10,
    "FRCC": 6, "BFCC": 6, "HMCC": 1, "UKCC": 1, "RDCC": 2, "SUCC": 2, "YKCC": 2,
}
PEMS_DISTRICTS_AVAILABLE = {3, 4, 5, 6, 7, 8, 10, 11, 12}

# Incident types that can affect freeway traffic (collisions, hazards, SigAlerts).
TRAFFIC_TYPE_CODES = ("1179", "1180", "1181", "1182", "1183", "20001", "20002", "1125", "1125A")


@dataclass(frozen=True)
class Settings:
    feed_url: str
    bucket: str
    glue_db: str
    state_table: str
    pairing_table: str
    athena_workgroup: str
    dispatch_ids: tuple[str, ...]
    lookback_days: int

    @property
    def pems_districts(self) -> list[int]:
        found = {DISPATCH_DISTRICT.get(d) for d in self.dispatch_ids}
        return sorted(found & PEMS_DISTRICTS_AVAILABLE)


def load() -> Settings:
    dispatch = os.environ.get("DISPATCH_IDS")
    return Settings(
        feed_url=os.environ.get("FEED_URL", "https://media.chp.ca.gov/sa_xml/sa.xml"),
        bucket=os.environ.get("BUCKET", "local-test"),
        glue_db=os.environ.get("GLUE_DB", "traffic_bronze"),
        state_table=os.environ.get("STATE_TABLE", "local-test"),
        pairing_table=os.environ.get("PAIRING_TABLE", "local-test"),
        athena_workgroup=os.environ.get("ATHENA_WORKGROUP", "traffic"),
        dispatch_ids=tuple(d.strip() for d in dispatch.split(",")) if dispatch else SOCAL_DISPATCH,
        lookback_days=int(os.environ.get("LOOKBACK_DAYS", "7")),
    )
