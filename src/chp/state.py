"""Track open incidents between polls to find changed incidents and closures.

All state lives in one DynamoDB item: the hash of the last snapshot plus a map of open
incidents. Each run reads it once and writes it once, with a version check so two
overlapping runs can never overwrite each other's state.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache

import boto3
import pandas as pd

from chp.parse import Snapshot

CLOSE_AFTER_MISSES = 3  # consecutive changed snapshots an incident must be absent from
STATE_KEY = "chp_open_incidents"
_HASH_EXCLUDE = ["polled_at", "dt", "unit_seq"]  # change every poll or with XML order

CLOSURE_COLUMNS = {
    "log_id": "string",
    "first_seen": "datetime64[ns, UTC]",
    "last_seen": "datetime64[ns, UTC]",
    "closed_detected_at": "datetime64[ns, UTC]",
    "dt": "string",
}


@dataclass
class State:
    snapshot_hash: str | None = None
    incidents: dict = field(default_factory=dict)  # log_id -> {hash, first_seen, last_seen, missed}
    version: int = 0


def incident_hashes(snap: Snapshot) -> dict[str, str]:
    """One content hash per incident over its fields, notes and unit events (order-independent)."""
    rows = defaultdict(list)
    for name, df in (("i", snap.incidents), ("d", snap.details), ("u", snap.units)):
        content = df.drop(columns=[c for c in _HASH_EXCLUDE if c in df])
        for log_id, values in zip(df["log_id"], content.itertuples(index=False)):
            rows[log_id].append(name + "|" + "|".join(map(str, values)))
    return {log_id: _sha("\n".join(sorted(r))) for log_id, r in rows.items()}


def snapshot_hash(hashes: dict[str, str]) -> str:
    return _sha("\n".join(f"{k}:{v}" for k, v in sorted(hashes.items())))


def diff(state: State, hashes: dict[str, str], polled_at: datetime) -> tuple[dict, set[str], list[dict]]:
    """Return (new open-incident map, IDs of new or changed incidents, closure records)."""
    now = polled_at.isoformat()
    incidents, changed, closures = {}, set(), []

    for log_id, h in hashes.items():
        prev = state.incidents.get(log_id)
        if prev is None or prev["hash"] != h:
            changed.add(log_id)
        first_seen = prev["first_seen"] if prev else now
        incidents[log_id] = {"hash": h, "first_seen": first_seen, "last_seen": now, "missed": 0}

    for log_id, prev in state.incidents.items():
        if log_id in hashes:
            continue
        missed = int(prev["missed"]) + 1
        if missed >= CLOSE_AFTER_MISSES:
            closures.append({
                "log_id": log_id,
                "first_seen": prev["first_seen"],
                "last_seen": prev["last_seen"],
                "closed_detected_at": now,
            })
        else:
            incidents[log_id] = {**prev, "missed": missed}

    return incidents, changed, closures


def closures_frame(closures: list[dict], polled_at: datetime) -> pd.DataFrame:
    df = pd.DataFrame(closures, columns=[c for c in CLOSURE_COLUMNS if c != "dt"])
    for col in ("first_seen", "last_seen", "closed_detected_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["dt"] = polled_at.strftime("%Y-%m-%d")
    return df.astype(CLOSURE_COLUMNS)


def load_state(table_name: str) -> State:
    item = _table(table_name).get_item(Key={"pk": STATE_KEY}, ConsistentRead=True).get("Item")
    if not item:
        return State()
    return State(item.get("snapshot_hash"), item.get("incidents", {}), int(item["version"]))


def save_state(table_name: str, state: State) -> None:
    """Write the new state; fails if another run saved since this one loaded (version changed)."""
    _table(table_name).put_item(
        Item={
            "pk": STATE_KEY,
            "snapshot_hash": state.snapshot_hash,
            "incidents": state.incidents,
            "version": state.version + 1,
        },
        ConditionExpression="attribute_not_exists(pk) OR version = :v",
        ExpressionAttributeValues={":v": state.version},
    )


@cache
def _table(name: str):
    return boto3.resource("dynamodb").Table(name)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
