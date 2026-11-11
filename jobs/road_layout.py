"""Road layout at each incident from OpenStreetMap -> bronze table road_layout.

The same crash backs up more where the road is already squeezed: a lane drop just ahead, on-ramps
pouring in just behind, a curve. PeMS sensor metadata gives lane counts at sensors; OpenStreetMap
gives the whole road. This step downloads every freeway and ramp in Southern California once (saved
under raw/osm/, refreshed after 90 days), places each new incident on its carriageway, walks the road
a mile each way and writes one row per incident:

  road_heading_deg      compass heading of travel at the crash (for sun glare)
  road_lanes            lanes at the crash
  lane_drop_ahead       lanes lost within 1 mile downstream (0 = none)
  exits_ahead_1mi       off-ramps and splits within 1 mile downstream
  merges_behind_1mi     on-ramps and joining freeways within 1 mile upstream
  dist_next_exit_mi     miles to the next off-ramp (capped at 3)
  dist_prev_merge_mi    miles back to the last on-ramp (capped at 3)
  curve_deg             how much the road turns over the 400 m before and after the crash
  maxspeed_mph, on_bridge, match_m (metres from the reported location to the road)

Only incidents not already done are processed; ones that don't match a freeway get a row of nulls so
they aren't retried. Data (c) OpenStreetMap contributors, ODbL. Runs daily before dbt.

    python jobs/road_layout.py --profile traffic
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from lake_job import ensure_table, query, session_and_bucket, write

TABLE = "road_layout"
OVERPASS = "https://overpass-api.de/api/interpreter"
TILES = [(lat, lon) for lat in range(32, 36) for lon in range(-121, -114)]  # 1-degree tiles over districts 7, 8, 11, 12
REFRESH_DAYS = 90
MILE = 1609.34
MATCH_M = 400          # the reported location must be this close to the freeway
STEP_M = 50            # spacing of the points used to find the nearest carriageway
CELL = 0.005           # grid cell (degrees, ~500 m) for the nearest-point search
CAP_MI = 3.0
COLUMNS = {"log_id": "string", "match_m": "double", "road_heading_deg": "double", "road_lanes": "double",
           "lane_drop_ahead": "double", "exits_ahead_1mi": "double", "merges_behind_1mi": "double",
           "dist_next_exit_mi": "double", "dist_prev_merge_mi": "double", "curve_deg": "double",
           "maxspeed_mph": "double", "on_bridge": "boolean", "osm_downloaded": "string"}
FEATURES = [c for c in COLUMNS if c not in ("log_id", "osm_downloaded")]
TARGET = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}


# ---------- geometry ----------

def metres(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371000.0 * 2 * math.asin(math.sqrt(min(1.0, a)))


def bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360


def turn(a: float, b: float) -> float:
    """Smallest angle between two headings (0-180)."""
    return abs((a - b + 180) % 360 - 180)


# ---------- the road network ----------

def _number(value: str | None) -> float | None:
    m = re.search(r"\d+(\.\d+)?", value or "")
    return float(m.group()) if m else None


def _refs(tags: dict) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", tags.get("ref", ""))}


@dataclass
class Way:
    id: int
    nodes: list[int]
    pts: list[tuple[float, float]]
    refs: set[int]
    lanes: float | None
    maxspeed: float | None
    bridge: bool
    cum: list[float] = field(default_factory=list)  # metres from the way's first node to each node

    def __post_init__(self):
        self.cum = [0.0]
        for a, b in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + metres(*a, *b))

    @property
    def length(self) -> float:
        return self.cum[-1]


class Network:
    """Freeway carriageways (one-way, in travel order) plus where ramps leave and join them."""

    def __init__(self, elements: list[dict]):
        self.ways: dict[int, Way] = {}
        self.diverge, self.merge = set(), set()
        self.starts, self.ends = defaultdict(list), defaultdict(list)
        for e in elements:
            if e.get("type") != "way" or "geometry" not in e:
                continue
            tags = e.get("tags", {})
            nodes, pts = e["nodes"], [(g["lat"], g["lon"]) for g in e["geometry"]]
            if len(nodes) != len(pts) or len(nodes) < 2:
                continue
            if tags.get("highway") == "motorway_link":
                self.diverge.add(nodes[0])
                self.merge.add(nodes[-1])
                continue
            if tags.get("oneway") == "-1":
                nodes, pts = nodes[::-1], pts[::-1]
            self._add(Way(e["id"], nodes, pts, _refs(tags), _number(tags.get("lanes")),
                          _number(tags.get("maxspeed")), tags.get("bridge", "no") != "no"))
            if tags.get("oneway") == "no":  # rare two-way freeway: travel both ways
                self._add(Way(-e["id"], nodes[::-1], pts[::-1], _refs(tags), _number(tags.get("lanes")),
                              _number(tags.get("maxspeed")), tags.get("bridge", "no") != "no"))
        # freeway-to-freeway splits and joins tagged as motorway, not motorway_link
        self.diverge |= {n for n, ws in self.starts.items() if len(ws) > 1}
        self.merge |= {n for n, ws in self.ends.items() if len(ws) > 1}
        self.grid = defaultdict(list)
        for w in self.ways.values():
            for i in range(len(w.pts) - 1):
                (la1, lo1), (la2, lo2) = w.pts[i], w.pts[i + 1]
                seg = w.cum[i + 1] - w.cum[i]
                for k in range(max(1, math.ceil(seg / STEP_M)) + 1):
                    f = min(1.0, k * STEP_M / seg) if seg else 0.0
                    lat, lon = la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1)
                    self.grid[(int(lat // CELL), int(lon // CELL))].append(
                        (w.id, w.cum[i] + f * seg, lat, lon, bearing(la1, lo1, la2, lo2)))

    def _add(self, w: Way) -> None:
        self.ways[w.id] = w
        self.starts[w.nodes[0]].append(w.id)
        self.ends[w.nodes[-1]].append(w.id)

    def match(self, lat: float, lon: float, freeway: int, direction: str):
        """(way id, metres along it, distance away, heading) of the point on the incident's freeway
        heading its way: of the carriageways within MATCH_M, the one pointing closest to the direction."""
        ci, cj = int(lat // CELL), int(lon // CELL)
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for wid, along, plat, plon, head in self.grid.get((ci + di, cj + dj), []):
                    if freeway not in self.ways[wid].refs:
                        continue
                    d = metres(lat, lon, plat, plon)
                    if d > MATCH_M:
                        continue
                    score = (round(math.cos(math.radians(head - TARGET[direction])), 1), -d)
                    if best is None or score > best[0]:
                        best = (score, wid, along, d, head)
        return None if best is None else best[1:]

    def walk(self, wid: int, along: float, freeway: int, ahead: bool, limit: float) -> list[tuple]:
        """Nodes passed going `limit` metres downstream (ahead) or upstream, as (metres, node, lat, lon,
        lanes of the road leading there). At a split, stay on the way that carries the same freeway."""
        out, seen, dist = [], set(), 0.0
        w, pos = self.ways[wid], along
        while w.id not in seen:
            seen.add(w.id)
            idx = range(len(w.nodes)) if ahead else range(len(w.nodes) - 1, -1, -1)
            for i in idx:
                d = dist + (w.cum[i] - pos if ahead else pos - w.cum[i])
                if d <= 0 or (out and out[-1][1] == w.nodes[i]):  # behind the start, or the joint between ways
                    continue
                if d > limit:
                    return out
                out.append((d, w.nodes[i], *w.pts[i], w.lanes))
            dist += (w.length - pos) if ahead else pos
            nxt = self.starts.get(w.nodes[-1], []) if ahead else self.ends.get(w.nodes[0], [])
            same = [n for n in nxt if freeway in self.ways[n].refs] or (nxt if len(nxt) == 1 else [])
            if not same:
                return out
            w = self.ways[same[0]]
            pos = 0.0 if ahead else w.length
        return out


def point_at(w: Way, along: float) -> tuple[float, float]:
    for i in range(len(w.cum) - 1):
        if w.cum[i] <= along <= w.cum[i + 1]:
            f = (along - w.cum[i]) / ((w.cum[i + 1] - w.cum[i]) or 1)
            return (w.pts[i][0] + f * (w.pts[i + 1][0] - w.pts[i][0]), w.pts[i][1] + f * (w.pts[i + 1][1] - w.pts[i][1]))
    return w.pts[-1]


def layout(net: Network, lat: float, lon: float, freeway: int, direction: str) -> dict:
    m = net.match(lat, lon, freeway, direction) if direction in TARGET else None
    if m is None:
        return {c: None for c in FEATURES}
    wid, along, dist, heading = m
    w = net.ways[wid]
    ahead = net.walk(wid, along, freeway, True, CAP_MI * MILE)
    behind = net.walk(wid, along, freeway, False, CAP_MI * MILE)
    exits = [d for d, node, *_ in ahead if node in net.diverge]
    merges = [d for d, node, *_ in behind if node in net.merge]
    lanes_ahead = [ln for d, *_, ln in ahead if d <= MILE and ln is not None]
    here = point_at(w, along)

    def heading_to(nodes, reverse):
        far = [(la, lo) for d, _, la, lo, _ in nodes if d >= 400]
        if not far:
            return None
        return bearing(*far[0], *here) if reverse else bearing(*here, *far[0])

    h_in, h_out = heading_to(behind, True), heading_to(ahead, False)
    return {
        "match_m": round(dist, 1),
        "road_heading_deg": round(heading, 1),
        "road_lanes": w.lanes,
        "lane_drop_ahead": None if w.lanes is None else max(0.0, w.lanes - min(lanes_ahead, default=w.lanes)),
        "exits_ahead_1mi": float(sum(d <= MILE for d in exits)),
        "merges_behind_1mi": float(sum(d <= MILE for d in merges)),
        "dist_next_exit_mi": round(min(exits, default=CAP_MI * MILE) / MILE, 2),
        "dist_prev_merge_mi": round(min(merges, default=CAP_MI * MILE) / MILE, 2),
        "curve_deg": None if h_in is None or h_out is None else round(turn(h_in, h_out), 1),
        "maxspeed_mph": w.maxspeed,
        "on_bridge": w.bridge,
    }


# ---------- download ----------

def overpass_query(lat: int, lon: int) -> str:
    bbox = f"{lat},{lon},{lat + 1},{lon + 1}"
    return f'[out:json][timeout:300];way["highway"~"^(motorway|motorway_link)$"]({bbox});out body geom;'


def fetch_tile(lat: int, lon: int, retries: int = 4) -> list[dict]:
    data = urllib.parse.urlencode({"data": overpass_query(lat, lon)}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(OVERPASS, data=data,
                                         headers={"User-Agent": "chp-backup/0.1 (student traffic research)"})
            with urllib.request.urlopen(req, timeout=360) as resp:
                return json.loads(resp.read())["elements"]
        except (OSError, ValueError) as e:  # busy server (429/504), dropped connection, cut-off JSON
            if attempt == retries:
                raise
            busy = isinstance(e, urllib.error.HTTPError) and e.code in (429, 504)
            time.sleep(60 if busy else 10 * (attempt + 1))


def download() -> list[dict]:
    ways = {}
    for i, (lat, lon) in enumerate(TILES):
        if i:
            time.sleep(5)  # Overpass is a shared free service
        for e in fetch_tile(lat, lon):
            ways[e["id"]] = e  # ways crossing tile edges come back from both tiles
    return list(ways.values())


def load_network(s3, bucket: str, now: datetime) -> tuple[Network, str]:
    """The newest saved download if it's recent enough, else a fresh one (saved for next time)."""
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="raw/osm/")
    keys = sorted(o["Key"] for p in pages for o in p.get("Contents", []))
    if keys:
        stamp = re.search(r"motorways_(\d{4}-\d{2}-\d{2})", keys[-1]).group(1)
        if (now.date() - datetime.fromisoformat(stamp).date()).days <= REFRESH_DAYS:
            body = gzip.decompress(s3.get_object(Bucket=bucket, Key=keys[-1])["Body"].read())
            return Network(json.loads(body)), stamp
    elements, stamp = download(), now.strftime("%Y-%m-%d")
    s3.put_object(Bucket=bucket, Key=f"raw/osm/motorways_{stamp}.json.gz", Body=gzip.compress(json.dumps(elements).encode()))
    return Network(elements), stamp


# ---------- AWS ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="")
    args = ap.parse_args()
    session, bucket = session_and_bucket(args.profile)
    ensure_table(session, bucket, TABLE, COLUMNS)
    todo = query(session, f"""
        select i.log_id, i.lat, i.lon, i.freeway, i.direction
        from traffic_silver.stg_chp_incidents i
        left join traffic_bronze.{TABLE} r on r.log_id = i.log_id
        where i.is_traffic and i.freeway is not null and i.direction is not null and i.lat is not null
          and r.log_id is null""")
    if todo.empty:
        print("road layout: no new incidents")
        return
    now = datetime.now(timezone.utc)
    net, stamp = load_network(session.client("s3"), bucket, now)
    rows = [{"log_id": r.log_id, **layout(net, float(r.lat), float(r.lon), int(r.freeway), r.direction)}
            for r in todo.itertuples()]
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "osm_downloaded"])
    df["osm_downloaded"], df["dt"] = stamp, now.strftime("%Y-%m-%d")
    write(df, session, bucket, TABLE, COLUMNS)
    print(f"road layout: {len(df):,} incidents, {df['match_m'].notna().sum():,} placed on a freeway "
          f"(OSM download {stamp}, {len(net.ways):,} carriageway pieces)")


if __name__ == "__main__":
    main()
