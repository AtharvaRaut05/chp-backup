"""Print each Dispatch ID in a saved feed with example area names and incident counts.

Usage: python scripts/list_dispatch.py [path/to/feed.xml]
"""

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sa_sample.xml"
root = ET.parse(path).getroot()

areas = defaultdict(set)
counts = defaultdict(int)
for center in root.iter("Center"):
    for dispatch in center.iter("Dispatch"):
        key = (center.get("ID"), dispatch.get("ID"))
        for log in dispatch.iter("Log"):
            counts[key] += 1
            areas[key].add(log.findtext("Area", "").strip('"'))

for (center_id, dispatch_id), names in sorted(areas.items(), key=lambda kv: kv[0][1]):
    print(f"{dispatch_id}  (center {center_id}, {counts[(center_id, dispatch_id)]} incidents)  {sorted(names)[:6]}")
