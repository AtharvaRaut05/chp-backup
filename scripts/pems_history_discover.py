"""Step 1 of the history backfill: which CHP incident files does the PeMS Clearinghouse offer?

PeMS keeps CHP incidents (and their dispatcher notes) alongside the sensor data, but the
Clearinghouse's file-type names aren't documented. This tries the likely names for one district
and year, prints how many files each has, and downloads the newest file of each to show its
first lines, so the parser can be written against the real layout. Read-only; nothing is saved
to AWS. Samples go to ml/data/pems_samples/ (not committed).

    python scripts/pems_history_discover.py
    python scripts/pems_history_discover.py --district 7 --year 2025
"""

import argparse
import gzip
import io
import sys
import zipfile
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pems.client import PemsClient, PemsError  # noqa: E402

CANDIDATES = ["chp_incidents_day", "chp_incidents_month", "chp_incidents_detail_day",
              "chp_incidents_details_day", "chp_incident_details_day", "chp_incidents_detail_month"]
OUT = ROOT / "ml" / "data" / "pems_samples"


def head(path: Path, n: int = 6) -> list[str]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    elif raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            raw = z.read(z.namelist()[0])
    return raw.decode("utf-8", "replace").splitlines()[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic")
    ap.add_argument("--district", type=int, default=7)
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    ssm = boto3.Session(profile_name=args.profile, region_name="us-west-2").client("ssm")
    user, password = (ssm.get_parameter(Name=f"/traffic/pems/{n}", WithDecryption=True)["Parameter"]["Value"]
                      for n in ("username", "password"))
    client = PemsClient(user, password)
    client.login()
    OUT.mkdir(parents=True, exist_ok=True)
    for file_type in CANDIDATES:
        try:
            files = client.list_files(args.district, file_type, args.year)
        except (PemsError, ValueError) as e:
            print(f"{file_type}: error {str(e)[:150]}")
            continue
        print(f"\n{file_type}: {len(files)} files in {args.year} for district {args.district}")
        if not files:
            continue
        names = sorted(files)
        print("  first:", names[0], " last:", names[-1])
        sample = client.download(files[names[-1]], OUT / names[-1])
        print(f"  sample {names[-1]} ({sample.stat().st_size:,} bytes), first lines:")
        for line in head(sample):
            print("   ", line[:300])


if __name__ == "__main__":
    main()
