"""Upload the code the scheduled job runs (transform/, ml/, jobs/) to S3.

Run this once after deploying, and again whenever you change the dbt models or ML code:

    python scripts/upload_job_code.py
"""

import argparse
import io
import zipfile
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ["transform", "ml", "jobs"]
SKIP_DIRS = {"target", "target_vars", "logs", "data", "output", "__pycache__", ".pytest_cache", "dbt_packages"}
SKIP_FILES = {".user.yml"}
KEY = "jobs/code.zip"


def build_zip(root: Path = ROOT) -> tuple[bytes, list[str]]:
    buf, names = io.BytesIO(), []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in FOLDERS:
            for path in sorted((root / folder).rglob("*")):
                rel = path.relative_to(root)
                if path.is_dir() or path.name in SKIP_FILES or SKIP_DIRS & set(rel.parts[:-1]):
                    continue
                name = rel.as_posix()  # forward slashes, even on Windows
                z.write(path, name)
                names.append(name)
    return buf.getvalue(), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic")
    ap.add_argument("--stack", default="chp-backup")
    args = ap.parse_args()
    session = boto3.Session(profile_name=args.profile, region_name="us-west-2")
    outputs = session.client("cloudformation").describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]
    bucket = next(o["OutputValue"] for o in outputs if o["OutputKey"] == "LakeBucket")
    data, names = build_zip()
    session.client("s3").put_object(Bucket=bucket, Key=KEY, Body=data)
    print(f"uploaded {len(names)} files ({len(data) / 1024:.0f} KB) to s3://{bucket}/{KEY}")


if __name__ == "__main__":
    main()
