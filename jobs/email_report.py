"""Email the weekly model report (sent through the stack's alert topic).

    python jobs/email_report.py train_output.txt label_audit_severity.csv 2026-10-19 [forward_output.txt]
"""

import os
import sys
from pathlib import Path

import boto3
import pandas as pd

MAX_CHARS = 200_000  # SNS messages are capped at 256 KB


def compose(train_output: str, severity: pd.DataFrame, day: str, bucket: str,
            forward_output: str = "") -> tuple[str, str]:
    lines = [ln for ln in train_output.splitlines() if "held out" not in ln]
    sev = severity.assign(backup_rate=lambda d: (d["backup_rate"] * 100).round(0)).rename(
        columns={"backup_rate": "backup_%", "median_minutes": "median_min", "median_reach_miles": "reach_mi",
                 "still_going_share": "still_going"})
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        sev_text = sev.to_string(index=False)
    body = "\n".join([
        f"Weekly traffic backup model report, {day}",
        "",
        "Everything below was computed automatically from the data collected so far.",
        "The full report (label audit page with speed charts, all scores as CSV) is saved in S3. To look at it:",
        f"    aws s3 sync s3://{bucket}/reports/latest ml/output/latest --profile traffic",
        "    then open ml/output/latest/label_audit.html",
        "",
        "=" * 70,
        "MODEL SCORES (leave-one-day-out)",
        "=" * 70,
        *lines,
        "",
        *(["=" * 70, "IN TIME ORDER: forward test, alert tradeoff, weekly retraining vs frozen", "=" * 70,
            forward_output, ""] if forward_output else []),
        "=" * 70,
        "LABEL CHECK: backups by incident severity",
        "=" * 70,
        sev_text,
    ])
    return f"Traffic backup model: weekly report {day}", body[:MAX_CHARS]


def main() -> None:
    train_path, severity_path, day = sys.argv[1], sys.argv[2], sys.argv[3]
    forward = Path(sys.argv[4]).read_text(encoding="utf-8") if len(sys.argv) > 4 else ""
    subject, body = compose(Path(train_path).read_text(encoding="utf-8"), pd.read_csv(severity_path),
                            day, os.environ["LAKE_BUCKET"], forward)
    boto3.client("sns").publish(TopicArn=os.environ["ALERT_TOPIC"], Subject=subject, Message=body)
    print(f"emailed: {subject}")


if __name__ == "__main__":
    main()
