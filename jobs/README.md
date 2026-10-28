# Scheduled processing (runs in AWS, no laptop needed)

Every morning the daily workflow (Step Functions, 10:00 AM Pacific) downloads yesterday's PeMS
traffic data, then starts this job (AWS CodeBuild project `chp-backup-processing`):

| When | What |
|---|---|
| Every day | Drops leftover dbt temp tables, then `dbt seed` + `dbt build` (all tables and tests) |
| Mondays | Also retrains the models (`ml/train.py`), runs the time-ordered test: forward test, alert tradeoff, retraining vs. frozen, failure cases (`ml/forward_test.py`), replaces the stored model in `s3://<lake bucket>/models/current/` only if the retrained one is at least as good on the latest 7 days (`ml/promote.py`), reruns the label audit (`ml/label_audit.py`), saves everything to `s3://<lake bucket>/reports/<date>/` and `reports/latest/`, and emails a summary |

If anything fails, the workflow is marked failed and the alarm emails you (once you've confirmed the alert subscription email). To see where a failure happened, including the error lines from the latest failed run:

```powershell
python scripts/pipeline_status.py
```

## After changing dbt models or ML code

The job runs a copy of `transform/`, `ml/` and `jobs/` stored in S3 (`jobs/code.zip`). If that copy is missing, every run fails before it starts. Upload the new version:

```powershell
python scripts/upload_job_code.py
```

## Run it now, with the model report

```powershell
aws codebuild start-build --project-name chp-backup-processing --environment-variables-override name=FORCE_REPORT,value=1,type=PLAINTEXT --profile traffic
```

## Look at the latest report

```powershell
aws s3 sync s3://<lake bucket>/reports/latest ml/output/latest --profile traffic
start ml/output/latest/label_audit.html
```

Logs: AWS console > CodeBuild > `chp-backup-processing` > Build history (kept 30 days).
