import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "jobs"), str(ROOT / "scripts")]

from drop_dbt_tmp import drop_tmp_tables  # noqa: E402
from email_report import compose  # noqa: E402
from upload_job_code import build_zip  # noqa: E402


class FakeGlue:
    def __init__(self, names):
        self.names, self.deleted = names, []

    def get_paginator(self, _):
        names = self.names

        class P:
            def paginate(self, DatabaseName):
                return [{"TableList": [{"Name": n} for n in names[:2]]}, {"TableList": [{"Name": n} for n in names[2:]]}]
        return P()

    def delete_table(self, DatabaseName, Name):
        self.deleted.append(Name)


def test_only_leftover_dbt_tables_are_dropped():
    glue = FakeGlue(["fct_incident_backup", "x__dbt_tmp", "stg_chp_notes", "backup_label__dbt_tmp"])
    assert drop_tmp_tables("traffic_silver", glue) == ["x__dbt_tmp", "backup_label__dbt_tmp"]
    assert glue.deleted == ["x__dbt_tmp", "backup_label__dbt_tmp"]


def test_job_code_zip_has_the_code_and_not_local_data(tmp_path):
    for rel in ["transform/models/marts/a.sql", "transform/dbt_project.yml", "transform/target/compiled.sql",
                "transform/logs/dbt.log", "transform/.user.yml", "ml/train.py", "ml/data/cache.parquet",
                "ml/output/metrics.csv", "ml/__pycache__/x.pyc", "jobs/buildspec.yml", "src/chp/handler.py"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    _, names = build_zip(tmp_path)
    assert sorted(names) == ["jobs/buildspec.yml", "ml/train.py", "transform/dbt_project.yml",
                             "transform/models/marts/a.sql"]


def test_real_job_zip_contains_what_the_buildspec_runs():
    _, names = build_zip(ROOT)
    for needed in ["jobs/buildspec.yml", "jobs/requirements.txt", "jobs/drop_dbt_tmp.py", "jobs/email_report.py",
                   "transform/dbt_project.yml", "transform/profiles.yml", "ml/train.py", "ml/label_audit.py",
                   "ml/backup_model.py", "ml/evaluate.py", "ml/simple_models.py"]:
        assert needed in names, needed
    assert all("\\" not in n for n in names)


def test_buildspec_rebuilds_daily_and_reports_on_mondays():
    spec = yaml.safe_load((ROOT / "jobs/buildspec.yml").read_text())
    cmds = "\n".join(spec["phases"]["build"]["commands"])
    assert "dbt build --profiles-dir . --target aws" in cmds
    assert cmds.index("drop_dbt_tmp.py") < cmds.index("dbt build")
    assert 'date +%u)" = "1"' in cmds and "FORCE_REPORT" in cmds
    assert '--profile ""' in cmds


def test_aws_dbt_target_uses_the_job_role_not_a_login_profile():
    prof = yaml.safe_load((ROOT / "transform/profiles.yml").read_text())["traffic"]["outputs"]
    assert "aws_profile_name" not in prof["aws"]
    assert {k: v for k, v in prof["aws"].items()} == {k: v for k, v in prof["dev"].items() if k != "aws_profile_name"}


def test_email_has_scores_and_severity_without_progress_lines():
    train = "4,000 incidents\n  held out 2026-10-01 (1/13): 300 incidents\n=== with live traffic ===\nrule 0.79"
    sev = pd.DataFrame({"flag": ["SigAlert issued"], "has_it": ["yes"], "incidents": [20], "backup_rate": [0.5],
                        "backups": [10], "median_minutes": [55.0], "still_going_share": [0.1], "median_reach_miles": [2.2]})
    subject, body = compose(train, sev, "2026-10-19", "my-bucket")
    assert subject == "Traffic backup model: weekly report 2026-10-19" and len(subject) <= 100
    assert "held out" not in body and "rule 0.79" in body and "SigAlert issued" in body
    assert "aws s3 sync s3://my-bucket/reports/latest" in body
