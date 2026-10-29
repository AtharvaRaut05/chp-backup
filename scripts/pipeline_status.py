"""Where is data stopping? One report covering every stage of the pipeline, newest days first.

    python scripts/pipeline_status.py            # prints and saves ml/output/pipeline_status.txt
    python scripts/pipeline_status.py --days 21

  1. alarms              any CloudWatch alarm currently firing, and whether the alert email is confirmed
     collectors          runs and errors per hour for the CHP poller and the travel-time collector
                         over the last 6 hours, with their latest error messages
  2. daily workflow      recent Step Functions runs and the error of any failed one
  3. processing job      recent CodeBuild runs (the dbt rebuild / weekly report)
  4. rows per day        CHP incidents (bronze), PeMS readings (bronze), pairing status,
                         labeled incidents and training rows (silver), side by side
  5. funnel              per dispatch center, how many incidents survive each step from the feed
                         to a label, and how many sensors each district has, to find where a
                         region drops out
"""

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

STACK = "chp-backup"
OUT = Path(__file__).resolve().parents[1] / "ml" / "output" / "pipeline_status.txt"


def stack_outputs(session) -> dict:
    stack = session.client("cloudformation").describe_stacks(StackName=STACK)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def alarms(session) -> None:
    firing = session.client("cloudwatch").describe_alarms(StateValue="ALARM")["MetricAlarms"]
    print("ALARMS FIRING:", ", ".join(a["AlarmName"] for a in firing) or "none")
    sns = session.client("sns")
    for topic in sns.list_topics()["Topics"]:
        if STACK in topic["TopicArn"]:
            for sub in sns.list_subscriptions_by_topic(TopicArn=topic["TopicArn"])["Subscriptions"]:
                confirmed = sub["SubscriptionArn"] not in ("PendingConfirmation", "Deleted")
                print(f"ALERT EMAIL: {sub['Protocol']} subscription {'confirmed' if confirmed else 'NOT CONFIRMED (no emails are sent)'}")


def collectors(session, hours: int = 6) -> None:
    from datetime import datetime, timedelta, timezone

    cfn, cw, logs = session.client("cloudformation"), session.client("cloudwatch"), session.client("logs")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    for logical in ("ChpPoller", "CaltransTravelTimes"):
        try:
            name = cfn.describe_stack_resource(StackName=STACK, LogicalResourceId=logical)[
                "StackResourceDetail"]["PhysicalResourceId"]
        except Exception:  # noqa: BLE001  (not deployed yet)
            print(f"\n{logical}: not deployed")
            continue
        counts = {}
        for metric in ("Invocations", "Errors"):
            points = cw.get_metric_statistics(Namespace="AWS/Lambda", MetricName=metric, StartTime=start, EndTime=end,
                                              Period=3600, Statistics=["Sum"],
                                              Dimensions=[{"Name": "FunctionName", "Value": name}])["Datapoints"]
            counts[metric] = {p["Timestamp"]: int(p["Sum"]) for p in points}
        print(f"\n{logical.upper()} (last {hours} hours, UTC)")
        for ts in sorted(set(counts["Invocations"]) | set(counts["Errors"])):
            print(f"  {ts:%m-%d %H:00}  runs {counts['Invocations'].get(ts, 0):>3}  errors {counts['Errors'].get(ts, 0):>3}")
        try:
            events = logs.filter_log_events(logGroupName=f"/aws/lambda/{name}", startTime=int(start.timestamp() * 1000),
                                            filterPattern='?ERROR ?"Task timed out" ?Traceback ?FeedError',
                                            limit=200)["events"]
        except Exception as e:  # noqa: BLE001
            print(f"  (logs not readable: {str(e)[:120]})")
            continue
        if not events:
            print("  no error messages in the logs")
        for e in events[-8:]:
            print(f"  {datetime.fromtimestamp(e['timestamp'] / 1000, timezone.utc):%m-%d %H:%M}  {e['message'].strip()[:300]}")


def workflow(session, arn: str, n: int = 10) -> None:
    sf = session.client("stepfunctions")
    runs = sf.list_executions(stateMachineArn=arn, maxResults=n)["executions"]
    print("\nDAILY WORKFLOW (latest first)")
    for r in runs:
        line = f"  {r['startDate']:%Y-%m-%d %H:%M}  {r['status']}"
        if r["status"] != "SUCCEEDED" and r["status"] != "RUNNING":
            d = sf.describe_execution(executionArn=r["executionArn"])
            line += f"  {d.get('error', '')}: {str(d.get('cause', ''))[:300]}"
        print(line)


def job(session, name: str, n: int = 10) -> None:
    cb = session.client("codebuild")
    ids = cb.list_builds_for_project(projectName=name, sortOrder="DESCENDING")["ids"][:n]
    print("\nPROCESSING JOB (latest first)")
    for b in cb.batch_get_builds(ids=ids)["builds"] if ids else []:
        failed = [p for p in b.get("phases", []) if p.get("phaseStatus") not in (None, "SUCCEEDED")]
        why = "; ".join(f"{p['phaseType']}: {' '.join(c.get('message', '') for c in p.get('contexts', []))}"[:300]
                        for p in failed)
        print(f"  {b['startTime']:%Y-%m-%d %H:%M}  {b['buildStatus']}  {why}")
    builds = cb.batch_get_builds(ids=ids[:1])["builds"] if ids else []
    if builds and builds[0]["buildStatus"] == "FAILED" and builds[0].get("logs", {}).get("streamName"):
        log_tail(session, builds[0]["logs"])


def log_tail(session, logs: dict, n: int = 60) -> None:
    """The latest failed run's errors (dbt prints a summary of them at the end) and last lines."""
    events = session.client("logs").get_log_events(logGroupName=logs["groupName"], logStreamName=logs["streamName"],
                                                    startFromHead=False, limit=2000)["events"]
    lines = [e["message"].rstrip() for e in events]
    flagged = [ln for ln in lines if any(k in ln for k in ("Error", "ERROR", "Failure", "FAIL ", "Compilation"))]
    print("\nLATEST FAILED RUN: error lines")
    print("\n".join("  " + ln[:400] for ln in flagged[-40:]) or "  none found")
    print(f"\nLATEST FAILED RUN: last {n} lines")
    print("\n".join("  " + ln[:400] for ln in lines[-n:]))


def per_day(session, days: int) -> None:
    def q(sql, db):
        try:
            return wr.athena.read_sql_query(sql, database=db, workgroup="traffic", boto3_session=session,
                                            ctas_approach=False)
        except Exception as e:  # noqa: BLE001  (show what failed and keep going)
            print(f"  query failed ({db}): {str(e)[:300]}")
            return pd.DataFrame(columns=["day"])

    since = f"date_add('day', -{days}, current_date)"
    parts = [
        q(f"select date(date_parse(dt, '%Y-%m-%d')) as day, count(distinct log_id) as chp_incidents_seen "
          f"from chp_incidents where date(date_parse(dt, '%Y-%m-%d')) >= {since} group by 1", "traffic_bronze"),
        q(f"select date(date_parse(dt, '%Y-%m-%d')) as day, count(distinct district) as pems_districts, "
          f"count(*) as pems_rows from pems_station_5min where date(date_parse(dt, '%Y-%m-%d')) >= {since} group by 1",
          "traffic_bronze"),
        q(f"select day, array_join(array_agg(distinct status), ',') as pairing from stg_pems_pairing "
          f"where day >= {since} group by 1", "traffic_silver"),
        q(f"select date(log_time at time zone 'America/Los_Angeles') as day, count(distinct log_id) as labeled_incidents "
          f"from fct_training_landmarks where log_time >= {since} group by 1", "traffic_silver"),
        q(f"select date(log_time at time zone 'America/Los_Angeles') as day, count(*) as training_rows "
          f"from fct_training_landmarks where log_time >= {since} group by 1", "traffic_silver"),
    ]
    table = None
    for p in parts:
        p["day"] = pd.to_datetime(p["day"]).dt.date
        table = p if table is None else table.merge(p, on="day", how="outer")
    print("\nROWS PER DAY (newest first)")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table.sort_values("day", ascending=False).to_string(index=False))


def funnel(session, days: int) -> None:
    def q(sql):
        return wr.athena.read_sql_query(sql, database="traffic_silver", workgroup="traffic",
                                        boto3_session=session, ctas_approach=False)

    since = f"date_add('day', -{days}, current_date)"
    steps = q(f"""
        select i.dispatch_id, i.district,
               count(*)                                                        as incidents,
               count_if(i.is_traffic)                                          as traffic,
               count_if(i.is_traffic and i.freeway is not null)                as freeway_parsed,
               count_if(i.is_traffic and i.freeway is not null and i.direction is not null
                        and i.lat is not null)                                 as located,
               count_if(i.is_traffic and i.pairing_status = 'paired')          as paired,
               count(distinct s.log_id)                                        as near_a_sensor,
               count(distinct if(p.is_measurable, p.log_id))                   as measurable,
               count(distinct b.log_id)                                        as labeled
        from stg_chp_incidents i
        left join (select distinct log_id from int_incident_stations) s on s.log_id = i.log_id
        left join int_incident_precrash p on p.log_id = i.log_id
        left join fct_incident_backup b on b.log_id = i.log_id
        where i.incident_day >= {since}
        group by 1, 2 order by labeled desc""")
    sensors = q("""
        select district, type, count(*) as stations, count_if(lat is not null and abs_pm is not null) as usable
        from stg_pems_stations where type in ('ML', 'OR', 'FR', 'FF')
        group by 1, 2 order by 1, 2""")
    health = q("""
        select s.district,
               count(*)                                                as station_days,
               round(avg(h.share_measured), 3)                         as avg_share_measured,
               round(avg(if(h.is_healthy, 1.0, 0.0)), 3)               as share_healthy,
               round(avg(if(h.share_measured >= 0.5, 1.0, 0.0)), 3)    as share_half_measured,
               round(avg(if(h.share_measured = 0, 1.0, 0.0)), 3)       as share_never_measured
        from int_station_day_health h
        join stg_pems_stations s on s.station_id = h.station_id
        where h.lane_type = 'ML' and h.dt >= cast(date_add('day', -7, current_date) as varchar)
        group by 1 order by 1""")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print("\nMAINLINE SENSOR HEALTH BY DISTRICT, last 7 days (healthy = at least 80% of a day's readings measured)")
        print(health.to_string(index=False))
        print(f"\nFUNNEL BY DISPATCH CENTER (last {days} days): incidents left after each step")
        print(steps.to_string(index=False))
        print("\nSENSORS BY DISTRICT AND TYPE (ML mainline, OR on-ramp, FR off-ramp, FF connector)")
        print(sensors.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic")
    ap.add_argument("--days", type=int, default=21)
    args = ap.parse_args()
    session = boto3.Session(profile_name=args.profile or None, region_name="us-west-2")
    buf = io.StringIO()
    with redirect_stdout(buf):
        outputs = stack_outputs(session)
        for step in (lambda: alarms(session), lambda: collectors(session),
                     lambda: workflow(session, outputs["WorkflowArn"]),
                     lambda: job(session, outputs["ProcessingJobName"]), lambda: per_day(session, args.days),
                     lambda: funnel(session, args.days)):
            try:
                step()
            except Exception as e:  # noqa: BLE001
                print(f"  failed: {type(e).__name__}: {str(e)[:300]}")
    text = buf.getvalue()
    print(text)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
