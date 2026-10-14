"""Check whether the backup labels make sense, before more modelling.

    python ml/label_audit.py            # writes ml/output/label_audit.html and opens nothing
    python ml/label_audit.py --open     # also opens it in your browser

1. Severity check: if the labels are right, incidents with a SigAlert, an overturned vehicle,
   all lanes blocked or a fatality should back up more often and for longer than the rest.
2. Trace check: speed charts for 30 random backups and 10 "no backup" incidents where CHP said
   lanes were blocked, with the label's start and end drawn on, so you can see whether the
   label matches what the sensors show.
"""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from backup_model import km_median

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SLOW_RATIO = 0.6       # transform/dbt_project.yml slow_ratio
RECOVER_RATIO = 0.8    # transform/dbt_project.yml recover_ratio
ONSET_MINUTES = 45     # transform/dbt_project.yml onset_minutes
HEAD_MILES = 1.0       # transform/dbt_project.yml head_miles
FLAGS = {
    "sigalert": "SigAlert issued",
    "overturned": "Overturned vehicle",
    "all_lanes": "All lanes blocked",
    "fatality": "Fatality (1144)",
    "hazmat": "Spill / hazmat",
    "ambulance": "Ambulance",
    "tow": "Tow requested",
    "any_blocking": "Any lane blocked",
    "rush_hour": "Weekday rush hour (6-10 AM, 3-7 PM)",
    "weekend": "Weekend",
}

LABELS_SQL = """
with flags as (
    select
        log_id,
        bool_or(sigalert)    as sigalert,
        bool_or(overturned)  as overturned,
        bool_or(all_lanes)   as all_lanes,
        bool_or(fatality)    as fatality,
        bool_or(hazmat)      as hazmat,
        bool_or(ambulance)   as ambulance,
        bool_or(tow)         as tow,
        bool_or(is_blocking) as any_blocking
    from int_note_features
    group by log_id
)
select
    b.log_id, b.has_backup, b.backup_start_min, b.backup_end_min, b.backup_minutes, b.is_censored,
    b.observed_until_min, b.max_extent_miles, b.min_speed_ratio, b.label_adjusted_share,
    i.log_time, i.log_type, i.location, i.location_desc, i.freeway, i.direction, i.district,
    hour(i.log_time at time zone 'America/Los_Angeles')        as hour_local,
    day_of_week(i.log_time at time zone 'America/Los_Angeles') as day_of_week,
    coalesce(f.sigalert, false) as sigalert, coalesce(f.overturned, false) as overturned,
    coalesce(f.all_lanes, false) as all_lanes, coalesce(f.fatality, false) as fatality,
    coalesce(f.hazmat, false) as hazmat, coalesce(f.ambulance, false) as ambulance,
    coalesce(f.tow, false) as tow, coalesce(f.any_blocking, false) as any_blocking
from fct_incident_backup b
join stg_chp_incidents i on i.log_id = b.log_id
left join flags f on f.log_id = b.log_id
"""

TRACES_SQL = """
select log_id, station_id, miles_upstream, minutes_from_log, avg_speed, expected_speed, typical_speed,
       is_observed, baseline_method
from int_incident_traffic
where role = 'queue' and log_id in ({ids})
"""

NOTES_SQL = """
select log_id, detail_seq, first_seen_at, detail_text
from stg_chp_notes
where log_id in ({ids})
"""


# ---------- data ----------

def plain(df: pd.DataFrame) -> pd.DataFrame:
    """Athena returns pandas "nullable" types whose missing value (pd.NA) can't be used in an if or
    a format string. Convert to plain types: missing booleans -> False, numbers -> float with NaN,
    text -> str or None."""
    out = df.copy()
    for c in out.columns:
        dtype = str(out[c].dtype)
        if dtype == "boolean":
            out[c] = out[c].fillna(False).astype(bool)
        elif dtype in ("Int8", "Int16", "Int32", "Int64", "Float32", "Float64"):
            out[c] = out[c].astype("float64")
        elif dtype in ("string", "object"):
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out


def _text(value) -> str:
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else html.escape(str(value))


def _num(value, fmt: str = "{:.0f}", missing: str = "?") -> str:
    return missing if value is None or pd.isna(value) else fmt.format(value)


# ---------- severity check ----------

def add_derived(labels: pd.DataFrame) -> pd.DataFrame:
    df = plain(labels)
    for c in ["has_backup", "is_censored", *[f for f in FLAGS if f not in ("rush_hour", "weekend")]]:
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    df["weekend"] = df["day_of_week"].isin([6, 7])
    df["rush_hour"] = ~df["weekend"] & (df["hour_local"].between(6, 9) | df["hour_local"].between(15, 18))
    return df


def severity_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Backup rate and typical backup length with vs without each flag."""
    rows = []
    for flag, name in FLAGS.items():
        for value in (True, False):
            g = labels[labels[flag] == value]
            b = g[g["has_backup"]]
            rows.append({
                "flag": name, "has_it": "yes" if value else "no", "incidents": len(g),
                "backup_rate": round(g["has_backup"].mean(), 3) if len(g) else np.nan,
                "backups": len(b),
                "median_minutes": km_median(b["backup_minutes"], b["is_censored"]) if len(b) else np.nan,
                "still_going_share": round(b["is_censored"].mean(), 3) if len(b) else np.nan,
                "median_reach_miles": round(b["max_extent_miles"].median(), 1) if len(b) else np.nan,
            })
    return pd.DataFrame(rows)


def pick_sample(labels: pd.DataFrame, n_backups: int = 30, n_missed: int = 10, seed: int = 0) -> pd.DataFrame:
    """Random backups, plus "no backup" incidents where CHP said lanes were blocked (possible misses)."""
    backups = labels[labels["has_backup"]]
    missed = labels[~labels["has_backup"] & labels["any_blocking"]]
    picks = [backups.sample(min(n_backups, len(backups)), random_state=seed).assign(group="backup"),
             missed.sample(min(n_missed, len(missed)), random_state=seed).assign(group="no backup, lanes blocked")]
    return pd.concat(picks).reset_index(drop=True)


# ---------- page ----------

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 680, 230, 46, 12, 12, 30
X_MIN, X_MAX = -60, 180


def _x(minutes: float) -> float:
    return PAD_L + (minutes - X_MIN) / (X_MAX - X_MIN) * (W - PAD_L - PAD_R)


def _y(value: float, lo: float, hi: float) -> float:
    value = min(max(value, lo), hi)
    return PAD_T + (hi - value) / (hi - lo) * (H - PAD_T - PAD_B)


def _axes(lo: float, hi: float, ticks: list[float], fmt: str) -> list[str]:
    parts = []
    for t in ticks:
        y = _y(t, lo, hi)
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt.format(t)}</text>')
    for m in range(X_MIN, X_MAX + 1, 30):
        x = _x(m)
        anchor = "end" if m == X_MAX else "middle"
        parts.append(f'<text x="{x:.1f}" y="{H - 10}" class="tick" text-anchor="{anchor}">{m:+d} min</text>'
                     if m == X_MAX else f'<text x="{x:.1f}" y="{H - 10}" class="tick" text-anchor="{anchor}">{m:+d}</text>')
    return parts


def _lines(points: pd.DataFrame, value: str, lo: float, hi: float, cls: str, color: str = "") -> list[str]:
    """Polyline segments, broken wherever a 5-minute reading is missing."""
    out, seg, prev = [], [], None
    style = f' style="stroke:{color}"' if color else ""
    for _, r in points.sort_values("minutes_from_log").iterrows():
        if pd.isna(r[value]) or (prev is not None and r["minutes_from_log"] - prev > 5):
            if len(seg) > 1:
                out.append(f'<polyline class="{cls}"{style} points="{" ".join(seg)}"/>')
            seg = []
        if not pd.isna(r[value]):
            seg.append(f"{_x(r['minutes_from_log']):.1f},{_y(r[value], lo, hi):.1f}")
        prev = r["minutes_from_log"]
    if len(seg) > 1:
        out.append(f'<polyline class="{cls}"{style} points="{" ".join(seg)}"/>')
    return out


def _marks(label: pd.Series) -> list[str]:
    parts = []
    if label["has_backup"]:
        start = label["backup_start_min"]
        end = label["backup_end_min"] if pd.notna(label["backup_end_min"]) else start + label["backup_minutes"]
        parts.append(f'<rect x="{_x(start):.1f}" y="{PAD_T}" width="{max(_x(end) - _x(start), 1):.1f}" '
                     f'height="{H - PAD_T - PAD_B}" class="{"censored" if label["is_censored"] else "backup"}"/>')
    for m, cls in ((0, "logline"), (ONSET_MINUTES, "onset")):
        parts.append(f'<line x1="{_x(m):.1f}" x2="{_x(m):.1f}" y1="{PAD_T}" y2="{H - PAD_B}" class="{cls}"/>')
    return parts


def ratio_chart(traces: pd.DataFrame, label: pd.Series) -> str:
    """Every queue sensor's speed / expected speed. Below the red line = backed up."""
    lo, hi = 0.0, 1.4
    parts = _axes(lo, hi, [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], "{:.1f}") + _marks(label)
    for level, cls in ((SLOW_RATIO, "threshold"), (RECOVER_RATIO, "recover")):
        y = _y(level, lo, hi)
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="{cls}"/>')
    obs = traces[traces["is_observed"] & (traces["expected_speed"] > 0)].copy()
    obs["ratio"] = obs["avg_speed"] / obs["expected_speed"]
    sensors = obs.groupby("station_id")["miles_upstream"].first().sort_values()
    for i, (sid, miles) in enumerate(sensors.items()):
        shade = 30 + int(50 * i / max(len(sensors) - 1, 1))  # near = dark, far = light
        dash = "" if miles <= HEAD_MILES else ";stroke-dasharray:3 2"  # dashed = beyond the crash site
        parts += _lines(obs[obs["station_id"] == sid], "ratio", lo, hi, "sensor", f"hsl(212 70% {shade}%){dash}")
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="speed ratio chart">{"".join(parts)}</svg>'


def speed_chart(traces: pd.DataFrame, label: pd.Series) -> str:
    """The sensor nearest the crash: actual speed vs normal (typical) and expected speed."""
    obs = traces[traces["is_observed"] & traces["avg_speed"].notna()]
    if obs.empty:
        return "<p class='muted'>No measured readings.</p>"
    near = obs[obs["miles_upstream"] >= 0]
    sid = (near if not near.empty else obs).sort_values("miles_upstream")["station_id"].iloc[0]
    t = traces[traces["station_id"] == sid]
    lo, hi = 0.0, 80.0
    parts = _axes(lo, hi, [0, 20, 40, 60, 80], "{:.0f}") + _marks(label)
    parts += _lines(t, "typical_speed", lo, hi, "typical")
    parts += _lines(t, "expected_speed", lo, hi, "expected")
    parts += _lines(t[t["is_observed"]], "avg_speed", lo, hi, "actual")
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="speed chart">{"".join(parts)}</svg>'
            f'<p class="muted">Sensor {_num(sid)}, {_num(t["miles_upstream"].iloc[0], "{:.1f}")} mi upstream of the crash.</p>')


def incident_card(i: int, label: pd.Series, traces: pd.DataFrame, notes: pd.DataFrame) -> str:
    local = pd.Timestamp(label["log_time"]).tz_localize("UTC") if pd.Timestamp(label["log_time"]).tzinfo is None \
        else pd.Timestamp(label["log_time"])
    local = local.tz_convert("America/Los_Angeles")
    if label["has_backup"]:
        end = (f"still going at +{_num(label['observed_until_min'])} min (censored)" if label["is_censored"]
               else f"ended at +{_num(label['backup_end_min'])} min")
        verdict = (f"<b>Backup</b> from +{_num(label['backup_start_min'])} min, {end}: "
                   f"<b>{_num(label['backup_minutes'])} min</b>, reached {_num(label['max_extent_miles'], '{:.1f}')} mi back")
    else:
        verdict = (f"<b>No backup</b> (the crash site wasn't below {SLOW_RATIO:.0%} of expected for 10+ min "
                   f"within {ONSET_MINUTES} min)")
    flags = ", ".join(name for f, name in FLAGS.items() if f in label and bool(label[f]) and f != "any_blocking")
    note_rows = []
    log_utc = pd.Timestamp(label["log_time"])
    for _, n in notes.sort_values(["first_seen_at", "detail_seq"]).head(40).iterrows():
        if n["first_seen_at"] is None or pd.isna(n["first_seen_at"]):
            continue
        seen = pd.Timestamp(n["first_seen_at"])
        if seen.tzinfo is None and log_utc.tzinfo is not None:
            seen = seen.tz_localize("UTC")
        if seen.tzinfo is not None and log_utc.tzinfo is None:
            seen = seen.tz_convert("UTC").tz_localize(None)
        mins = (seen - log_utc).total_seconds() / 60
        note_rows.append(f"<tr><td>{_num(mins, '{:+.0f}')}</td><td>{_text(n['detail_text'])}</td></tr>")
    quality = label.get("label_adjusted_share")
    quality_txt = f"{_num(quality, '{:.0%}', 'unknown')} full baseline"
    return f"""
<section class="card" data-i="{i}">
  <h2>#{i + 1} <span class="pill">{_text(label['group'])}</span></h2>
  <p class="meta">{_text(label['location'])} &middot; {_text(label['location_desc'])}<br>
     {_text(label['log_type'])} &middot; {local:%a %b %d, %I:%M %p} &middot; log {_text(label['log_id'])}
     &middot; label quality: {quality_txt}</p>
  <p class="verdict">{verdict}</p>
  <p class="meta">{html.escape(flags) or "No severity flags in the notes"}</p>
  <h3>Speed / expected speed, every sensor in the 5 miles behind the crash</h3>
  {ratio_chart(traces, label)}
  <h3>Sensor nearest the crash: actual vs normal vs expected (mph)</h3>
  {speed_chart(traces, label)}
  <details><summary>Dispatcher notes (minutes after the log time our poller first saw them)</summary>
    <table class="notes">{"".join(note_rows) or "<tr><td colspan=2>No notes</td></tr>"}</table></details>
</section>"""


CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --grid:#e6e6ea; --card:#fafafb; --line:#d2d2d7;
        --backup:rgba(255,159,10,.18); --censored:rgba(191,90,242,.16); --red:#d70015; }
@media (prefers-color-scheme: dark) { :root { --bg:#1c1c1e; --fg:#f2f2f7; --muted:#a1a1a6; --grid:#3a3a3c;
        --card:#2c2c2e; --line:#48484a; --backup:rgba(255,159,10,.25); --censored:rgba(191,90,242,.25); --red:#ff6961; } }
body { font: 15px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--fg);
       max-width: 760px; margin: 0 auto; padding: 16px; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin: 0 0 4px; } h3 { font-size: 13px; color: var(--muted); margin: 14px 0 2px; font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--grid); }
.sev td:nth-child(n+3), .sev th:nth-child(n+3) { text-align: right; }
.sev tr.yes td { font-weight: 600; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; display: none; }
.card.on { display: block; }
.nav { display: flex; gap: 8px; align-items: center; margin: 16px 0 10px; position: sticky; top: 0; background: var(--bg); padding: 6px 0; }
button, select { font: inherit; padding: 4px 12px; border-radius: 6px; border: 1px solid var(--line); background: var(--card); color: var(--fg); }
.meta, .muted { color: var(--muted); font-size: 13px; margin: 2px 0; }
.verdict { margin: 6px 0; }
.pill { font-size: 12px; font-weight: 500; padding: 1px 8px; border-radius: 99px; border: 1px solid var(--line); color: var(--muted); }
svg { width: 100%; height: auto; display: block; }
svg .grid { stroke: var(--grid); } svg .tick { fill: var(--muted); font-size: 10px; }
svg polyline { fill: none; stroke-width: 1.6; }
svg .threshold { stroke: var(--red); stroke-dasharray: 5 4; } svg .recover { stroke: #30a46c; stroke-dasharray: 5 4; } svg .logline { stroke: var(--fg); stroke-width: 1.2; }
svg .onset { stroke: var(--muted); stroke-dasharray: 2 3; }
svg .backup { fill: var(--backup); } svg .censored { fill: var(--censored); }
svg .actual { stroke: #0a84ff; stroke-width: 2; } svg .typical { stroke: var(--muted); stroke-dasharray: 4 3; }
svg .expected { stroke: #30a46c; }
.legend span { margin-right: 14px; white-space: nowrap; }
.notes td:first-child { width: 48px; color: var(--muted); text-align: right; }
"""

JS = """
const cards = [...document.querySelectorAll('.card')];
const pick = document.getElementById('pick');
let i = 0;
function show(n) { i = (n + cards.length) % cards.length; cards.forEach((c, k) => c.classList.toggle('on', k === i));
  pick.value = i; document.getElementById('pos').textContent = `${i + 1} / ${cards.length}`; }
document.getElementById('prev').onclick = () => show(i - 1);
document.getElementById('next').onclick = () => show(i + 1);
pick.onchange = () => show(+pick.value);
document.addEventListener('keydown', e => { if (e.key === 'ArrowRight') show(i + 1); if (e.key === 'ArrowLeft') show(i - 1); });
show(0);
"""


def build_page(severity: pd.DataFrame, sample: pd.DataFrame, traces: pd.DataFrame, notes: pd.DataFrame) -> str:
    sample, traces, notes = plain(sample), plain(traces), plain(notes)
    sev_rows = "".join(
        f'<tr class="{r.has_it}"><td>{html.escape(r.flag) if r.has_it == "yes" else ""}</td><td>{r.has_it}</td>'
        f"<td>{r.incidents:,}</td><td>{'&ndash;' if pd.isna(r.backup_rate) else f'{r.backup_rate:.0%}'}</td><td>{r.backups:,}</td>"
        f"<td>{'' if pd.isna(r.median_minutes) else f'{r.median_minutes:.0f}'}</td>"
        f"<td>{'' if pd.isna(r.still_going_share) else f'{r.still_going_share:.0%}'}</td>"
        f"<td>{'' if pd.isna(r.median_reach_miles) else f'{r.median_reach_miles:.1f}'}</td></tr>"
        for r in severity.itertuples()
    )
    cards = "".join(
        incident_card(i, row, traces[traces["log_id"] == row["log_id"]], notes[notes["log_id"] == row["log_id"]])
        for i, row in sample.iterrows()
    )
    options = "".join(f'<option value="{i}">#{i + 1} {_text(r["group"])}</option>' for i, r in sample.iterrows())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Backup Label Audit</title>
<style>{CSS}</style></head><body>
<h1>Backup label audit</h1>
<p>If the labels are right, the bolded "yes" rows should back up more often and for longer than their "no" rows.
Median minutes counts backups still going when the data ended as "at least that long" (Kaplan-Meier).</p>
<table class="sev"><tr><th>Flag</th><th></th><th>Incidents</th><th>Backed up</th><th>Backups</th>
<th>Median min</th><th>Still going</th><th>Median reach (mi)</th></tr>{sev_rows}</table>
<h2 style="margin-top:22px">Speed traces</h2>
<p class="muted legend"><span>Top chart: each line is a sensor's speed as a share of its expected speed (darker = closer to the crash;
dashed = more than 1 mile back). A backup has to start at the crash site: a solid line below the
<b style="color:var(--red)">red dashed line</b> (60%) for 10+ minutes. It ends once the crash site is back above the
<b style="color:#30a46c">green dashed line</b> (80%) for 15 minutes.</span>
<span><b>Black line</b> = CHP log time.</span><span>Dotted line = 45 min (latest a backup can start).</span>
<span>Orange band = labeled backup; purple = still going when we lost sight of it.</span>
<span>Bottom chart: <b style="color:#0a84ff">actual speed</b>, <b style="color:#30a46c">expected speed</b> (the label's baseline), dashed gray = normal for this time of day.</span></p>
<p class="muted">Use the arrow keys or buttons. Note the numbers of any incident whose label looks wrong.</p>
<div class="nav"><button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
<select id="pick">{options}</select><span id="pos" class="muted"></span></div>
{cards}
<script>{JS}</script></body></html>"""


# ---------- run ----------

def query(sql: str, session) -> pd.DataFrame:
    import awswrangler as wr
    return wr.athena.read_sql_query(sql, database="traffic_silver", workgroup="traffic",
                                    boto3_session=session, ctas_approach=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic", help='AWS login profile; "" = default credentials (the AWS job)')
    ap.add_argument("--seed", type=int, default=0, help="change to see a different random sample")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    import boto3
    session = boto3.Session(profile_name=args.profile or None, region_name="us-west-2")
    labels = add_derived(query(LABELS_SQL, session))
    severity = severity_table(labels)
    sample = pick_sample(labels, seed=args.seed)
    ids = ", ".join(f"'{x}'" for x in sample["log_id"])
    traces = query(TRACES_SQL.format(ids=ids), session)
    notes = query(NOTES_SQL.format(ids=ids), session)

    OUT.mkdir(exist_ok=True)
    page = OUT / "label_audit.html"
    page.write_text(build_page(severity, sample, traces, notes), encoding="utf-8")
    severity.to_csv(OUT / "label_audit_severity.csv", index=False)
    (OUT / "label_audit_sample.json").write_text(json.dumps({
        "sample": sample.astype(str).to_dict("records"),
        "traces": traces.astype(str).to_dict("records"),
    }), encoding="utf-8")

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(severity.to_string(index=False))
    print(f"\nWrote {page}")
    if args.open:
        webbrowser.open(page.as_uri())


if __name__ == "__main__":
    main()
