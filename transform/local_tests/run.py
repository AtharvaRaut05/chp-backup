"""Run the dbt unit tests (the unit_tests: blocks in transform/models/**/*.yml) locally in DuckDB.

dbt only runs them against Athena, so a mistake in a model or a test otherwise shows up when the
daily job fails. This renders each model's SQL (refs, sources, vars and macros), loads the test's
given rows into DuckDB tables, runs the model and compares the expected rows. Trino functions DuckDB
lacks are added as macros below. It's a quick check, not a replacement: Athena has the final say
(for example, it requires every expected row to list the same columns, which is checked here too).

    python transform/local_tests/run.py                   # every unit test
    python transform/local_tests/run.py int_incident_calendar stg_chp_notes   # tests of these models
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import duckdb
import jinja2
import yaml

T = Path(__file__).resolve().parents[1]
SCHEMAS = yaml.safe_load(open(Path(__file__).with_name("schemas.yml")))
VARS = yaml.safe_load(open(T / "dbt_project.yml"))["vars"]
SHIMS = """
create macro great_circle_distance(a, b, c, d) as 6371.01 * acos(least(1.0, greatest(-1.0,
  sin(radians(a)) * sin(radians(c)) + cos(radians(a)) * cos(radians(c)) * cos(radians(d - b)))));
create macro to_unixtime(t) as epoch(t);
create macro day_of_week(t) as isodow(t);
create macro approx_percentile(x, q) as quantile_cont(x, q);
create macro regexp_like(s, p) as regexp_matches(s, p);
create macro trino_greatest(a, b) as case when a is null or b is null then null else greatest(a, b) end;
create macro trino_least(a, b) as case when a is null or b is null then null else least(a, b) end;
create macro iff(c, a, b := null) as case when c then a else b end;
create macro trino_date_add(u, n, t) as t + case u when 'day' then to_days(cast(n as integer))
  when 'hour' then to_hours(cast(n as integer)) else to_minutes(cast(n as integer)) end;
"""
_TS = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def guess(values) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "VARCHAR"
    if all(isinstance(v, bool) for v in vals):
        return "BOOLEAN"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return "BIGINT"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return "DOUBLE"
    if all(isinstance(v, str) and _TS.match(v) for v in vals):
        return "TIMESTAMP"
    return "VARCHAR"


def render(model: str) -> str:
    path = next(T.glob(f"models/**/{model}.sql"))
    macros = "".join(p.read_text() for p in (T / "macros").glob("*.sql"))
    sql = jinja2.Template(macros + path.read_text()).render(ref=lambda n: n, source=lambda s, n: n,
                                                            var=lambda n: VARS[n], config=lambda **k: "")
    sql = re.sub(r"\bdate_add\('(\w+)',", r"trino_date_add('\1',", sql)
    for trino, duck in (("greatest", "trino_greatest"), ("least", "trino_least"), ("transform", "list_transform"),
                        ("cardinality", "len"), ("array_distinct", "list_distinct"), ("sequence", "generate_series")):
        sql = re.sub(rf"\b{trino}\(", f"{duck}(", sql)
    return re.sub(r"\bif\(", "iff(", sql)


def same(got, want) -> bool:
    if got is None or (isinstance(got, float) and math.isnan(got)) or str(got) in ("NaT", "<NA>", "None", "nan"):
        got = None
    if got is None or want is None:
        return got is None and want is None
    got = str(got).removesuffix("+00:00") if hasattr(got, "tzinfo") else got
    if isinstance(want, bool):
        return bool(got) == want
    if isinstance(want, (int, float)):
        return abs(float(got) - float(want)) < 1e-6
    return str(got) == str(want) or str(got).startswith(str(want) + " ") or str(got)[:10] == str(want) == str(got)[:10]


def run(test: dict, overrides: dict) -> list[str]:
    errors = []
    con = duckdb.connect()
    con.execute("set TimeZone = 'UTC'")
    try:  # Trino's lambda syntax (x -> ...)
        con.execute("set lambda_syntax = 'ENABLE_SINGLE_ARROW'")
    except duckdb.Error:
        pass
    con.execute(SHIMS)
    for part, rows in [(g["input"], g["rows"]) for g in test["given"]] + [("expect", test["expect"]["rows"])]:
        if len({frozenset(r) for r in rows}) > 1:
            errors.append(f"{part}: rows list different columns (Athena rejects this)")
    for g in test["given"]:
        name = re.search(r"(?:ref|source)\((?:'\w+', )?'(\w+)'\)", g["input"]).group(1)
        schema = dict(SCHEMAS.get(name, {}))
        for col in {k for r in g["rows"] for k in r}:
            schema.setdefault(col, guess([r.get(col) for r in g["rows"]]))
        # TIMESTAMP WITH TIME ZONE in UTC behaves like Athena's timestamps under "at time zone"
        types = {c: "TIMESTAMPTZ" if t == "TIMESTAMP" else t for c, t in schema.items()}
        con.execute(f"create or replace table {name} (" + ", ".join(f'"{c}" {t}' for c, t in types.items()) + ")")
        for r in g["rows"]:
            cols = list(r)
            con.execute(f"insert into {name} ({', '.join(chr(34) + c + chr(34) for c in cols)}) "
                        f"values ({', '.join('?' for _ in cols)})", [r[c] for c in cols])
    VARS.update(overrides)
    try:
        out = con.execute(render(test["model"])).df()
    except duckdb.Error as e:
        if "perl operator" in str(e):  # Java-style regex (look-ahead) that DuckDB's RE2 can't run
            return ["SKIP"]
        return errors + [f"model failed in DuckDB: {str(e).splitlines()[0]}"]
    want = test["expect"]["rows"]
    if len(out) != len(want):
        errors.append(f"{len(out)} rows, expected {len(want)}")
    for exp in want:
        keys = [k for k in list(exp)[:1] + ["landmark_min", "t", "minutes_from_log", "slot", "station_id"] if k in exp]
        mask = True
        for k in keys:
            mask = mask & (out[k].astype(str) == str(exp[k]))
        got = out[mask]
        if len(got) != 1:
            errors.append(f"row {[exp[k] for k in keys]}: found {len(got)} matches")
            continue
        for c, v in exp.items():
            if c not in got.columns:
                errors.append(f"row {[exp[k] for k in keys]}: no column {c}")
            elif not same(got.iloc[0][c], v):
                errors.append(f"row {[exp[k] for k in keys]}.{c}: expected {v!r}, got {got.iloc[0][c]!r}")
    return errors


def main() -> int:
    wanted = set(sys.argv[1:])
    base_vars = dict(VARS)
    failed = 0
    for f in sorted(T.glob("models/**/*.yml")):
        for test in (yaml.safe_load(open(f)) or {}).get("unit_tests", []):
            if wanted and test["model"] not in wanted:
                continue
            VARS.clear()
            VARS.update(base_vars)
            errors = run(test, test.get("overrides", {}).get("vars", {}))
            if errors == ["SKIP"]:
                print(f"skip {test['model']}: {test['name']} (regex DuckDB can't run; Athena checks it)")
                continue
            print(("FAIL " if errors else "pass ") + f"{test['model']}: {test['name']}")
            for e in errors:
                print("     " + e)
            failed += bool(errors)
    print(f"\n{failed} failed" if failed else "\nall passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
