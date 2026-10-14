import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from label_audit import add_derived, build_page, pick_sample, severity_table  # noqa: E402


def fake_labels(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    severe = rng.random(n) < 0.2
    has = rng.random(n) < np.where(severe, 0.7, 0.15)
    minutes = np.where(has, np.where(severe, 60, 20) * rng.lognormal(0, 0.3, n) // 5 * 5 + 5, 0)
    return pd.DataFrame({
        "log_id": [f"L{i}" for i in range(n)], "has_backup": pd.array(has, dtype="boolean"),
        "backup_start_min": np.where(has, 0, np.nan), "backup_end_min": np.where(has, minutes, np.nan),
        "backup_minutes": minutes, "is_censored": pd.array(has & (minutes > 150), dtype="boolean"),
        "observed_until_min": pd.array([185] * (n - 1) + [None], dtype="Int64"),
        "max_extent_miles": pd.array(np.where(has, 2.0, np.nan), dtype="Float64"),
        "min_speed_ratio": 0.4, "label_adjusted_share": pd.array([1.0, None] * (n // 2), dtype="Float64"),
        "log_time": pd.Timestamp("2026-10-01 15:00") + pd.to_timedelta(rng.integers(0, 9 * 1440, n), "min"),
        "log_type": "1183-Trfc Collision",
        "location": pd.array(["I405 N / Sunset Blvd"] * (n - 1) + [None], dtype="string"),
        "location_desc": pd.array(["<b>x</b> & y", None] * (n // 2), dtype="string"),
        "freeway": 405, "direction": "N", "district": "07",
        "hour_local": rng.integers(0, 24, n), "day_of_week": rng.integers(1, 8, n),
        "sigalert": pd.array(severe, dtype="boolean"), "overturned": False, "all_lanes": False,
        "fatality": False, "hazmat": False, "ambulance": False, "tow": False,
        "any_blocking": pd.array(severe | (rng.random(n) < 0.3), dtype="boolean"),
    })


def fake_traces(ids) -> pd.DataFrame:
    rows = []
    for lid in ids:
        for sid, miles in ((1, 0.2), (2, 2.5)):
            for t in range(-60, 185, 5):
                if t == 40:
                    continue  # a gap the chart must break at
                rows.append({"log_id": lid, "station_id": sid, "miles_upstream": miles, "minutes_from_log": t,
                             "avg_speed": 30.0 if 0 <= t < 30 else 62.0, "expected_speed": 60.0,
                             "typical_speed": 63.0, "is_observed": True, "baseline_method": "precrash_adjusted"})
    df = pd.DataFrame(rows)
    # Athena-style nullable types, with gaps: a reading with no speed and an unknown observed flag
    df["avg_speed"] = df["avg_speed"].astype("Float64")
    df.loc[5, "avg_speed"] = None
    df["is_observed"] = df["is_observed"].astype("boolean")
    df.loc[6, "is_observed"] = None
    df["expected_speed"] = df["expected_speed"].astype("Float64")
    df.loc[7, "expected_speed"] = None
    df["station_id"] = df["station_id"].astype("Int64")
    return df


def test_severity_table_shows_longer_backups_for_severe_incidents():
    labels = add_derived(fake_labels())
    sev = severity_table(labels).set_index(["flag", "has_it"])
    yes, no = sev.loc[("SigAlert issued", "yes")], sev.loc[("SigAlert issued", "no")]
    assert yes["backup_rate"] > no["backup_rate"]
    assert yes["median_minutes"] > no["median_minutes"]
    assert yes["incidents"] + no["incidents"] == len(labels)


def test_sample_has_backups_and_possible_misses():
    labels = add_derived(fake_labels())
    sample = pick_sample(labels, n_backups=30, n_missed=10)
    assert (sample["group"] == "backup").sum() == 30 and sample.loc[sample["group"] == "backup", "has_backup"].all()
    missed = sample[sample["group"] != "backup"]
    assert len(missed) == 10 and not missed["has_backup"].any() and missed["any_blocking"].all()
    assert pick_sample(labels, seed=0)["log_id"].tolist() == pick_sample(labels, seed=0)["log_id"].tolist()


def test_page_builds_one_card_per_incident_and_escapes_text():
    labels = add_derived(fake_labels())
    sample = pick_sample(labels, n_backups=5, n_missed=2)
    # make sure the incidents with missing location / extent / quality are on the page
    gaps = labels[labels["location"].isna() | labels["location_desc"].isna()].head(2).assign(group="backup")
    sample = pd.concat([sample, gaps]).drop_duplicates("log_id").reset_index(drop=True)
    traces = fake_traces(sample["log_id"])
    notes = pd.DataFrame({"log_id": sample["log_id"], "detail_seq": pd.array([1] * len(sample), dtype="Int64"),
                          "first_seen_at": sample["log_time"] + pd.Timedelta(minutes=3),
                          "detail_text": pd.array(["<script>alert(1)</script> BLKG #2"] * (len(sample) - 1) + [None],
                                                  dtype="string")})
    page = build_page(severity_table(labels), sample, traces, notes)
    assert page.count('<section class="card"') == len(sample)
    assert "<script>alert(1)" not in page and "&lt;script&gt;" in page   # notes are escaped
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in page                         # location text too
    assert page.count("<svg") == 2 * len(sample)                           # two charts per incident
    assert 'class="threshold"' in page and 'class="backup"' in page
