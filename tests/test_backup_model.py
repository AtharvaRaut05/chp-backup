import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from backup_model import (  # noqa: E402
    FEATURES, LIVE, concordance, evaluate, fit_baseline, fit_model, gain_interval, predict_baseline, prepare,
    split_by_day,
)


def fake_landmarks(n_incidents=1500, days=9, seed=0) -> pd.DataFrame:
    """Same columns and Athena-style nullable dtypes as fct_training_landmarks.

    Blocked lanes make backups likelier and longer. The notes and "live" traffic columns are noisy
    views of whether traffic is backed up, so no single column gives the answer away.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_incidents):
        log_time = pd.Timestamp("2026-10-01 07:00") + pd.Timedelta(minutes=int(rng.integers(0, days * 1440)))
        blocking = rng.random() < 0.4
        lanes = int(rng.integers(1, 4)) if blocking else None
        has_backup = rng.random() < (0.45 if blocking else 0.1)
        total = float((20 + 15 * (lanes or 0)) * rng.lognormal(0, 0.5)) // 5 * 5 + 5 if has_backup else 0.0
        censored = has_backup and total > 150
        for lm in (10, 20, 30):
            remaining = max(total - lm, 0.0)
            looks_backed_up = (remaining > 0) != (rng.random() < 0.15)
            row = {c: None for c in FEATURES}
            row.update({
                "log_id": f"i{i}", "landmark_min": lm, "log_time": log_time,
                "dispatch_id": rng.choice(["LACC", "OCCC", "INCC"]), "district": "07", "freeway": "405",
                "direction": rng.choice(["N", "S"]), "hour_local": log_time.hour, "day_of_week": 3,
                "day_type": "weekday", "type_code": rng.choice(["1182", "1183"]), "n_notes": int(rng.integers(0, 6)),
                "ever_blocking": blocking, "blocking_now": blocking and ((remaining > 0) != (rng.random() < 0.2)),
                "max_lanes_blocked": lanes,
                "all_lanes": False, "vehicles": int(rng.integers(1, 4)), "tow": blocking,
                "units_assigned": 1, "chp_closed": False, "site_speed": float(rng.normal(60, 8)),
                "queue_ratio_now": float(rng.normal(0.45 if looks_backed_up else 0.95, 0.1)),
                "backed_up_now": looks_backed_up,
                "slowed_minutes_so_far": float(min(total, lm - 5) * rng.uniform(0.3, 1.0) + 5 * (rng.random() < 0.2)),
                "has_backup": has_backup, "total_backup_minutes": total,
                "remaining_minutes": remaining, "is_censored": censored and remaining > 0,
            })
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ["max_lanes_blocked", "n_notes", "units_assigned", "vehicles", "landmark_min"]:
        df[c] = df[c].astype("Int64")
    for c in ["ever_blocking", "blocking_now", "all_lanes", "tow", "chp_closed", "backed_up_now",
              "is_censored", "has_backup", "lanes_reopened", "ambulance"]:
        df[c] = df[c].astype("boolean")
    for c in ["dispatch_id", "district", "freeway", "direction", "day_type", "type_code"]:
        df[c] = df[c].astype("string")
    return df


@pytest.fixture(scope="module")
def data():
    return prepare(fake_landmarks())


def test_prepare_converts_nullable_types(data):
    assert data["max_lanes_blocked"].dtype == float and data["max_lanes_blocked"].isna().any()
    assert data["blocking_now"].dtype == float
    assert data["type_code"].dtype == "category"


def test_prepare_rejects_missing_columns():
    with pytest.raises(ValueError, match="site_speed"):
        prepare(fake_landmarks(50).drop(columns="site_speed"))


def test_split_is_by_whole_days_and_test_is_latest(data):
    train, test = split_by_day(data)
    assert set(train["log_id"]).isdisjoint(test["log_id"])
    assert train["incident_day"].max() < test["incident_day"].min()
    share = test["log_id"].nunique() / data["log_id"].nunique()
    assert 0.1 < share <= 0.26


def test_split_edge_cases(data):
    days = sorted(data["incident_day"].unique())
    two = data[data["incident_day"].isin(days[:2])]
    train, test = split_by_day(two)
    assert train["incident_day"].nunique() == 1 and test["incident_day"].nunique() == 1
    with pytest.raises(ValueError, match="at least 2 days"):
        split_by_day(data[data["incident_day"] == days[0]])


def test_baseline_falls_back_to_coarser_groups(data):
    keys = ["landmark_min", "backed_up_now", "lanes_group", "type_code"]
    model = fit_baseline(data, keys)
    unseen = data.head(1).assign(type_code="9999", lanes_group="all")
    pred = predict_baseline(model, unseen)
    same = (data["landmark_min"] == unseen["landmark_min"].iloc[0]) & (data["backed_up_now"] == unseen["backed_up_now"].iloc[0])
    assert pred[0] == data.loc[same, "remaining_minutes"].median()


def test_concordance():
    t, e = np.array([1, 2, 3]), np.array([1, 1, 1])
    assert concordance(t, e, np.array([1, 2, 3])) == 1.0
    assert concordance(t, e, np.array([3, 2, 1])) == 0.0
    # censored at 1: we only know it outlasted nothing -> pairs (2,3) only, ranked right
    assert concordance(np.array([1, 2, 3]), np.array([0, 1, 1]), np.array([9, 2, 3])) == 1.0
    # tie: an event and a censored row at the same time; the event counts as shorter
    assert concordance(np.array([5, 5]), np.array([1, 0]), np.array([1, 2])) == 1.0
    assert concordance(np.array([5, 5]), np.array([1, 0]), np.array([2, 1])) == 0.0


@pytest.fixture(scope="module")
def fitted(data):
    train, test = split_by_day(data)
    return train, test, fit_model(train)


def test_model_learns_from_noisy_signal(fitted):
    train, test, model = fitted
    pred = model.predict(test)
    assert (pred["minutes_if_more"] > 0).all()
    keys = ["landmark_min", "backed_up_now", "lanes_group", "type_code"]
    metrics = evaluate(test, predict_baseline(fit_baseline(train, keys), test), pred).set_index("landmark_min")
    assert metrics.loc["all", "auc_more_backup"] > 0.75
    assert metrics.loc["all", "c_index_duration"] > 0.6
    assert metrics.loc["all", "mae_model"] < metrics.loc["all", "mae_always_0"]


def test_answer_columns_are_never_features():
    answers = {"has_backup", "total_backup_minutes", "remaining_minutes", "is_censored", "latest_slot_min",
               "backup_start_min", "backup_end_min", "max_extent_miles", "min_speed_ratio", "slowed_minutes",
               "observed_until_min", "log_id", "log_time"}
    assert answers.isdisjoint(FEATURES)


def test_unseen_category_becomes_missing_not_an_error(fitted):
    _, test, model = fitted
    odd = test.head(3).copy()
    odd["type_code"] = pd.Categorical(["NEW1", "NEW2", "NEW3"])
    assert len(model.predict(odd)) == 3


def test_without_live_traffic_uses_no_live_columns(data):
    train, _ = split_by_day(data)
    columns = [c for c in FEATURES if c not in LIVE]
    model = fit_model(train, columns)
    used = set(model.classifier.get_booster().feature_names)
    assert used.isdisjoint(LIVE)


def test_gain_interval_contains_the_point_estimate(fitted):
    train, test, model = fitted
    keys = ["landmark_min", "backed_up_now", "lanes_group", "type_code"]
    gain, lo, hi = gain_interval(test, predict_baseline(fit_baseline(train, keys), test), model.predict(test))
    assert lo <= gain <= hi


def test_too_little_data_is_an_error_not_a_bad_model():
    tiny = prepare(fake_landmarks(n_incidents=40))
    with pytest.raises(ValueError, match="Collect more days"):
        fit_model(tiny)
