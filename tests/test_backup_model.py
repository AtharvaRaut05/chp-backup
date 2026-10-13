import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from backup_model import (  # noqa: E402
    FEATURES, LIVE, VARIANTS, BoostedModel, Rule, add_engineered, km_median, prepare, split_by_day,
)
from conftest import fake_landmarks  # noqa: E402


def test_prepare_converts_nullable_types(data):
    assert data["max_lanes_blocked"].dtype == float and data["max_lanes_blocked"].isna().any()
    assert data["blocking_now"].dtype == float
    assert data["type_code"].dtype == "category"


def test_prepare_rejects_missing_columns():
    with pytest.raises(ValueError, match="site_lanes"):
        prepare(fake_landmarks(50).drop(columns="site_lanes"))


def test_location_columns_are_not_features():
    for v in VARIANTS:
        assert {"freeway", "dispatch_id", "district", "direction"}.isdisjoint(v.features)


def test_answer_columns_are_never_features():
    answers = {"has_backup", "total_backup_minutes", "remaining_minutes", "is_censored", "latest_slot_min",
               "backup_start_min", "backup_end_min", "max_extent_miles", "min_speed_ratio", "slowed_minutes",
               "observed_until_min", "label_adjusted_share", "log_id", "log_time"}
    for v in VARIANTS:
        assert answers.isdisjoint(v.features) and answers.isdisjoint(v.simple_features)


def test_no_live_variant_has_no_live_columns():
    chp_only = [v for v in VARIANTS if "live" not in v.name][0]
    assert set(LIVE).isdisjoint(chp_only.features + chp_only.simple_features + chp_only.rule_keys)


def test_engineered_capacity_features():
    df = pd.DataFrame({
        "day_type": ["weekday", "weekday", "weekend", "weekday"],
        "hour_local": [7.0, 14.0, 7.0, 12.0],
        "blocking_now": [1.0, 1.0, 0.0, 1.0],
        "all_lanes": [0.0, 1.0, 0.0, 0.0],
        "max_lanes_blocked": [3.0, np.nan, np.nan, np.nan],
        "latest_lanes_blocked": [2.0, np.nan, np.nan, np.nan],  # 3 were blocked, one has reopened
        "site_lanes": [4.0, 3.0, 4.0, 4.0],
        "approach_flow_per_sensor": [500.0, 300.0, 500.0, np.nan],  # vehicles per 5 min
        "approach_lanes": [4.0, 3.0, 4.0, np.nan],
        "site_flow": [500.0, 300.0, 500.0, 400.0],
    })
    out = add_engineered(df)
    assert list(out["rush_hour_now"]) == [1, 0, 0, 0]       # weekend 7 AM isn't rush hour
    assert list(out["rush_hour_soon"]) == [0, 1, 0, 0]
    assert list(out["lanes_blocked_now"]) == [2, 3, 0, 1]   # latest count; all lanes -> all 3; unknown -> 1
    assert list(out["lanes_open_share"]) == [0.5, 0.0, 1.0, 0.75]
    assert out["demand_per_lane"].iloc[0] == 1500           # 500 * 12 / 4
    assert out["demand_per_lane"].iloc[3] == 1200           # no approach sensor -> site: 400 * 12 / 4
    # 1500/h/lane x 4 lanes arriving, 2 lanes x 2000/h left -> 1.5x more than can get through
    assert out["demand_vs_capacity"].iloc[0] == pytest.approx(1.5)
    # fully closed: 1200/h/lane x 3 lanes arriving, only a trickle (half a lane) gets through
    assert out["demand_vs_capacity"].iloc[1] == pytest.approx(3.6)
    assert list(out["lanes_group"]) == ["2+", "all", "none", "1"]


def test_zero_lane_sensor_records_dont_produce_infinities():
    df = pd.DataFrame({"day_type": ["weekday"], "hour_local": [8.0], "blocking_now": [1.0], "all_lanes": [0.0],
                       "max_lanes_blocked": [1.0], "latest_lanes_blocked": [1.0], "site_lanes": [0.0],
                       "approach_flow_per_sensor": [400.0], "approach_lanes": [0.0], "site_flow": [400.0]})
    out = add_engineered(df)
    numeric = out.select_dtypes("number")
    assert not np.isinf(numeric.to_numpy()).any()


def test_km_median_counts_still_going_backups_as_lower_bounds():
    # 4 backups: 10 and 20 min ended; two were still going at 15 and 40.
    # Naive median of the numbers = 17.5; Kaplan-Meier: S(10)=3/4, S(20)=3/4*(1-1/2)=3/8 <= 1/2 -> 20
    assert km_median(pd.Series([10, 15, 20, 40]), pd.Series([False, True, False, True])) == 20
    assert km_median(pd.Series([10, 20, 30]), pd.Series([False, False, False])) == 20
    # more than half still going: the longest time is the best lower bound we have
    assert km_median(pd.Series([10, 50, 60]), pd.Series([False, True, True])) == 60


def test_split_is_by_whole_days_and_test_is_latest(data):
    train, test = split_by_day(data)
    assert set(train["log_id"]).isdisjoint(test["log_id"])
    assert train["incident_day"].max() < test["incident_day"].min()
    assert 0.1 < test["log_id"].nunique() / data["log_id"].nunique() <= 0.26


def test_split_edge_cases(data):
    days = sorted(data["incident_day"].unique())
    train, test = split_by_day(data[data["incident_day"].isin(days[:2])])
    assert train["incident_day"].nunique() == 1 and test["incident_day"].nunique() == 1
    with pytest.raises(ValueError, match="at least 2 days"):
        split_by_day(data[data["incident_day"] == days[0]])


def test_rule_falls_back_to_coarser_groups(data):
    rule = Rule(["landmark_min", "backed_up_now", "lanes_group", "type_code"]).fit(data)
    unseen = data.head(1).assign(type_code="9999", lanes_group="never seen")
    pred = rule.predict(unseen)
    same = (data["landmark_min"] == unseen["landmark_min"].iloc[0]) & (data["backed_up_now"] == unseen["backed_up_now"].iloc[0])
    assert pred["p_more_backup"].iloc[0] == pytest.approx((data.loc[same, "remaining_minutes"] > 0).mean())
    pos = data[same & (data["remaining_minutes"] > 0)]
    assert pred["minutes_if_more"].iloc[0] == km_median(pos["remaining_minutes"], pos["is_censored"] == 1)


@pytest.fixture(scope="module")
def boosted(data):
    train, test = split_by_day(data)
    return train, test, BoostedModel(FEATURES).fit(train)


def test_boosted_predicts_probabilities_and_minutes(boosted):
    _, test, model = boosted
    pred = model.predict(test)
    assert pred["p_more_backup"].between(0, 1).all()
    assert (pred["minutes_if_more"] > 0).all()


def test_unseen_category_becomes_missing_not_an_error(boosted):
    _, test, model = boosted
    odd = test.head(3).copy()
    odd["type_code"] = pd.Categorical(["NEW1", "NEW2", "NEW3"])
    assert len(model.predict(odd)) == 3


def test_too_little_data_is_an_error_not_a_bad_model():
    tiny = prepare(fake_landmarks(n_incidents=40))
    with pytest.raises(ValueError, match="Collect more days"):
        BoostedModel(FEATURES).fit(tiny)
