import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from backup_model import VARIANTS, drop_duplicate_reports, more_backup, prepare  # noqa: E402
from conftest import fake_landmarks  # noqa: E402
import forward_test as ft  # noqa: E402


@pytest.fixture(scope="module")
def long_data():
    return prepare(fake_landmarks(n_incidents=2600, days=24, seed=3))


def test_chronological_split_never_trains_on_later_days(long_data):
    train, test = ft.chronological_split(long_data, test_days=7)
    assert train["incident_day"].max() < test["incident_day"].min()
    assert test["incident_day"].nunique() == 7
    with pytest.raises(ValueError):
        ft.chronological_split(long_data[long_data["incident_day"] < sorted(long_data["incident_day"].unique())[10]], 7)


def test_alert_stats_counts():
    more = pd.Series([True, True, True, False, False, False])
    p = pd.Series([0.9, 0.6, 0.1, 0.7, 0.2, 0.1])
    s = ft.alert_stats(more, p, 0.5, n_days=2)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["alerts_per_day"] == 1.5 and s["missed_per_day"] == 0.5 and s["false_alarms_per_day"] == 0.5


def test_recall_threshold_catches_the_target_share():
    rng = np.random.default_rng(0)
    p = pd.Series(rng.random(1000))
    more = pd.Series(rng.random(1000) < p)
    t = ft.recall_threshold(more, p, 0.8)
    assert ((p > t) & more).sum() / more.sum() >= 0.8
    assert ((p > t + 0.02) & more).sum() / more.sum() < 0.8  # and is the highest such threshold, near enough


def test_cost_thresholds_follow_the_formula(long_data):
    train, _ = ft.chronological_split(long_data, 7)
    th = ft.thresholds_from_validation(train, VARIANTS[1])
    assert th["miss = 4x false alarm"] == pytest.approx(0.2)
    assert th["miss = 1x false alarm"] == pytest.approx(0.5)
    assert 0 < th[f"catch {ft.TARGET_RECALL:.0%} (set on validation days)"] < 1


def test_retrain_points_are_mondays_with_enough_history(long_data):
    days = sorted(long_data["incident_day"].unique())
    points = ft.retrain_points(long_data)
    assert points and all(pd.Timestamp(d).dayofweek == 0 for d in points)
    assert all(sum(x < d for x in days) >= ft.MIN_RETRAIN_DAYS for d in points)


def test_retraining_study_only_uses_past_data_and_compares_same_rows(long_data):
    table, pred = ft.retraining_study(long_data, VARIANTS[1], n_boot=20, log=lambda *_: None)
    days = sorted(long_data["incident_day"].unique())
    for week, g in pred.groupby("week_of"):
        assert g["incident_day"].min() >= week
        assert g["training_days_retrained"].iloc[0] == sum(d < week for d in days)
        assert g["training_days_frozen"].iloc[0] < g["training_days_retrained"].iloc[0]
    assert set(table.index.get_level_values("model")) == {"frozen", "retrained", "retrained - frozen"}


def test_drop_duplicate_reports_keeps_one_report_per_crash():
    t = pd.Timestamp("2026-10-01 08:00", tz="UTC")
    rows = [
        ("261001LAFSP0001", t - pd.Timedelta("3min"), 55.0, 300.0),       # FSP logged it first...
        ("261001LA0001", t, 55.0, 300.0),                                   # ...CHP's report is kept
        ("261001LA0002", t + pd.Timedelta("1min"), 55.0, 300.0),          # CHP's own second log
        ("261001LAFSP0002", t + pd.Timedelta("12min"), 61.0, 280.0),       # different spot
        ("261001LAFSP0003", t + pd.Timedelta("2h"), 55.0, 300.0),          # same reading, hours later
    ]
    df = pd.DataFrame([{"log_id": i, "log_time": lt, "site_speed": s, "site_flow": f, "freeway": "405",
                        "direction": "N", "landmark_min": lm} for i, lt, s, f in rows for lm in (5, 10)])
    kept = drop_duplicate_reports(df)
    assert set(kept["log_id"]) == {"261001LA0001", "261001LAFSP0002", "261001LAFSP0003"}
    assert len(kept) == 6  # every checkpoint row of a kept incident stays


def test_run_produces_every_table(long_data, tmp_path):
    res = ft.run(long_data, test_days=7, n_boot=20, out=tmp_path, log=lambda *_: None)
    r = res[VARIANTS[1].name]
    for k in ("forward_scores", "alert_tradeoff", "precision_at_recall", "failure_profile", "learning_curve",
              "retraining"):
        assert isinstance(r[k], pd.DataFrame) and len(r[k])
    assert {"precision", "recall", "auc", "avg_precision"} <= set(r["forward_scores"].columns)
    # more alerts as a missed backup is weighted more heavily
    a = r["alert_tradeoff"]["alerts_per_day"]
    assert a["miss = 1x false alarm"] <= a["miss = 4x false alarm"] <= a["miss = 9x false alarm"]
    assert any(p.name.startswith("forward__alert_tradeoff") for p in tmp_path.iterdir())
    pred = r["forward_predictions"]
    assert more_backup(pred).any() and pred["incident_day"].nunique() == 7


# ---------- promotion ----------

import promote  # noqa: E402


def test_first_model_is_promoted_and_saved(long_data, tmp_path):
    row = promote.decide(long_data, "CHP + history only", tmp_path / "m")
    assert row["decision"] == "promote" and row["reason"] == "no current model"
    model, meta = promote.load(tmp_path / "m")
    assert meta["trained_through"] == str(max(long_data["incident_day"]))
    p = model.predict(long_data.head(50))["p_more_backup"]
    assert p.between(0, 1).all()


def test_champion_kept_until_it_has_unseen_days(long_data, tmp_path):
    days = sorted(long_data["incident_day"].unique())
    promote.decide(long_data[long_data["incident_day"] <= days[-3]], "CHP + history only", tmp_path / "m")
    row = promote.decide(long_data, "CHP + history only", tmp_path / "m")
    assert row["decision"] == "keep" and "only 2 new days" in row["reason"]


def test_challenger_only_replaces_a_better_champion(long_data, tmp_path, monkeypatch):
    days = sorted(long_data["incident_day"].unique())
    old = long_data[long_data["incident_day"] <= days[-8]]
    promote.decide(old, "CHP + history only", tmp_path / "m")
    real = promote.BoostedModel

    class Coin(real):  # a challenger that guesses
        def predict(self, df):
            out = super().predict(df)
            out["p_more_backup"] = np.random.default_rng(0).random(len(df))
            return out

    monkeypatch.setattr(promote, "challenger", lambda features, seed=0: Coin(features, seed))
    row = promote.decide(long_data, "CHP + history only", tmp_path / "m")
    assert row["decision"] == "keep" and row["window"].startswith(str(days[-7]))
    assert promote.load(tmp_path / "m")[1]["trained_through"] == str(days[-8])  # champion untouched

    monkeypatch.setattr(promote, "challenger", lambda features, seed=0: real(features, seed))
    row = promote.decide(long_data, "CHP + history only", tmp_path / "m")
    assert row["decision"] == "promote" and "difference" in row
    assert promote.load(tmp_path / "m")[1]["trained_through"] == str(days[-1])
