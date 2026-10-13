"""Predict how many more minutes a freeway backup will last, from fct_training_landmarks.

Two-part model, made at 10, 20 or 30 minutes after CHP logs an incident:
  1. Will there be any more backup?      gradient-boosted classifier
  2. If so, how many more minutes?       gradient-boosted survival model (XGBoost AFT), which
                                         learns from censored rows ("at least N minutes")
  prediction = 0 if P(more backup) < 0.5, otherwise the survival model's median.

Compared against two baselines:
  always 0     "no more backup" (right ~80% of the time, since most incidents never back up)
  rule         the typical remaining time for past incidents at the same landmark with the same
               lanes blocked and incident type (and, when live traffic is used, whether traffic
               was backed up right then)

Two feature sets:
  with live traffic   includes sensor readings since the crash. Realistic only with real-time
                      sensor data; the daily PeMS files we collect arrive a day late.
  without             CHP's notes and units plus traffic before the crash: what you'd know from
                      the CHP feed alone.

The test set is the most recent days, so the model is always scored on incidents that happened
after everything it learned from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

CATEGORICAL = ["dispatch_id", "district", "freeway", "direction", "day_type", "type_code"]
LIVE = ["queue_ratio_now", "backed_up_now", "extent_now_miles", "min_ratio_so_far",
        "slowed_minutes_so_far", "max_extent_so_far"]
# Explicit list: a column added to the mart later never becomes a feature by accident.
FEATURES = [
    "landmark_min", *CATEGORICAL, "hour_local", "day_of_week",
    # what CHP knew
    "n_notes", "ever_blocking", "blocking_now", "max_lanes_blocked", "all_lanes", "lanes_reopened",
    "vehicles", "ambulance", "tow", "fsp", "fire", "sigalert", "overturned", "hazmat", "fatality",
    "units_assigned", "units_at_scene", "first_at_scene_min", "chp_closed",
    # traffic before the crash
    "site_speed", "site_occupancy", "site_flow", "site_ratio_to_typical", "site_speed_trend",
    "queue_speed", "queue_min_speed", "queue_occupancy", "queue_speed_trend", "queue_sensors",
    "approach_speed", "approach_flow_per_sensor", "approach_flow_trend",
    "merge_flow", "connector_merge_flow", "exit_flow", "merge_points",
    # traffic since the crash
    *LIVE,
]
VARIANTS = {
    "with live traffic": (FEATURES, ["landmark_min", "backed_up_now", "lanes_group", "type_code"]),
    "CHP + pre-crash only": ([f for f in FEATURES if f not in LIVE], ["landmark_min", "lanes_group", "type_code"]),
}
MIN_POSITIVE = 10  # backed-up rows needed in each split to fit and score anything meaningful


# ---------- data ----------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Typed copy with the incident's Pacific date and a lanes-blocked-now bucket for the baseline."""
    missing = set(FEATURES + ["log_id", "log_time", "remaining_minutes", "is_censored"]) - set(df.columns)
    if missing:
        raise ValueError(f"training table is missing columns: {sorted(missing)}")
    df = df.copy()
    df["log_time"] = pd.to_datetime(df["log_time"], utc=True)
    df["incident_day"] = df["log_time"].dt.tz_convert("America/Los_Angeles").dt.date
    for c in CATEGORICAL:
        df[c] = df[c].astype("string").astype("category")
    for c in df.columns.difference([*CATEGORICAL, "log_id", "log_time", "incident_day"]):
        df[c] = pd.to_numeric(df[c].astype("object"), errors="coerce").astype(float)  # bool/Int64 -> float, NA -> nan
    blocked = df["blocking_now"] == 1
    df["lanes_group"] = np.select(
        [blocked & (df["all_lanes"] == 1), blocked & (df["max_lanes_blocked"] >= 2), blocked],
        ["all", "2+", "1"],
        default="none",
    )
    return df


def split_by_day(df: pd.DataFrame, test_share: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The most recent whole days, up to test_share of incidents (at least one day), are the test set."""
    per_day = df.groupby("incident_day")["log_id"].nunique().sort_index(ascending=False)
    if len(per_day) < 2:
        raise ValueError(f"need at least 2 days of incidents to split, have {len(per_day)}")
    share = per_day.cumsum() / per_day.sum()
    n_test = max(1, int((share <= test_share + 1e-9).sum()))
    n_test = min(n_test, len(per_day) - 1)
    is_test = df["incident_day"].isin(per_day.index[:n_test])
    return df[~is_test], df[is_test]


def _has_enough(part: pd.DataFrame, name: str) -> None:
    more = part["remaining_minutes"] > 0
    if more.sum() < MIN_POSITIVE or (~more).sum() < MIN_POSITIVE:
        raise ValueError(f"{name} has {more.sum()} backed-up and {(~more).sum()} clear rows; "
                         f"need {MIN_POSITIVE} of each. Collect more days of data.")


# ---------- baseline ----------

def fit_baseline(train: pd.DataFrame, keys: list[str]) -> dict:
    """Median remaining minutes per group of `keys`, falling back to coarser groups."""
    levels = [keys[:k] for k in range(len(keys), 0, -1)]
    tables = [train.groupby(lv, observed=True)["remaining_minutes"].agg(["median", "size"]) for lv in levels]
    return {"levels": levels, "tables": tables, "overall": float(train["remaining_minutes"].median())}


def predict_baseline(model: dict, df: pd.DataFrame, min_group: int = 10) -> np.ndarray:
    pred = pd.Series(np.nan, index=df.index)
    for keys, table in zip(model["levels"], model["tables"]):
        table = table[table["size"] >= min_group]
        looked_up = df[keys].merge(table, left_on=keys, right_index=True, how="left")["median"]
        pred = pred.fillna(looked_up)
    return pred.fillna(model["overall"]).to_numpy()


# ---------- model ----------

@dataclass
class BackupModel:
    classifier: xgb.XGBClassifier
    survival: xgb.Booster
    columns: list[str]
    categories: dict[str, list]

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model inputs; categories never seen in training become missing instead of an error."""
        x = df[self.columns].copy()
        for c, levels in self.categories.items():
            x[c] = pd.Categorical(x[c].astype("object"), categories=levels)
        return x

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self.features(df)
        p_more = self.classifier.predict_proba(x)[:, 1]
        median_if_more = self.survival.predict(xgb.DMatrix(x, enable_categorical=True))
        return pd.DataFrame(
            {"p_more_backup": p_more, "minutes_if_more": median_if_more,
             "predicted_minutes": np.where(p_more >= 0.5, median_if_more, 0.0)},
            index=df.index,
        )


CLASSIFIER = dict(learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8,
                  colsample_bytree=0.8, enable_categorical=True, tree_method="hist")
SURVIVAL = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0, "learning_rate": 0.05, "max_depth": 3,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist"}


def fit_model(train: pd.DataFrame, columns: list[str] = FEATURES, seed: int = 0) -> BackupModel:
    """Pick the number of trees on the latest training days, then refit on all training days."""
    cats = {c: list(train[c].cat.categories) for c in columns if c in CATEGORICAL}
    shell = BackupModel(None, None, list(columns), cats)  # type: ignore[arg-type]
    fit, valid = split_by_day(train, test_share=0.2)
    for name, part in (("training data", fit), ("validation data", valid)):
        _has_enough(part, name)
    more = lambda part: part["remaining_minutes"] > 0  # noqa: E731

    trial = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=50, eval_metric="logloss",
                              random_state=seed, **CLASSIFIER)
    trial.fit(shell.features(fit), more(fit), eval_set=[(shell.features(valid), more(valid))], verbose=False)
    classifier = xgb.XGBClassifier(n_estimators=trial.best_iteration + 1, random_state=seed, **CLASSIFIER)
    classifier.fit(shell.features(train), more(train))

    def survival_matrix(part: pd.DataFrame) -> xgb.DMatrix:
        part = part[more(part)]
        m = xgb.DMatrix(shell.features(part), enable_categorical=True)
        lower = part["remaining_minutes"].to_numpy(float)
        m.set_float_info("label_lower_bound", lower)
        m.set_float_info("label_upper_bound", np.where(part["is_censored"] == 1, np.inf, lower))
        return m

    params = {**SURVIVAL, "seed": seed}
    trial_s = xgb.train(params, survival_matrix(fit), num_boost_round=500,
                        evals=[(survival_matrix(valid), "valid")], early_stopping_rounds=50, verbose_eval=False)
    survival = xgb.train(params, survival_matrix(train), num_boost_round=trial_s.best_iteration + 1)
    return BackupModel(classifier, survival, list(columns), cats)


# ---------- scoring ----------

def concordance(time: np.ndarray, event: np.ndarray, predicted: np.ndarray) -> float:
    """Harrell's C: of pairs where we know which backup cleared first, how often the model ranked
    them right (0.5 = guessing). A censored backup is only known to outlast events before it;
    at a tied time, the one that cleared counts as shorter than the one still going."""
    time, event, predicted = (np.asarray(a, dtype=float) for a in (time, event, predicted))
    ti, tj = time[:, None], time[None, :]
    ei, ej = event[:, None] == 1, event[None, :] == 1
    comparable = ei & ((ti < tj) | ((ti == tj) & ~ej))
    right = (predicted[:, None] < predicted[None, :]) + 0.5 * (predicted[:, None] == predicted[None, :])
    return float((right * comparable).sum() / comparable.sum()) if comparable.any() else float("nan")


def evaluate(test: pd.DataFrame, baseline_pred: np.ndarray, model_pred: pd.DataFrame) -> pd.DataFrame:
    """Scores per landmark and overall. MAE uses rows whose true remaining time is known (not
    censored), so it slightly favours short predictions."""
    df = test.assign(baseline=baseline_pred, model=model_pred["predicted_minutes"],
                     p_more=model_pred["p_more_backup"], if_more=model_pred["minutes_if_more"])
    rows = []
    for landmark, g in [*df.groupby("landmark_min"), ("all", df)]:
        known = g[g["is_censored"] == 0]
        more = g[g["remaining_minutes"] > 0]
        has_both = (g["remaining_minutes"] > 0).nunique() == 2
        rows.append({
            "landmark_min": landmark if landmark == "all" else f"{landmark:.0f}",
            "rows": len(g),
            "share_more_backup": round((g["remaining_minutes"] > 0).mean(), 3),
            "mae_always_0": round(known["remaining_minutes"].mean(), 1),
            "mae_rule": round((known["baseline"] - known["remaining_minutes"]).abs().mean(), 1),
            "mae_model": round((known["model"] - known["remaining_minutes"]).abs().mean(), 1),
            "auc_more_backup": round(roc_auc_score(g["remaining_minutes"] > 0, g["p_more"]), 3) if has_both else np.nan,
            "c_index_duration": round(concordance(more["remaining_minutes"], 1 - more["is_censored"], more["if_more"]), 3),
        })
    return pd.DataFrame(rows)


def gain_interval(test: pd.DataFrame, baseline_pred: np.ndarray, model_pred: pd.DataFrame,
                  n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """How many minutes of error the model saves vs the rule (MAE_rule - MAE_model), with a 95%
    interval from resampling whole incidents. If the interval includes 0, the difference could be luck."""
    df = test.assign(err_rule=np.abs(baseline_pred - test["remaining_minutes"]),
                     err_model=np.abs(model_pred["predicted_minutes"] - test["remaining_minutes"]))
    df = df[df["is_censored"] == 0]
    per = df.groupby("log_id")[["err_rule", "err_model"]].agg(["sum", "size"])
    gain_sum = (per[("err_rule", "sum")] - per[("err_model", "sum")]).to_numpy()
    size = per[("err_rule", "size")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n, len(per)))
    boots = gain_sum[idx].sum(axis=1) / size[idx].sum(axis=1)
    return float(gain_sum.sum() / size.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def importance(model: BackupModel, top: int = 15) -> pd.DataFrame:
    gain_c = model.classifier.get_booster().get_score(importance_type="total_gain")
    gain_s = model.survival.get_score(importance_type="total_gain")
    out = pd.DataFrame({"will_back_up": pd.Series(gain_c, dtype=float),
                        "how_long": pd.Series(gain_s, dtype=float)}).fillna(0)
    out = out / out.sum().replace(0, 1)
    return out.sort_values("will_back_up", ascending=False).head(top).round(3)
