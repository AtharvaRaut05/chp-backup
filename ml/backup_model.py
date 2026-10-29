"""Data preparation, features, the rule baseline and the boosted model for backup prediction.

Two questions, asked at 5, 10, 20 or 30 minutes after CHP logs an incident:
  1. Will there be any more backup?        (classification)
  2. If so, how many more minutes?          (survival: learns from "still going at the end" rows)

Three ways to answer them, compared in evaluate.py:
  rule      typical answer for past incidents in the same group (checkpoint, lanes blocked, type)
  boosted   XGBoost classifier + XGBoost survival model (AFT) on all features
  simple    logistic regression + log-normal survival model on ~15 features (simple_models.py)

Two feature sets:
  with live traffic     includes sensor readings since the crash. Realistic only with real-time
                        sensor data; the daily PeMS files arrive a day late.
  CHP + pre-crash only  CHP's notes and units plus traffic before the crash.

Normal congestion at the spot (how slow it usually is now and in the next hour, and how long
until it usually flows freely) tells the models whether a crash lands in the middle of rush hour
or as it is ending.

Location columns (freeway, dispatch center, district, direction) are deliberately NOT features:
with a few weeks of data they let a model memorize which corridors had long backups instead of
learning why. The road's lanes, traffic per lane and capacity left after the crash stand in for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

CAPACITY_PER_LANE = 2000  # vehicles/hour a freeway lane can carry (Highway Capacity Manual, rounded)

CATEGORICAL = ["type_code"]
LIVE = ["queue_ratio_now", "backed_up_now", "extent_now_miles", "min_ratio_so_far",
        "slowed_minutes_so_far", "max_extent_so_far"]
ENGINEERED = ["is_weekend", "rush_hour_now", "rush_hour_soon", "lanes_blocked_now", "lanes_open_share",
              "demand_per_lane", "demand_vs_capacity"]
# Explicit list: a column added to the mart later never becomes a feature by accident.
FEATURES = [
    "landmark_min", *CATEGORICAL, "hour_local", "day_of_week",
    # what CHP knew
    "n_notes", "ever_blocking", "blocking_now", "max_lanes_blocked", "latest_lanes_blocked", "all_lanes",
    "lanes_reopened",
    "vehicles", "ambulance", "tow", "fsp", "fire", "sigalert", "overturned", "hazmat", "fatality",
    "units_assigned", "units_at_scene", "first_at_scene_min", "chp_closed",
    # the road and traffic before the crash
    "site_speed", "site_occupancy", "site_flow", "site_ratio_to_typical", "site_speed_trend", "site_lanes",
    "queue_speed", "queue_min_speed", "queue_occupancy", "queue_speed_trend", "queue_sensors",
    "approach_speed", "approach_flow_per_sensor", "approach_flow_trend", "approach_lanes",
    "merge_flow", "connector_merge_flow", "exit_flow", "merge_points",
    # normal congestion at this spot, from other days' history (int_incident_recurring)
    "typical_ratio_now", "typical_ratio_in_30", "typical_ratio_in_60", "typical_congestion_minutes_left",
    # ... and along the whole backup and approach zones (int_incident_zone_recurring)
    "zone_typical_min_now", "zone_typical_min_in_30", "zone_typical_min_in_60", "zone_typical_mean_now",
    "zone_share_congested_now", "approach_typical_mean_now",
    # built from the above (prepare)
    *ENGINEERED,
    # traffic since the crash
    *LIVE,
]
SIMPLE_FEATURES = [
    "landmark_min", "blocking_now", "lanes_blocked_now", "all_lanes", "lanes_open_share", "vehicles",
    "overturned", "sigalert", "ambulance", "rush_hour_now", "rush_hour_soon", "is_weekend",
    "demand_vs_capacity", "site_ratio_to_typical", "queue_speed_trend",
    "typical_ratio_now", "typical_ratio_in_60", "typical_congestion_minutes_left",
]
SIMPLE_LIVE = ["queue_ratio_now", "extent_now_miles", "slowed_minutes_so_far"]


@dataclass(frozen=True)
class Variant:
    name: str
    features: list[str]
    simple_features: list[str]
    rule_keys: list[str]


# Same-day sensor readings (before or after the crash). The pipeline gets them a day late, so a model
# that uses them needs a real-time sensor feed to run live. Everything else is known at the checkpoint:
# CHP notes from the per-minute poller, the road from sensor metadata, normal congestion from past days.
SAME_DAY_SENSOR = [
    "site_speed", "site_occupancy", "site_flow", "site_ratio_to_typical", "site_speed_trend",
    "queue_speed", "queue_min_speed", "queue_occupancy", "queue_speed_trend",
    "approach_speed", "approach_flow_per_sensor", "approach_flow_trend",
    "merge_flow", "connector_merge_flow", "exit_flow", "demand_per_lane", "demand_vs_capacity", *LIVE]

VARIANTS = [
    Variant("with live traffic", FEATURES, SIMPLE_FEATURES + SIMPLE_LIVE,
            ["landmark_min", "backed_up_now", "lanes_group", "type_code"]),
    Variant("CHP + pre-crash only", [f for f in FEATURES if f not in LIVE], SIMPLE_FEATURES,
            ["landmark_min", "lanes_group", "type_code"]),
    Variant("CHP + history only", [f for f in FEATURES if f not in SAME_DAY_SENSOR],
            [f for f in SIMPLE_FEATURES if f not in SAME_DAY_SENSOR],
            ["landmark_min", "lanes_group", "type_code"]),
]
# Newer columns in the training table. Older cached tables don't have them; prepare() fills them in
# as missing. ZONE is in FEATURES (it helped on the development days); WEATHER is collected but not
# used yet: with few rainy days it didn't help (see the README, "Improving the models").
ZONE = ["zone_typical_min_now", "zone_typical_min_in_30", "zone_typical_min_in_60", "zone_typical_mean_now",
        "zone_share_congested_now", "approach_typical_mean_now"]
WEATHER = ["rain_last_hour_mm", "rain_last_3h_mm", "rain_last_24h_mm"]
OPTIONAL = [*ZONE, *WEATHER]
MIN_POSITIVE = 10  # backed-up rows needed in each split to fit and score anything meaningful
RAW_COLUMNS = [c for c in FEATURES if c not in ENGINEERED] + [
    "log_id", "log_time", "day_type", "remaining_minutes", "is_censored"]


# ---------- data ----------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Typed copy with the incident's Pacific date, engineered features and a lanes bucket for the rule."""
    missing = set(RAW_COLUMNS) - set(df.columns) - set(OPTIONAL)
    if missing:
        raise ValueError(f"training table is missing columns: {sorted(missing)}")
    df = df.copy()
    df["log_time"] = pd.to_datetime(df["log_time"], utc=True)
    df["incident_day"] = df["log_time"].dt.tz_convert("America/Los_Angeles").dt.date
    keep_text = {*CATEGORICAL, "log_id", "log_time", "incident_day", "day_type",
                 "dispatch_id", "district", "freeway", "direction"}
    for c in CATEGORICAL:
        df[c] = df[c].astype("string").astype("category")
    for c in df.columns.difference(list(keep_text)):
        df[c] = pd.to_numeric(df[c].astype("object"), errors="coerce").astype(float)  # bool/Int64 -> float, NA -> nan
    for c in ["label_adjusted_share", *OPTIONAL]:
        if c not in df:
            df[c] = np.nan
    return add_engineered(df)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Features from how jams form: a backup grows when more cars arrive than can get past the crash."""
    weekday = df["day_type"].astype("string") == "weekday"
    hour = df["hour_local"]
    df["is_weekend"] = (~weekday).astype(float)
    df["rush_hour_now"] = (weekday & (hour.between(6, 9) | hour.between(15, 18))).astype(float)
    df["rush_hour_soon"] = (weekday & hour.isin([5, 14])).astype(float)

    blocked = df["blocking_now"] == 1
    lanes = df["site_lanes"].where(df["site_lanes"] > 0)  # a 0-lane sensor record is a metadata error
    # Lanes blocked right now: the latest note's count (lanes reopen one at a time during clearance);
    # "all lanes" when that's all we know; 1 when a note says blocked but gives no count.
    count = df["latest_lanes_blocked"].where(df["latest_lanes_blocked"].notna(),
                                             np.where(df["all_lanes"] == 1, lanes, 1.0))
    df["lanes_blocked_now"] = np.where(blocked, np.fmin(count, lanes.fillna(np.inf)), 0.0)
    df["lanes_open_share"] = ((lanes - df["lanes_blocked_now"]) / lanes).clip(0, 1)

    approach_lanes = df["approach_lanes"].where(df["approach_lanes"] > 0)
    per_lane_approach = df["approach_flow_per_sensor"] * 12 / approach_lanes  # vehicles/hour/lane
    per_lane_site = df["site_flow"] * 12 / lanes
    df["demand_per_lane"] = per_lane_approach.fillna(per_lane_site)
    open_lanes = (lanes - df["lanes_blocked_now"]).clip(lower=0.5)  # a fully closed road still leaks a little
    df["demand_vs_capacity"] = (df["demand_per_lane"] * lanes / (CAPACITY_PER_LANE * open_lanes)).clip(upper=10)

    df["lanes_group"] = np.select(
        [blocked & (df["lanes_open_share"] == 0), blocked & (df["lanes_blocked_now"] >= 2), blocked],
        ["all", "2+", "1"], default="none")
    return df


def drop_duplicate_reports(df: pd.DataFrame, window: str = "30min") -> pd.DataFrame:
    """Keep one report per crash.

    The same crash is sometimes logged more than once: Freeway Service Patrol tow trucks log their
    own incidents (log ids containing "FSP"), and CHP occasionally opens a second incident for the
    same crash. Duplicates sit on the same freeway and direction, within 30 minutes, at the same sensor
    (identical pre-crash reading, so the same 5-minute slot), and always get the same label. Kept,
    each such crash counts twice, which overstates how much data there is.

    Each group of duplicates keeps one report: a CHP one over an FSP one, then the earliest.
    """
    keys = ["freeway", "direction", "site_speed", "site_flow"]
    first = df.sort_values("landmark_min").drop_duplicates("log_id")[["log_id", "log_time", *keys]].dropna()
    pairs = first.merge(first, on=keys, suffixes=("", "_b"))
    pairs = pairs[(pairs["log_id"] < pairs["log_id_b"])
                  & ((pairs["log_time"] - pairs["log_time_b"]).abs() <= pd.Timedelta(window))]
    parent = {}

    def root(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for a, b in zip(pairs["log_id"], pairs["log_id_b"]):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    if not parent:
        return df
    grouped = first[first["log_id"].isin(set(pairs["log_id"]) | set(pairs["log_id_b"]))].copy()
    grouped["group"] = grouped["log_id"].map(root)
    grouped["is_fsp"] = grouped["log_id"].str.contains("FSP", na=False)
    keep = grouped.sort_values(["is_fsp", "log_time", "log_id"]).drop_duplicates("group")["log_id"]
    return df[~df["log_id"].isin(set(grouped["log_id"]) - set(keep))]


def split_by_day(df: pd.DataFrame, test_share: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The most recent whole days, up to test_share of incidents (at least one day), are the second part."""
    per_day = df.groupby("incident_day")["log_id"].nunique().sort_index(ascending=False)
    if len(per_day) < 2:
        raise ValueError(f"need at least 2 days of incidents to split, have {len(per_day)}")
    share = per_day.cumsum() / per_day.sum()
    n_test = min(max(1, int((share <= test_share + 1e-9).sum())), len(per_day) - 1)
    is_test = df["incident_day"].isin(per_day.index[:n_test])
    return df[~is_test], df[is_test]


def more_backup(df: pd.DataFrame) -> pd.Series:
    return df["remaining_minutes"] > 0


def check_enough(part: pd.DataFrame, name: str) -> None:
    more = more_backup(part)
    if more.sum() < MIN_POSITIVE or (~more).sum() < MIN_POSITIVE:
        raise ValueError(f"{name} has {more.sum()} backed-up and {(~more).sum()} clear rows; "
                         f"need {MIN_POSITIVE} of each. Collect more days of data.")


# ---------- rule baseline ----------

def km_median(minutes: pd.Series, censored: pd.Series) -> float:
    """Kaplan-Meier median: the duration half of backups outlast, counting "still going at the end"
    rows as lower bounds instead of as if they ended there. If more than half are still going
    when the data ends, the longest observed time is a lower bound and is used."""
    t, e = np.asarray(minutes, float), ~np.asarray(censored, bool)
    surv = 1.0
    for u in np.unique(t[e]):
        at_risk = (t >= u).sum()
        surv *= 1 - ((t == u) & e).sum() / at_risk
        if surv <= 0.5:
            return float(u)
    return float(t.max())


@dataclass
class Rule:
    """Typical answers for past incidents in the same group, falling back to coarser groups when a
    group has fewer than min_group examples."""
    keys: list[str]
    min_group: int = 10

    def fit(self, train: pd.DataFrame) -> Rule:
        levels = [self.keys[:k] for k in range(len(self.keys), 0, -1)]
        more = more_backup(train)
        pos = train[more]
        self._p = [(lv, train.assign(m=more).groupby(lv, observed=True)["m"].agg(["mean", "size"])) for lv in levels]
        def durations(g: pd.DataFrame) -> pd.Series:
            return pd.Series({"median": km_median(g["remaining_minutes"], g["is_censored"] == 1), "size": len(g)})

        self._d = [(lv, pos.groupby(lv, observed=True)[["remaining_minutes", "is_censored"]].apply(durations))
                   for lv in levels]
        self._p_all = float(more.mean())
        self._d_all = km_median(pos["remaining_minutes"], pos["is_censored"] == 1)
        return self

    def _lookup(self, df: pd.DataFrame, tables, stat: str, overall: float) -> np.ndarray:
        out = pd.Series(np.nan, index=df.index)
        for keys, table in tables:
            table = table[table["size"] >= self.min_group]
            out = out.fillna(df[keys].merge(table, left_on=keys, right_index=True, how="left")[stat])
        return out.fillna(overall).to_numpy()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"p_more_backup": self._lookup(df, self._p, "mean", self._p_all),
                             "minutes_if_more": self._lookup(df, self._d, "median", self._d_all)}, index=df.index)


# ---------- boosted model ----------

# Shallow trees, a slow learning rate and at least 20 rows per leaf: chosen on the development days
# (before Oct 15) only. With a few thousand incidents, deeper trees fit day-specific noise.
CLASSIFIER = dict(learning_rate=0.03, max_depth=3, min_child_weight=20, subsample=0.8,
                  colsample_bytree=0.8, enable_categorical=True, tree_method="hist")
SURVIVAL = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0, "learning_rate": 0.05, "max_depth": 3,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist"}


@dataclass
class BoostedModel:
    columns: list[str]
    seed: int = 0

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model inputs; categories never seen in training become missing instead of an error."""
        x = df[self.columns].copy()
        for c, levels in self.categories.items():
            x[c] = pd.Categorical(x[c].astype("object"), categories=levels)
        return x

    def _survival_matrix(self, part: pd.DataFrame) -> xgb.DMatrix:
        part = part[more_backup(part)]
        m = xgb.DMatrix(self._x(part), enable_categorical=True)
        lower = part["remaining_minutes"].to_numpy(float)
        m.set_float_info("label_lower_bound", lower)
        m.set_float_info("label_upper_bound", np.where(part["is_censored"] == 1, np.inf, lower))
        return m

    def fit(self, train: pd.DataFrame) -> BoostedModel:
        """Pick the number of trees on the latest training days, then refit on all training days."""
        self.categories = {c: list(train[c].dropna().unique()) for c in self.columns if c in CATEGORICAL}
        fit, valid = split_by_day(train, test_share=0.2)
        check_enough(fit, "training data")
        check_enough(valid, "validation data")

        trial = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=50, eval_metric="logloss",
                                  random_state=self.seed, **CLASSIFIER)
        trial.fit(self._x(fit), more_backup(fit), eval_set=[(self._x(valid), more_backup(valid))], verbose=False)
        self.classifier = xgb.XGBClassifier(n_estimators=trial.best_iteration + 1, random_state=self.seed, **CLASSIFIER)
        self.classifier.fit(self._x(train), more_backup(train))

        params = {**SURVIVAL, "seed": self.seed}
        trial_s = xgb.train(params, self._survival_matrix(fit), num_boost_round=500,
                            evals=[(self._survival_matrix(valid), "valid")], early_stopping_rounds=50,
                            verbose_eval=False)
        self.survival = xgb.train(params, self._survival_matrix(train), num_boost_round=trial_s.best_iteration + 1)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._x(df)
        return pd.DataFrame({"p_more_backup": self.classifier.predict_proba(x)[:, 1],
                             "minutes_if_more": self.survival.predict(xgb.DMatrix(x, enable_categorical=True))},
                            index=df.index)

    def importance(self, top: int = 15) -> pd.DataFrame:
        gain_c = self.classifier.get_booster().get_score(importance_type="total_gain")
        gain_s = self.survival.get_score(importance_type="total_gain")
        out = pd.DataFrame({"will_back_up": pd.Series(gain_c, dtype=float),
                            "how_long": pd.Series(gain_s, dtype=float)}).fillna(0)
        out = out / out.sum().replace(0, 1)
        return out.sort_values("will_back_up", ascending=False).head(top).round(3)
