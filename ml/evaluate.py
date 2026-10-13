"""Leave-one-day-out evaluation: train on every other day, predict the held-out day, repeat for
each day. Every incident gets exactly one out-of-sample prediction from each method, so weekdays
and weekends are all scored.

95% intervals come from resampling whole DAYS (not incidents): incidents on the same day share
weather, events and the same fold's model, so with ~2 weeks of data the honest sample size is
the number of days. The comparison that matters, each model minus the rule, is resampled in
pairs from the same days, so "boosted beats rule" means that difference's interval excludes 0.

Scores, separately for the two questions:
  will it back up?   AUC (0.5 = guessing, 1 = perfect ranking), Brier score (mean squared error of
                     the probability; lower is better), calibration table
  how much longer?   C-index on rows that did back up (0.5 = guessing), and the median error in
                     minutes on those whose end was observed (the shorter ones, so this error
                     slightly favours short predictions; the C-index handles censoring properly)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backup_model import BoostedModel, Rule, Variant, more_backup
from simple_models import SimpleModel

METHODS = ["rule", "boosted", "simple"]
GOOD_LABEL = 0.8  # label_adjusted_share at or above this counts as a full-baseline label


def cross_validate(df: pd.DataFrame, variant: Variant, seed: int = 0, log=print) -> pd.DataFrame:
    """Out-of-sample predictions for every row, one held-out day at a time."""
    parts = []
    days = sorted(df["incident_day"].unique())
    for i, day in enumerate(days, 1):
        train, test = df[df["incident_day"] != day], df[df["incident_day"] == day]
        try:
            preds = {
                "rule": Rule(variant.rule_keys).fit(train).predict(test),
                "boosted": BoostedModel(variant.features, seed).fit(train).predict(test),
                "simple": SimpleModel(variant.simple_features).fit(train).predict(test),
            }
        except ValueError as e:  # too little data in this fold: skip the day for every method alike
            log(f"  skipped {day}: {e}")
            continue
        out = test[["log_id", "landmark_min", "incident_day", "day_type", "remaining_minutes",
                    "is_censored", "label_adjusted_share"]].copy()
        for m, p in preds.items():
            out[f"{m}_p"], out[f"{m}_min"] = p["p_more_backup"], p["minutes_if_more"]
        parts.append(out)
        log(f"  held out {day} ({i}/{len(days)}): {test['log_id'].nunique()} incidents")
    return pd.concat(parts)


# ---------- metrics ----------

def concordance(time, event, predicted) -> float:
    """Harrell's C: of pairs where we know which backup cleared first, how often the model ranked
    them right (0.5 = guessing). A censored backup is only known to outlast events before it;
    at a tied time, the one that cleared counts as shorter than the one still going.

    Fast because durations come in 5-minute steps: for each distinct event time, the predictions
    of everything that lasted longer are sorted once and counted with a binary search.
    """
    time, event, predicted = (np.asarray(a, dtype=float) for a in (time, event, predicted))
    is_event = event == 1
    num = den = 0.0
    for t in np.unique(time[is_event]):
        ev = predicted[is_event & (time == t)]
        longer = np.sort(predicted[(time > t) | ((time == t) & ~is_event)])
        if len(longer) == 0:
            continue
        right_edge = np.searchsorted(longer, ev, side="right")
        ties = right_edge - np.searchsorted(longer, ev, side="left")
        num += (len(longer) - right_edge).sum() + 0.5 * ties.sum()
        den += len(longer) * len(ev)
    return float(num / den) if den else float("nan")


def _auc(more, p) -> float:
    return float(roc_auc_score(more, p)) if more.nunique() == 2 else float("nan")


def scores(oof: pd.DataFrame, method: str) -> dict:
    more = more_backup(oof)
    pos = oof[more]
    known = pos[pos["is_censored"] == 0]
    return {
        "auc": _auc(more, oof[f"{method}_p"]),
        "brier": float(((oof[f"{method}_p"] - more) ** 2).mean()),
        "c_index": concordance(pos["remaining_minutes"], 1 - pos["is_censored"], pos[f"{method}_min"]),
        "median_err_min_uncensored": float((known[f"{method}_min"] - known["remaining_minutes"]).abs().median()),
    }


def summary(oof: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Overall scores per method, and each model minus the rule, with 95% intervals from
    resampling whole days."""
    rng = np.random.default_rng(seed)
    days = oof["incident_day"].unique()
    groups = oof.groupby("incident_day").indices
    keys = ["auc", "brier", "c_index", "median_err_min_uncensored"]

    def all_scores(frame):
        s = {m: scores(frame, m) for m in METHODS}
        for m in ("boosted", "simple"):
            s[f"{m} - rule"] = {k: s[m][k] - s["rule"][k] for k in keys}
        return s

    point = all_scores(oof)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        boots.append(all_scores(oof.iloc[np.concatenate([groups[d] for d in pick])]))
    rows = []
    for name in point:
        row = {"method": name}
        for k in keys:
            lo, hi = np.nanpercentile([b[name][k] for b in boots], [2.5, 97.5])
            digits = 1 if k.startswith("median") else 3
            row[k] = f"{point[name][k]:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def by_group(oof: pd.DataFrame) -> pd.DataFrame:
    """AUC and C-index per checkpoint and for weekdays vs weekends."""
    groups = [(f"{lm:.0f} min", g) for lm, g in oof.groupby("landmark_min")]
    groups += [(str(dt), g) for dt, g in oof.groupby("day_type")]
    rows = []
    for name, g in groups:
        row = {"group": name, "rows": len(g), "share_more": round(more_backup(g).mean(), 3)}
        for m in METHODS:
            s = scores(g, m)
            row[f"auc_{m}"], row[f"c_{m}"] = round(s["auc"], 3), round(s["c_index"], 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def calibration(oof: pd.DataFrame, method: str, bins=(0, 0.05, 0.1, 0.2, 0.4, 1.0)) -> pd.DataFrame:
    """Do incidents given X% chance of more backup actually back up about X% of the time?"""
    p = oof[f"{method}_p"]
    cut = pd.cut(p, bins=list(bins), include_lowest=True)
    return (oof.assign(bin=cut, more=more_backup(oof), p=p)
               .groupby("bin", observed=True)
               .agg(rows=("more", "size"), predicted=("p", "mean"), actual=("more", "mean"))
               .round(3))


def per_day(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, g in oof.groupby("incident_day"):
        row = {"day": day, "day_type": g["day_type"].iloc[0], "incidents": g["log_id"].nunique()}
        for m in METHODS:
            s = scores(g, m)
            row[f"auc_{m}"], row[f"c_{m}"] = round(s["auc"], 3), round(s["c_index"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def good_labels(oof: pd.DataFrame) -> pd.DataFrame:
    """Rows whose label rests mostly on the full "typical traffic" baseline."""
    return oof[oof["label_adjusted_share"] >= GOOD_LABEL]
