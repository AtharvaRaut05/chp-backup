"""Evaluation in time order, the way the models would actually be used.

Leave-one-day-out (evaluate.py) uses every day for testing, but trains on days that come after the
day it predicts. The checks here never let a model see the future:

  forward test     train on every day before the cutoff, test on the days after it. Alert thresholds
                   are fixed from probabilities before any test day is scored.
  alert tradeoff   precision, recall, alerts per day and missed backups per day at thresholds set by
                   how much worse a missed backup is than an unneeded alert.
  retraining       a model trained once and never updated ("frozen") against one retrained every
                   Monday on all data so far, scored on the same later days. Plus a learning curve:
                   the same test week, predicted from the most recent 3, 5, 7... training days.
  failure cases    which incidents the deployable model misses or alerts on wrongly, and where it
                   ranks worst.

    python ml/forward_test.py            # uses the cached table from train.py
    python ml/forward_test.py --refresh  # download it again first
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from backup_model import VARIANTS, BoostedModel, Rule, Variant, drop_duplicate_reports, more_backup, prepare
from evaluate import METHODS, scores
from simple_models import SimpleModel

# A missed backup costs this many times an unneeded alert -> alert when P(backup) > 1 / (1 + ratio).
# For calibrated probabilities that threshold minimizes expected cost (see the README, "Missing a backup vs. an unnecessary alert").
COST_RATIOS = [1, 2, 4, 9]
ALERT_CHECKPOINT = 5     # minutes after the CHP log: the first moment an alert could go out
TARGET_RECALL = 0.8      # the second way to set a threshold: catch 80% of backups on validation days
MIN_TRAIN_DAYS = 7       # forward test: at least a full week of every day of the week to train on
MIN_RETRAIN_DAYS = 5     # retraining study: the first weekly model, so the study starts as early as possible


# ---------- fitting ----------

@dataclass
class Fitted:
    models: dict

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["log_id", "landmark_min", "incident_day", "day_type", "remaining_minutes", "is_censored"]].copy()
        for m, model in self.models.items():
            p = model.predict(df)
            out[f"{m}_p"], out[f"{m}_min"] = p["p_more_backup"].to_numpy(), p["minutes_if_more"].to_numpy()
        return out


def fit_all(train: pd.DataFrame, variant: Variant, seed: int = 0) -> Fitted:
    return Fitted({"rule": Rule(variant.rule_keys).fit(train),
                   "boosted": BoostedModel(variant.features, seed).fit(train),
                   "simple": SimpleModel(variant.simple_features).fit(train)})


def days_of(df: pd.DataFrame) -> list:
    return sorted(df["incident_day"].unique())


def chronological_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = days_of(df)
    if len(days) < test_days + MIN_TRAIN_DAYS:
        raise ValueError(f"need {test_days + MIN_TRAIN_DAYS} days for a forward test, have {len(days)}")
    cutoff = days[-test_days]
    return df[df["incident_day"] < cutoff], df[df["incident_day"] >= cutoff]


# ---------- alert metrics ----------

def alert_stats(more: pd.Series, p: pd.Series, threshold: float, n_days: int) -> dict:
    alert = p > threshold
    tp, fp = int((alert & more).sum()), int((alert & ~more).sum())
    fn = int((~alert & more).sum())
    return {"threshold": round(threshold, 3),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "alerts_per_day": (tp + fp) / n_days, "missed_per_day": fn / n_days,
            "false_alarms_per_day": fp / n_days}


def recall_threshold(more: pd.Series, p: pd.Series, target: float) -> float:
    """Highest threshold that still catches `target` of the backups."""
    pos = np.sort(p[more].to_numpy())[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos))) - 1
    return float(np.nextafter(pos[k], -np.inf))


def thresholds_from_validation(train: pd.DataFrame, variant: Variant, seed: int = 0) -> dict:
    """Thresholds chosen without touching the test days: cost-ratio thresholds are fixed by formula;
    the recall threshold comes from a model fit on earlier training days and scored on the latest ones."""
    out = {f"miss = {k}x false alarm": 1 / (1 + k) for k in COST_RATIOS}
    days = days_of(train)
    n_valid = max(1, len(days) // 5)
    fit_part = train[train["incident_day"] < days[-n_valid]]
    valid = train[train["incident_day"] >= days[-n_valid]]
    valid = valid[valid["landmark_min"] == ALERT_CHECKPOINT]
    p = BoostedModel(variant.features, seed).fit(fit_part).predict(valid)["p_more_backup"]
    out[f"catch {TARGET_RECALL:.0%} (set on validation days)"] = recall_threshold(more_backup(valid), p, TARGET_RECALL)
    return out


def alert_table(test_pred: pd.DataFrame, thresholds: dict, method: str = "boosted") -> pd.DataFrame:
    first = test_pred[test_pred["landmark_min"] == ALERT_CHECKPOINT]
    more, n_days = more_backup(first), first["incident_day"].nunique()
    rows = [{"policy": name, **alert_stats(more, first[f"{method}_p"], t, n_days)} for name, t in thresholds.items()]
    return pd.DataFrame(rows).set_index("policy").round(3)


# ---------- forward test scores ----------

def forward_scores(pred: pd.DataFrame, threshold: float, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """AUC, average precision (area under the precision-recall curve), Brier, precision and recall
    at `threshold`, C-index; 95% intervals from resampling test days; model minus rule paired."""
    rng = np.random.default_rng(seed)
    groups = pred.groupby("incident_day").indices
    days = list(groups)
    keys = ["auc", "avg_precision", "precision", "recall", "brier", "c_index"]

    def all_scores(frame):
        more = more_backup(frame)
        s = {}
        for m in METHODS:
            base = scores(frame, m)
            alert = frame[f"{m}_p"] > threshold
            tp = (alert & more).sum()
            s[m] = {"auc": base["auc"], "brier": base["brier"], "c_index": base["c_index"],
                    "avg_precision": average_precision_score(more, frame[f"{m}_p"]) if more.any() else np.nan,
                    "precision": tp / alert.sum() if alert.sum() else np.nan,
                    "recall": tp / more.sum() if more.sum() else np.nan}
        for m in ("boosted", "simple"):
            s[f"{m} - rule"] = {k: s[m][k] - s["rule"][k] for k in keys}
        return s

    point = all_scores(pred)
    boots = [all_scores(pred.iloc[np.concatenate([groups[d] for d in rng.choice(days, len(days))])])
             for _ in range(n_boot)]
    rows = []
    for name in point:
        row = {"method": name}
        for k in keys:
            lo, hi = np.nanpercentile([b[name][k] for b in boots], [2.5, 97.5])
            row[k] = f"{point[name][k]:.3f} ({lo:.3f} to {hi:.3f})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def precision_at_recall(pred: pd.DataFrame, method: str = "boosted", levels=(0.5, 0.7, 0.9)) -> pd.DataFrame:
    """For each checkpoint: the precision you'd get if the threshold were set to catch X% of backups
    (an oracle threshold, so it describes the ranking, not an alert policy)."""
    rows = []
    for lm, g in pred.groupby("landmark_min"):
        more, p = more_backup(g), g[f"{method}_p"]
        row = {"checkpoint_min": lm, "rows": len(g), "backup_rate": more.mean()}
        for r in levels:
            t = recall_threshold(more, p, r)
            row[f"precision_at_{r:.0%}_recall"] = ((p > t) & more).sum() / max((p > t).sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("checkpoint_min").round(3)


# ---------- retraining ----------

def retrain_points(df: pd.DataFrame, min_train_days: int = MIN_RETRAIN_DAYS) -> list:
    """Mondays with at least `min_train_days` days of data before them: when the weekly job retrains."""
    days = days_of(df)
    return [d for d in days if pd.Timestamp(d).dayofweek == 0 and sum(x < d for x in days) >= min_train_days]


def retraining_study(df: pd.DataFrame, variant: Variant, n_boot: int = 200, seed: int = 0, log=print):
    """Frozen (trained at the first retraining point) vs weekly retrained, on days where they differ."""
    points = retrain_points(df)
    if len(points) < 2:
        return None, f"need 2 retraining Mondays with {MIN_RETRAIN_DAYS}+ days before the first; have {len(points)}"
    models = {}
    for pt in points:
        log(f"  training on days before {pt}")
        models[pt] = fit_all(df[df["incident_day"] < pt], variant, seed)
    parts = []
    for i, pt in enumerate(points[1:], 1):
        end = points[i + 1] if i + 1 < len(points) else None
        week = df[(df["incident_day"] >= pt) & ((df["incident_day"] < end) if end else True)]
        frozen, fresh = models[points[0]].predict(week), models[pt].predict(week)
        out = fresh[["log_id", "landmark_min", "incident_day", "remaining_minutes", "is_censored"]].copy()
        out["week_of"] = pt
        out["training_days_frozen"] = sum(d < points[0] for d in days_of(df))
        out["training_days_retrained"] = sum(d < pt for d in days_of(df))
        for m in METHODS:
            out[f"frozen_{m}_p"], out[f"retrained_{m}_p"] = frozen[f"{m}_p"], fresh[f"{m}_p"]
        parts.append(out)
    pred = pd.concat(parts)
    return compare_frozen(pred, n_boot, seed), pred


def compare_frozen(pred: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = pred.groupby("incident_day").indices
    days = list(groups)

    def stat(frame):
        more = more_backup(frame)
        s = {}
        for m in METHODS:
            for kind in ("frozen", "retrained"):
                p = frame[f"{kind}_{m}_p"]
                s[(m, kind, "auc")] = roc_auc_score(more, p) if more.nunique() == 2 else np.nan
                s[(m, kind, "brier")] = float(((p - more) ** 2).mean())
            for k in ("auc", "brier"):
                s[(m, "retrained - frozen", k)] = s[(m, "retrained", k)] - s[(m, "frozen", k)]
        return s

    point = stat(pred)
    boots = [stat(pred.iloc[np.concatenate([groups[d] for d in rng.choice(days, len(days))])]) for _ in range(n_boot)]
    rows = []
    for (m, kind, k) in point:
        lo, hi = np.nanpercentile([b[(m, kind, k)] for b in boots], [2.5, 97.5])
        rows.append({"method": m, "model": kind, "metric": k, "value": f"{point[(m, kind, k)]:.3f} ({lo:.3f} to {hi:.3f})"})
    return (pd.DataFrame(rows).pivot_table(index=["method", "model"], columns="metric", values="value", aggfunc="first")
            .reindex(METHODS, level=0))


def learning_curve(df: pd.DataFrame, variant: Variant, test_days: int, windows=(3, 5, 7, 10, 14, 21, 28),
                   seed: int = 0, log=print) -> pd.DataFrame:
    """Same test days, predicted from the most recent N training days: does more history help?"""
    train, test = chronological_split(df, test_days)
    days = days_of(train)
    rows = []
    for n in [w for w in windows if w < len(days)] + [len(days)]:
        part = train[train["incident_day"] >= days[-n]]
        try:
            pred = fit_all(part, variant, seed).predict(test)
        except ValueError as e:
            log(f"  {n} days: {e}")
            continue
        more = more_backup(pred)
        row = {"training_days": n, "training_incidents": part["log_id"].nunique()}
        for m in ("rule", "boosted"):
            row[f"auc_{m}"] = scores(pred, m)["auc"]
            row[f"avg_precision_{m}"] = average_precision_score(more, pred[f"{m}_p"])
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("training_days").set_index("training_days").round(3)


# ---------- failure cases ----------

FLAGS = {
    "lanes blocked (per notes)": lambda d: d["blocking_now"] == 1,
    "no blocking note yet": lambda d: d["n_notes"].fillna(0) == 0,
    "SigAlert": lambda d: d["sigalert"] == 1,
    "weekday rush hour": lambda d: d["rush_hour_now"] == 1,
    "night (9 PM to 5 AM)": lambda d: (d["hour_local"] >= 21) | (d["hour_local"] < 5),
    "already slow before crash (<80% of typical)": lambda d: d["site_ratio_to_typical"] < 0.8,
    "normally congested now (<70% of free flow)": lambda d: d["typical_ratio_now"] < 0.7,
    "2 or fewer queue sensors": lambda d: d["queue_sensors"].fillna(0) <= 2,
    "label on partial baseline": lambda d: d["label_adjusted_share"] < 0.8,
}


def failure_cases(test: pd.DataFrame, pred: pd.DataFrame, threshold: float, method: str = "boosted"):
    """At the alert checkpoint and threshold: how misses and false alarms differ from correct calls."""
    first = test[test["landmark_min"] == ALERT_CHECKPOINT]
    p = pred.loc[first.index, f"{method}_p"]
    more, alert = more_backup(first), p > threshold
    outcome = np.select([more & alert, more & ~alert, ~more & alert], ["caught", "missed", "false alarm"],
                        "correct no-alert")
    flags = pd.DataFrame({name: f(first).astype(bool) for name, f in FLAGS.items()}, index=first.index)
    order = ["caught", "missed", "false alarm", "correct no-alert"]
    table = flags.groupby(outcome).mean().T.reindex(columns=order)  # an outcome can be absent in a small test
    table.loc["incidents"] = pd.Series(outcome).value_counts().reindex(order).fillna(0)
    table.loc["median backup minutes"] = (first.assign(o=outcome).loc[more]
                                          .groupby("o")["total_backup_minutes"].median()
                                          if "total_backup_minutes" in first else np.nan)
    subgroups = []
    for name, f in FLAGS.items():
        for value, g in pred.loc[test.index].groupby(f(test).astype(bool).to_numpy()):
            m = more_backup(g)
            if m.nunique() == 2 and m.sum() >= 10:
                subgroups.append({"group": f"{name}: {'yes' if value else 'no'}", "rows": len(g),
                                  "backup_rate": m.mean(), "auc": scores(g, method)["auc"]})
    worst = (first.assign(p=p, outcome=outcome)
             .query("outcome in ['missed', 'false alarm']")
             .assign(confidence=lambda d: np.where(d["outcome"] == "missed", 1 - d["p"], d["p"]))
             .sort_values("confidence", ascending=False)
             [["log_id", "incident_day", "outcome", "p", "type_code", "blocking_now", "lanes_blocked_now",
               "hour_local", "site_ratio_to_typical", "queue_sensors", "remaining_minutes"]].head(25))
    return table.round(3), pd.DataFrame(subgroups).set_index("group").round(3), worst


# ---------- main ----------

def run(df: pd.DataFrame, test_days: int = 7, n_boot: int = 200, out=None, log=print) -> dict:
    """All checks for every feature set. Returns the tables; writes them as CSV when `out` is a folder."""
    results = {}
    train, test = chronological_split(df, test_days)
    log(f"Forward test: train {min(days_of(train))} to {max(days_of(train))} ({train['log_id'].nunique():,} incidents), "
        f"test {min(days_of(test))} to {max(days_of(test))} ({test['log_id'].nunique():,} incidents)")
    for v in VARIANTS:
        slug = v.name.replace(" + ", "_").replace(" ", "_").replace("-", "_").lower()
        log(f"\n=== {v.name} ===")
        thresholds = thresholds_from_validation(train, v)
        pred = fit_all(train, v).predict(test)
        cost4 = thresholds["miss = 4x false alarm"]
        r = {
            "forward_scores": forward_scores(pred, cost4, n_boot),
            "forward_scores_at_alert_checkpoint": forward_scores(pred[pred["landmark_min"] == ALERT_CHECKPOINT], cost4, n_boot),
            # Rows not already backed up at the checkpoint: can the model see a backup coming, rather than
            # report one it can already see? (The live features partly observe a backup in progress.)
            "forward_scores_not_yet_backed_up": forward_scores(pred[test["backed_up_now"] != 1], cost4, n_boot),
            "precision_at_recall": precision_at_recall(pred),
            "alert_tradeoff": alert_table(pred, thresholds),
            "alert_tradeoff_rule": alert_table(pred, thresholds, "rule"),
        }
        r["failure_profile"], r["failure_subgroups"], r["failure_examples"] = failure_cases(test, pred, cost4)
        log("  retraining study")
        r["retraining"], retrain_pred = retraining_study(df, v, n_boot, log=log)
        log("  learning curve")
        r["learning_curve"] = learning_curve(df, v, test_days, log=log)
        r["forward_predictions"] = pred
        if isinstance(retrain_pred, pd.DataFrame):
            r["retraining_predictions"] = retrain_pred
        results[v.name] = r
        for k, t in r.items():
            if isinstance(t, pd.DataFrame):
                if k not in ("forward_predictions", "retraining_predictions"):
                    with pd.option_context("display.width", 220, "display.max_columns", 30):
                        log(f"\n{k}\n{t.to_string()}")
                if out is not None:
                    t.to_csv(out / f"forward__{k}__{slug}.csv")
            else:
                log(f"\n{k}: {t}")
    return results


def main() -> None:
    from train import OUT, load  # noqa: PLC0415  (train.py owns the download and cache)

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic", help='AWS login profile; "" = default credentials (the AWS job)')
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    df = drop_duplicate_reports(prepare(load(args.profile, args.refresh)))
    OUT.mkdir(exist_ok=True)
    run(df, args.test_days, args.boot, OUT)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
