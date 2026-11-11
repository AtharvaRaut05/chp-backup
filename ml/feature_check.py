"""Does a new kind of data help? Test each candidate feature group before it joins FEATURES.

For each group in backup_model.CANDIDATES (weather, road layout, calendar, lane closures, speed at report) and each feature set that can run live (CHP + history only, CHP + pre-crash only):

  1. Keep only incidents where the group exists (after its collection started, placed on the road...).
     Comparing on the same rows keeps "newer days" from posing as "better data".
  2. Split the days our poller saw into 3 groups; score each group with a model trained on everything
     else from the last HISTORY_DAYS, PeMS history included (whole days stay together, because incidents on the same day share
     weather, holidays and traffic). History is never scored.
  3. Same model settings with and without the group: AUC for "more backup ahead?" both ways, and the
     difference with a 95% interval from resampling days.

A group is worth adding when the interval is above zero on the development days and the gain holds on
later weeks. Groups with too few covered days or backups are reported as "not enough data yet".

    python ml/feature_check.py                        # uses ml/data (run python ml/train.py --refresh first)
    python ml/feature_check.py --group "lane closures"
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from backup_model import (CANDIDATES, CATEGORICAL, CLASSIFIER, VARIANTS, check_enough, drop_duplicate_reports,
                          is_live, live_days, more_backup, prepare, split_by_day)

MIN_DAYS = 5
FOLDS = 3
HISTORY_DAYS = 180  # train on the latest 6 months only: covers Memorial Day to now, at a third of the cost
DEPLOYABLE = [v for v in VARIANTS if v.name in ("CHP + history only", "CHP + pre-crash only")]


def covered(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    return df[df[group].notna().any(axis=1)]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int = 0) -> np.ndarray:
    """The boosted classifier (same settings and early stopping as BoostedModel) -> P(more backup)."""
    cats = {c: list(train[c].dropna().unique()) for c in columns if c in CATEGORICAL}

    def x(part):
        out = part[columns].copy()
        for c, levels in cats.items():
            out[c] = pd.Categorical(out[c].astype("object"), categories=levels)
        return out

    fit, valid = split_by_day(train, test_share=0.2)
    check_enough(fit, "training data")
    check_enough(valid, "validation data")
    trial = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=50, eval_metric="logloss",
                              random_state=seed, **CLASSIFIER)
    trial.fit(x(fit), more_backup(fit), eval_set=[(x(valid), more_backup(valid))], verbose=False)
    final = xgb.XGBClassifier(n_estimators=trial.best_iteration + 1, random_state=seed, **CLASSIFIER)
    final.fit(x(train), more_backup(train))
    return final.predict_proba(x(test))[:, 1]


def day_folds(days: list, k: int) -> list[list]:
    """Days in date order dealt round-robin into k folds, so each fold spans the whole period."""
    return [days[i::k] for i in range(k)]


def compare(df: pd.DataFrame, base: list[str], group: list[str], n_boot: int = 200, seed: int = 0) -> dict:
    days = live_days(df)
    more = more_backup(df)
    if len(days) < MIN_DAYS or more.sum() < 50 or (~more).sum() < 50:
        return {"rows": len(df), "days": len(days), "backed_up": int(more.sum()), "note": "not enough data yet"}
    with_group = base + [c for c in group if c not in base]
    p_base, p_with = pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    for fold in day_folds(days, min(FOLDS, len(days))):
        held = df["incident_day"].isin(fold)
        test = held & is_live(df)
        try:
            p_base[test] = fit_predict(df[~held], df[test], base, seed)
            p_with[test] = fit_predict(df[~held], df[test], with_group, seed)
        except ValueError:  # a fold too small to fit: leave its rows out of both
            continue
    ok = p_base.notna()
    y, day = more[ok].to_numpy(), df.loc[ok, "incident_day"].to_numpy()
    a, b = p_base[ok].to_numpy(), p_with[ok].to_numpy()
    rng, uniq, diffs = np.random.default_rng(seed), np.unique(day), []
    idx = {d: np.flatnonzero(day == d) for d in uniq}
    for _ in range(n_boot):
        rows = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq))])
        if 0 < y[rows].sum() < len(rows):
            diffs.append(roc_auc_score(y[rows], b[rows]) - roc_auc_score(y[rows], a[rows]))
    auc_base, auc_with = roc_auc_score(y, a), roc_auc_score(y, b)
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
    return {"rows": int(ok.sum()), "days": len(np.unique(day)), "backed_up": int(y.sum()),
            "auc_without": round(auc_base, 4), "auc_with": round(auc_with, 4),
            "gain": round(auc_with - auc_base, 4), "gain_low": round(lo, 4), "gain_high": round(hi, 4),
            "note": "helps" if lo > 0 else ("hurts" if hi < 0 else "no clear difference")}


def run(df: pd.DataFrame, groups: dict[str, list[str]], n_boot: int = 200, log=print) -> pd.DataFrame:
    rows = []
    for name, group in groups.items():
        part = covered(df, group)
        for v in DEPLOYABLE:
            log(f"{name} / {v.name}: {part['log_id'].nunique():,} incidents covered")
            rows.append({"group": name, "feature_set": v.name, **compare(part, v.features, group, n_boot)})
    return pd.DataFrame(rows)


def main() -> None:
    from train import OUT, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--group", choices=list(CANDIDATES), help="test one group (default: all)")
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    df = drop_duplicate_reports(prepare(load(args.profile, args.refresh)))
    df = df[df["incident_day"] >= max(df["incident_day"]) - pd.Timedelta(days=HISTORY_DAYS)]
    groups = {args.group: CANDIDATES[args.group]} if args.group else CANDIDATES
    table = run(df, groups, args.boot)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "feature_check.csv", index=False)


if __name__ == "__main__":
    main()
