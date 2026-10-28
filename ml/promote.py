"""Replace the current model only when a retrained one is better, not just because it's Monday.

For each deployable feature set (CHP + history only, CHP + pre-crash):

  1. The champion is the model currently in ml/models/current/<feature set>/ (synced from S3 by the
     weekly job). It was trained on data up to some day.
  2. The challenger is this week's code trained on the same kind of data the champion had: every
     day before the comparison window (the latest 7 days the champion never saw).
  3. Both predict the comparison window. The challenger is promoted when its AUC is at least the
     champion's (point estimate, with the day-resampled interval of the difference reported).
  4. A promoted challenger is refit on all data and saved as the new champion.

No champion yet, or a champion that can't score the current data (its features no longer exist):
the challenger is promoted. Fewer than MIN_WINDOW_DAYS days the champion hasn't seen: the champion
stays and the comparison waits for more data.

    python ml/promote.py              # uses the cached table from train.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from backup_model import VARIANTS, BoostedModel, drop_duplicate_reports, more_backup, prepare

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models" / "current"
DEPLOYABLE = ["CHP + history only", "CHP + pre-crash only"]
WINDOW_DAYS = 7
MIN_WINDOW_DAYS = 3


def slug(name: str) -> str:
    return name.replace(" + ", "_").replace(" ", "_").replace("-", "_").lower()


# ---------- saving and loading ----------

def save(model: BoostedModel, folder: Path, trained_through, extra: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.classifier.save_model(folder / "will_back_up.json")
    model.survival.save_model(folder / "how_long.json")
    meta = {"features": model.columns, "categories": {c: [str(v) for v in lv] for c, lv in model.categories.items()},
            "trained_through": str(trained_through), "saved_on": str(date.today()), **extra}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def load(folder: Path) -> tuple[BoostedModel, dict] | None:
    if not (folder / "meta.json").exists():
        return None
    meta = json.loads((folder / "meta.json").read_text())
    model = BoostedModel(meta["features"])
    model.categories = meta["categories"]
    model.classifier = xgb.XGBClassifier()
    model.classifier.load_model(folder / "will_back_up.json")
    model.survival = xgb.Booster()
    model.survival.load_model(folder / "how_long.json")
    return model, meta


# ---------- the decision ----------

def paired_interval(y, p_new, p_old, days, n_boot=200, seed=0) -> tuple[float, float, float]:
    """AUC(challenger) - AUC(champion), with a 95% interval from resampling days."""
    rng = np.random.default_rng(seed)
    y, p_new, p_old, days = (np.asarray(a) for a in (y, p_new, p_old, days))
    groups = {d: np.flatnonzero(days == d) for d in np.unique(days)}
    keys = list(groups)

    def diff(idx):
        return roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_old[idx]) if len(set(y[idx])) == 2 else np.nan

    point = diff(np.arange(len(y)))
    boots = [diff(np.concatenate([groups[d] for d in rng.choice(keys, len(keys))])) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def challenger(features: list[str], seed: int = 0) -> BoostedModel:
    return BoostedModel(features, seed)


def decide(df: pd.DataFrame, name: str, folder: Path, seed: int = 0) -> dict:
    variant = next(v for v in VARIANTS if v.name == name)
    days = sorted(df["incident_day"].unique())
    current = load(folder)
    trained_through = pd.Timestamp(current[1]["trained_through"]).date() if current else None
    unseen = [d for d in days if trained_through is None or d > trained_through][-WINDOW_DAYS:]
    row = {"feature_set": name, "champion_trained_through": str(trained_through) if current else "none",
           "window": f"{unseen[0]} to {unseen[-1]}" if unseen else "none"}

    if current is None:
        row.update(decision="promote", reason="no current model")
    elif len(unseen) < MIN_WINDOW_DAYS:
        return {**row, "decision": "keep", "reason": f"only {len(unseen)} new days since the current model"}
    else:
        window = df[df["incident_day"].isin(unseen)]
        before = df[df["incident_day"] < unseen[0]]
        champion, meta = current
        missing = set(meta["features"]) - set(df.columns)
        if missing:
            row.update(decision="promote", reason=f"current model needs columns no longer present: {sorted(missing)}")
        else:
            p_old = champion.predict(window)["p_more_backup"]
            p_new = challenger(variant.features, seed).fit(before).predict(window)["p_more_backup"]
            y = more_backup(window)
            diff, lo, hi = paired_interval(y, p_new, p_old, window["incident_day"])
            row.update(champion_auc=round(roc_auc_score(y, p_old), 4), challenger_auc=round(roc_auc_score(y, p_new), 4),
                       difference=f"{diff:+.4f} ({lo:+.4f} to {hi:+.4f})")
            if diff >= 0:
                row.update(decision="promote", reason="challenger ranks the new days at least as well")
            else:
                return {**row, "decision": "keep", "reason": "current model ranks the new days better"}

    final = challenger(variant.features, seed).fit(df)
    save(final, folder, max(days), {"feature_set": name, "promotion": {k: v for k, v in row.items()}})
    return row


def main() -> None:
    from train import OUT, load as load_table  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df = drop_duplicate_reports(prepare(load_table(args.profile, args.refresh)))
    rows = [decide(df, name, MODELS / slug(name)) for name in DEPLOYABLE]
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("MODEL PROMOTION (replace the current model only if the retrained one is at least as good)")
        print(table.to_string(index=False))
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "promotion.csv", index=False)


if __name__ == "__main__":
    main()
