"""Evaluate and train the backup models.

    python ml/train.py              # downloads fct_training_landmarks from Athena (cached in ml/data/)
    python ml/train.py --refresh    # download again after new data / a dbt build

For each feature set (with / without live traffic):
  1. leave-one-day-out evaluation of the rule, boosted and simple models (evaluate.py)
  2. the same scores on incidents whose labels rest on a full "typical traffic" baseline
  3. final models trained on all days, saved for later use

Writes ml/output/: scores, per-day scores, calibration, out-of-sample predictions, the simple
models' effects, boosted feature importance and the boosted model files.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from backup_model import VARIANTS, BoostedModel, prepare
from evaluate import by_group, calibration, cross_validate, good_labels, per_day, summary

MIN_DAYS = 3  # days needed for a day-resampled interval to mean anything
from simple_models import SimpleModel

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "fct_training_landmarks.parquet"
OUT = HERE / "output"


def load(profile: str, refresh: bool) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    import awswrangler as wr
    import boto3

    session = boto3.Session(profile_name=profile or None, region_name="us-west-2")
    df = wr.athena.read_sql_query(
        "select * from fct_training_landmarks",
        database="traffic_silver", workgroup="traffic", boto3_session=session, ctas_approach=False,
    )
    CACHE.parent.mkdir(exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def show(title: str, table: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(f"\n{title}\n{table.to_string()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic", help='AWS login profile; "" = default credentials (the AWS job)')
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples for the 95%% intervals")
    args = ap.parse_args()

    df = prepare(load(args.profile, args.refresh))
    more = df["remaining_minutes"] > 0
    print(f"{df['log_id'].nunique():,} incidents over {df['incident_day'].nunique()} days "
          f"({df['incident_day'].min()} to {df['incident_day'].max()}), {len(df):,} rows; "
          f"{df.loc[more, 'log_id'].nunique():,} incidents still backed up at a checkpoint "
          f"({more.sum():,} rows); {(df['label_adjusted_share'] >= 0.8).mean():.0%} of rows have full-baseline labels")
    OUT.mkdir(exist_ok=True)
    outputs = {k: [] for k in ("scores", "groups", "days", "calibration", "effects", "odds", "importance", "oof")}

    for v in VARIANTS:
        slug = re.sub(r"[^a-z]+", "_", v.name.lower()).strip("_")
        print(f"\n=== {v.name}: leave-one-day-out ===")
        t0 = time.time()
        oof = cross_validate(df, v)
        print(f"  ({time.time() - t0:.0f}s)")

        overall = summary(oof, n_boot=args.boot)
        good = good_labels(oof)
        overall_good = (summary(good, n_boot=args.boot) if good["incident_day"].nunique() >= MIN_DAYS
                        else pd.DataFrame({"note": [f"only {good['log_id'].nunique()} incidents with full-baseline labels"]}))
        groups, days = by_group(oof), per_day(oof)
        calib = pd.concat({m: calibration(oof, m) for m in ("boosted", "simple")}, axis=1)

        show("Will it back up (AUC, Brier) and how much longer (C-index, median error in minutes), "
             "with 95% intervals from resampling days. 'boosted - rule' beats the rule if its whole "
             "interval is above 0 (below 0 for Brier and error):", overall)
        show(f"Same, only incidents with full-baseline labels ({good['log_id'].nunique():,} incidents):", overall_good)
        show("AUC / C-index by checkpoint and day type:", groups)
        show("Calibration (predicted vs actual chance of more backup):", calib)

        final_simple = SimpleModel(v.simple_features).fit(df)
        final_boosted = BoostedModel(v.features).fit(df)
        effects, odds = final_simple.duration_effects(), final_simple.backup_odds()
        show("Simple model, how much longer: % change in typical remaining minutes", effects)
        show("Simple model, will it back up: % change in odds per standard deviation", odds)
        show("Boosted model leans on (share of total gain):", final_boosted.importance())

        final_boosted.classifier.save_model(OUT / f"will_back_up__{slug}.json")
        final_boosted.survival.save_model(OUT / f"how_long__{slug}.json")
        for key, table in (("scores", pd.concat({"all": overall, "good_labels": overall_good})),
                           ("groups", groups), ("days", days), ("calibration", calib),
                           ("effects", effects), ("odds", odds), ("importance", final_boosted.importance()),
                           ("oof", oof)):
            outputs[key].append(table.assign(variant=v.name))
        save(outputs)  # after each variant, so a later failure doesn't lose finished work

    print(f"\nSaved to {OUT}")


def save(outputs: dict) -> None:
    for key, tables in outputs.items():
        name = "out_of_sample_predictions" if key == "oof" else key
        pd.concat(tables).to_csv(OUT / f"{name}.csv")


if __name__ == "__main__":
    main()
