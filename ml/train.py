"""Train and score the backup-duration model.

    python ml/train.py              # downloads fct_training_landmarks from Athena (cached in ml/data/)
    python ml/train.py --refresh    # download again after new data / a dbt build

Trains two versions (with and without live traffic, see backup_model.py) and writes ml/output/:
metrics.csv, importance.csv, test_predictions.csv and the model files.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from backup_model import (
    VARIANTS, evaluate, fit_baseline, fit_model, gain_interval, importance, predict_baseline, prepare, split_by_day,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "fct_training_landmarks.parquet"
OUT = HERE / "output"


def load(profile: str, refresh: bool) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    import awswrangler as wr
    import boto3

    session = boto3.Session(profile_name=profile, region_name="us-west-2")
    df = wr.athena.read_sql_query(
        "select * from fct_training_landmarks",
        database="traffic_silver", workgroup="traffic", boto3_session=session, ctas_approach=False,
    )
    CACHE.parent.mkdir(exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="traffic")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    df = prepare(load(args.profile, args.refresh))
    train, test = split_by_day(df)
    print(f"{df['log_id'].nunique():,} incidents, {len(df):,} rows")
    print(f"train: {train['incident_day'].min()} to {train['incident_day'].max()} ({train['log_id'].nunique():,} incidents)")
    print(f"test:  {test['incident_day'].min()} to {test['incident_day'].max()} ({test['log_id'].nunique():,} incidents)\n")

    OUT.mkdir(exist_ok=True)
    all_metrics, all_importance = [], []
    predictions = test[["log_id", "landmark_min", "log_time", "type_code", "lanes_group",
                        "backed_up_now", "remaining_minutes", "is_censored"]].copy()

    for name, (columns, baseline_keys) in VARIANTS.items():
        baseline = fit_baseline(train, baseline_keys)
        model = fit_model(train, columns)
        base_pred, model_pred = predict_baseline(baseline, test), model.predict(test)
        metrics = evaluate(test, base_pred, model_pred)
        gain, lo, hi = gain_interval(test, base_pred, model_pred)
        top = importance(model)

        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(f"=== {name} ===")
            print("Test scores (MAE in minutes, lower is better; AUC and C-index, higher is better, 0.5 = guessing)")
            print(metrics.to_string(index=False))
            print(f"\nModel vs rule: saves {gain:.1f} min of error per prediction "
                  f"(95% interval {lo:.1f} to {hi:.1f}; if it includes 0, the difference could be luck)\n")
            print("What the model leans on (share of total gain)")
            print(top.to_string(), "\n")

        slug = re.sub(r"[^a-z]+", "_", name.lower()).strip("_")
        all_metrics.append(metrics.assign(variant=name))
        all_importance.append(top.assign(variant=name))
        predictions[f"rule_minutes__{slug}"] = base_pred
        predictions[f"model_minutes__{slug}"] = model_pred["predicted_minutes"]
        predictions[f"p_more_backup__{slug}"] = model_pred["p_more_backup"]
        model.classifier.save_model(OUT / f"will_back_up__{slug}.json")
        model.survival.save_model(OUT / f"how_long__{slug}.json")

    pd.concat(all_metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.concat(all_importance).to_csv(OUT / "importance.csv")
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
