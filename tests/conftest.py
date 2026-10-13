import numpy as np
import pandas as pd
import pytest


def fake_landmarks(n_incidents=1500, days=9, seed=0) -> pd.DataFrame:
    """Same columns and Athena-style nullable dtypes as fct_training_landmarks.

    Blocked lanes and heavy traffic per lane make backups likelier and longer. The notes and
    "live" traffic columns are noisy views of whether traffic is backed up, so no single column
    gives the answer away.
    """
    from backup_model import RAW_COLUMNS  # imported here so the other test files don't need xgboost

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_incidents):
        log_time = pd.Timestamp("2026-10-01 07:00") + pd.Timedelta(minutes=int(rng.integers(0, days * 1440)))
        local = log_time.tz_localize("UTC").tz_convert("America/Los_Angeles")
        weekend = local.dayofweek >= 5
        site_lanes = int(rng.integers(2, 6))
        flow = float(rng.uniform(40, 160)) * site_lanes  # vehicles per 5 min across all lanes
        blocking = rng.random() < 0.4
        lanes = int(rng.integers(1, site_lanes + 1)) if blocking else None
        load = flow * 12 / (2000 * max(site_lanes - (lanes or 0), 0.5))  # demand vs capacity left
        has_backup = rng.random() < min(0.9, 0.05 + 0.4 * blocking + 0.3 * min(load, 1.5))
        total = float(15 * np.exp(0.5 * min(load, 3) + 0.3 * (lanes or 0)) * rng.lognormal(0, 0.4)) // 5 * 5 + 5 \
            if has_backup else 0.0
        censored = has_backup and total > 150
        total = min(total, 160.0) if censored else total
        for lm in (5, 10, 20, 30):
            remaining = max(total - lm, 0.0)
            looks_backed_up = (remaining > 0) != (rng.random() < 0.15)
            row = {c: None for c in RAW_COLUMNS}
            row.update({
                "log_id": f"i{i}", "landmark_min": lm, "log_time": log_time,
                "dispatch_id": "LACC", "district": "07", "freeway": "405", "direction": "N",
                "hour_local": local.hour, "day_of_week": local.dayofweek + 1,
                "day_type": "weekend" if weekend else "weekday",
                "type_code": rng.choice(["1182", "1183"]), "n_notes": int(rng.integers(0, 6)),
                "ever_blocking": blocking, "blocking_now": blocking and ((remaining > 0) != (rng.random() < 0.2)),
                "max_lanes_blocked": lanes, "latest_lanes_blocked": lanes, "all_lanes": False, "vehicles": int(rng.integers(1, 4)), "tow": blocking,
                "units_assigned": 1, "chp_closed": False, "site_speed": float(rng.normal(60, 8)),
                "site_flow": flow, "site_lanes": site_lanes, "approach_flow_per_sensor": flow * rng.uniform(0.9, 1.1),
                "approach_lanes": float(site_lanes), "site_ratio_to_typical": float(rng.normal(1, 0.1)),
                "queue_ratio_now": float(rng.normal(0.45 if looks_backed_up else 0.95, 0.1)),
                "backed_up_now": looks_backed_up,
                "slowed_minutes_so_far": float(min(total, lm - 5) * rng.uniform(0.3, 1.0) + 5 * (rng.random() < 0.2)),
                "has_backup": has_backup, "total_backup_minutes": total,
                "remaining_minutes": remaining, "is_censored": censored and remaining > 0,
                "label_adjusted_share": float(rng.choice([1.0, 0.5])),
            })
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ["max_lanes_blocked", "n_notes", "units_assigned", "vehicles", "landmark_min", "site_lanes"]:
        df[c] = df[c].astype("Int64")
    for c in ["ever_blocking", "blocking_now", "all_lanes", "tow", "chp_closed", "backed_up_now",
              "is_censored", "has_backup", "lanes_reopened", "ambulance"]:
        df[c] = df[c].astype("boolean")
    for c in ["dispatch_id", "district", "freeway", "direction", "day_type", "type_code"]:
        df[c] = df[c].astype("string")
    return df


@pytest.fixture(scope="session")
def raw():
    return fake_landmarks()


@pytest.fixture(scope="session")
def data(raw):
    from backup_model import prepare
    return prepare(raw)
