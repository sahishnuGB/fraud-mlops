"""Stage 4: stateless feature engineering.

Nothing here is fitted on data, so the same function can run at serving time.
Scaling (for logreg) lives inside the trained model pipeline instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import MERCHANT_CATEGORIES, TARGET
from src.utils import FEATURES_DIR, SPLIT_DIR, get_logger

log = get_logger("featurize")

PASSTHROUGH = ["card_present", "is_foreign", "device_change", "customer_age", "txn_count_24h"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60
    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log1p(df["amount"])
    out["log_avg_amount_30d"] = np.log1p(df["avg_amount_30d"])
    out["amount_ratio_log"] = np.log((df["amount"] + 1) / (df["avg_amount_30d"] + 1))
    out["log_distance"] = np.log1p(df["distance_from_home_km"])
    out["log_account_age"] = np.log1p(df["account_age_days"])
    out["new_account"] = (df["account_age_days"] < 60).astype(np.int8)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = (ts.dt.hour < 6).astype(np.int8)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)
    for col in PASSTHROUGH:
        out[col] = df[col]
    # Fixed category list => identical columns for every batch, even if a category is absent.
    for cat in MERCHANT_CATEGORIES:
        out[f"cat_{cat}"] = (df["merchant_category"] == cat).astype(np.int8)
    if TARGET in df:
        out[TARGET] = df[TARGET]
    return out


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["train", "test"]:
        feats = build_features(pd.read_parquet(SPLIT_DIR / f"{name}.parquet"))
        feats.to_parquet(FEATURES_DIR / f"{name}.parquet", index=False)
        log.info("%s: %d rows x %d features", name, len(feats), feats.shape[1] - 1)


if __name__ == "__main__":
    main()
