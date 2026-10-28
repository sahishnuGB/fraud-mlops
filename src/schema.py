"""Single source of truth for the raw transaction schema."""

from __future__ import annotations

import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery",
    "restaurant",
    "gas",
    "online_retail",
    "electronics",
    "travel",
    "entertainment",
    "other",
]

# column -> pandas dtype kind ("i" int, "f" float, "O" object/string, "M" datetime)
RAW_COLUMNS: dict[str, str] = {
    "transaction_id": "O",
    "timestamp": "M",
    "amount": "f",
    "merchant_category": "O",
    "card_present": "i",
    "is_foreign": "i",
    "distance_from_home_km": "f",
    "customer_age": "i",
    "account_age_days": "i",
    "txn_count_24h": "i",
    "avg_amount_30d": "f",
    "device_change": "i",
    "is_fraud": "i",
}

TARGET = "is_fraud"
BINARY_COLUMNS = ["card_present", "is_foreign", "device_change", TARGET]

# Features monitored for drift (raw values, plus the derived transaction hour).
DRIFT_NUMERIC = [
    "amount",
    "avg_amount_30d",
    "distance_from_home_km",
    "customer_age",
    "account_age_days",
    "txn_count_24h",
    "hour",
]
DRIFT_CATEGORICAL = ["merchant_category", "card_present", "is_foreign", "device_change"]


def monitoring_view(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns plus derived values that are useful to monitor."""
    out = df.copy()
    out["hour"] = pd.to_datetime(out["timestamp"]).dt.hour
    return out
