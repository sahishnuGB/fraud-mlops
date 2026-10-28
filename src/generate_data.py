"""Deterministic synthetic credit card transactions, one batch per month.

The same (batch, seed, rows, drift) always produces the same data, so batches can be
regenerated anywhere instead of being downloaded.

Fraud labels come from a hidden logistic risk model over the features, so a classifier
can learn real signal. ``--drift`` (0..1) shifts the feature distributions (covariate
drift: inflation, more online / night / foreign activity, younger customers) and the
fraud pattern itself (concept drift: device changes matter more, foreign less).

Examples:
    python -m src.generate_data --batch 2026-01 2026-02 2026-03
    python -m src.generate_data --batch 2026-06 --drift 0.8
    python -m src.generate_data --batch 2026-01 --no-upload --out-dir /tmp/batches
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import storage
from src.schema import MERCHANT_CATEGORIES, RAW_COLUMNS
from src.utils import BATCH_FILE, batch_prefix, check_batch, get_logger

log = get_logger("generate_data")

DEFAULT_ROWS = 30_000
DEFAULT_FRAUD_RATE = 0.02
DEFAULT_SEED = 42

_CAT_PROBS = np.array([0.24, 0.16, 0.12, 0.18, 0.07, 0.06, 0.08, 0.09])
_CAT_DRIFT = np.array([-0.08, 0.0, -0.04, 0.10, 0.03, 0.0, 0.0, -0.01])
_CAT_AMOUNT = np.array([0.8, 0.6, 0.7, 1.2, 3.0, 4.0, 0.9, 1.0])
_CAT_CARD_PRESENT = np.array([0.95, 0.90, 0.97, 0.03, 0.45, 0.35, 0.80, 0.70])
_HIGH_RISK_CATS = {"online_retail", "electronics"}
_SIGNAL = 2.5  # sharpness of the hidden risk model (higher = more separable)
_NOISE = 0.7  # std of the unobserved risk component


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _calibrate_intercept(z: np.ndarray, target_rate: float) -> float:
    """Bisection for b such that mean(sigmoid(z + b)) == target_rate."""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_batch(
    batch: str,
    rows: int = DEFAULT_ROWS,
    drift: float = 0.0,
    seed: int = DEFAULT_SEED,
    fraud_rate: float = DEFAULT_FRAUD_RATE,
) -> pd.DataFrame:
    batch = check_batch(batch)
    if not 0.0 <= drift <= 1.0:
        raise ValueError("drift must be between 0 and 1")
    year, month = (int(x) for x in batch.split("-"))
    rng = np.random.default_rng([seed, year, month, rows, round(drift * 1000)])
    d = drift

    # --- customers & merchants -------------------------------------------------
    customer_age = np.clip(rng.normal(42 - 6 * d, 13, rows), 18, 90).round().astype(np.int64)
    account_age_days = np.clip(rng.exponential(900 * (1 - 0.3 * d), rows), 1, 7300).astype(np.int64)

    cat_p = np.clip(_CAT_PROBS + d * _CAT_DRIFT, 0.001, None)
    cat_idx = rng.choice(len(MERCHANT_CATEGORIES), size=rows, p=cat_p / cat_p.sum())
    merchant_category = np.array(MERCHANT_CATEGORIES, dtype=object)[cat_idx]

    card_present_p = np.clip(_CAT_CARD_PRESENT[cat_idx] - 0.10 * d, 0.0, 1.0)
    card_present = (rng.random(rows) < card_present_p).astype(np.int64)

    # --- amounts ----------------------------------------------------------------
    inflation = 1 + 0.30 * d
    avg_amount_30d = rng.lognormal(np.log(55), 0.55, rows) * _CAT_AMOUNT[cat_idx] * inflation
    amount_ratio = rng.lognormal(0.0, 0.65, rows)
    amount = avg_amount_30d * amount_ratio

    # --- time, location, behaviour ---------------------------------------------
    night = rng.random(rows) < 0.08 + 0.08 * d
    hour = np.where(night, rng.integers(0, 6, rows), np.clip(rng.normal(14, 3.5, rows), 6, 23.99)).astype(int)
    days_in_month = pd.Period(batch, freq="M").days_in_month
    start = pd.Timestamp(f"{batch}-01")
    offsets = rng.integers(0, days_in_month, rows) * 86_400 + hour * 3_600 + rng.integers(0, 3_600, rows)
    timestamp = start + pd.to_timedelta(np.sort(offsets), unit="s")
    order = np.argsort(offsets, kind="stable")  # keep rows time-ordered

    travel = merchant_category == "travel"
    is_foreign = (rng.random(rows) < np.where(travel, 0.30, 0.04 + 0.05 * d)).astype(np.int64)
    distance = rng.lognormal(np.log(8), 1.0, rows) + is_foreign * rng.uniform(500, 6000, rows)
    txn_count_24h = rng.poisson(2.5 + 1.0 * d, rows).astype(np.int64)
    device_change = (rng.random(rows) < 0.04 + 0.04 * d).astype(np.int64)

    # --- hidden fraud model -----------------------------------------------------
    is_night = (hour < 6).astype(float)
    high_risk = np.isin(merchant_category, list(_HIGH_RISK_CATS)).astype(float)
    log_ratio = np.log(amount_ratio)
    not_present = 1 - card_present
    # Thresholds and interactions (not just a linear score), like real fraud patterns.
    z = _SIGNAL * (
        2.2 * np.maximum(log_ratio - 0.3, 0)  # only unusually large purchases are risky
        + 0.6 * is_night
        + 1.2 * is_night * not_present
        + (1.4 - 0.7 * d) * is_foreign
        + 0.7 * not_present
        + (1.8 + 0.8 * d) * device_change
        + 1.2 * (txn_count_24h >= 6)  # velocity spike
        + 0.9 * (account_age_days < 60)
        + 0.6 * high_risk * (log_ratio > 0.5)
        + 0.8 * (distance > 150) * (1 - is_foreign)
        - 0.01 * (customer_age - 42)
    ) + rng.normal(0, _NOISE, rows)  # unobserved factors
    target = fraud_rate * (1 + 0.5 * d)
    p_fraud = _sigmoid(z + _calibrate_intercept(z, target))
    is_fraud = (rng.random(rows) < p_fraud).astype(np.int64)

    df = (
        pd.DataFrame(
            {
                "amount": np.round(amount, 2),
                "merchant_category": merchant_category,
                "card_present": card_present,
                "is_foreign": is_foreign,
                "distance_from_home_km": np.round(distance, 2),
                "customer_age": customer_age,
                "account_age_days": account_age_days,
                "txn_count_24h": txn_count_24h,
                "avg_amount_30d": np.round(avg_amount_30d, 2),
                "device_change": device_change,
                "is_fraud": is_fraud,
            }
        )
        .iloc[order]
        .reset_index(drop=True)
    )
    df.insert(0, "timestamp", timestamp)
    df.insert(0, "transaction_id", [f"{batch}-{i:07d}" for i in range(rows)])
    return df[list(RAW_COLUMNS)]


def write_batch(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", nargs="+", required=True, help="one or more months, YYYY-MM")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument(
        "--drift",
        type=float,
        nargs="?",
        const=1.0,
        default=0.0,
        help="drift severity 0..1 (flag alone = 1.0)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fraud-rate", type=float, default=DEFAULT_FRAUD_RATE)
    parser.add_argument("--no-upload", action="store_true", help="only write files to --out-dir")
    parser.add_argument("--out-dir", type=Path, default=None, help="also keep a local copy here")
    args = parser.parse_args(argv)

    for batch in args.batch:
        df = generate_batch(batch, rows=args.rows, drift=args.drift, seed=args.seed, fraud_rate=args.fraud_rate)
        key = batch_prefix(batch) + BATCH_FILE
        with tempfile.TemporaryDirectory() as tmp:
            out = (args.out_dir / key) if args.out_dir else Path(tmp) / BATCH_FILE
            write_batch(df, out)
            if not args.no_upload:
                storage.upload(out, key)
        dest = "not uploaded" if args.no_upload else storage.uri(key)
        log.info(
            "batch %s: %d rows, fraud rate %.3f%%, drift %.2f -> %s",
            batch,
            len(df),
            100 * df["is_fraud"].mean(),
            args.drift,
            dest,
        )


if __name__ == "__main__":
    main()
