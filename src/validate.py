"""Stage 2: data quality checks. Exits non-zero (stopping the pipeline) on failure."""

from __future__ import annotations

import sys

import pandas as pd

from src.schema import BINARY_COLUMNS, MERCHANT_CATEGORIES, RAW_COLUMNS, TARGET
from src.utils import BATCH_FILE, RAW_CURRENT, RAW_REFERENCE, REPORTS_DIR, get_logger, load_params, write_json

log = get_logger("validate")


def validate_frame(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Return a list of human-readable problems (empty = valid)."""
    errors: list[str] = []

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    for col, kind in RAW_COLUMNS.items():
        if df[col].dtype.kind != kind:
            errors.append(f"{col}: expected dtype kind {kind!r}, got {df[col].dtype}")

    if len(df) < cfg["min_rows"]:
        errors.append(f"only {len(df)} rows, need at least {cfg['min_rows']}")

    nulls = df[list(RAW_COLUMNS)].isna().sum()
    for col, n in nulls[nulls > 0].items():
        errors.append(f"{col}: {n} null values")

    if df["transaction_id"].duplicated().any():
        errors.append(f"{int(df['transaction_id'].duplicated().sum())} duplicate transaction_id values")

    for col in BINARY_COLUMNS:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            errors.append(f"{col}: {int(bad.sum())} values outside {{0, 1}}")

    unknown = sorted(set(df["merchant_category"].dropna()) - set(MERCHANT_CATEGORIES))
    if unknown:
        errors.append(f"merchant_category: unknown values {unknown}")

    for col in ["amount", "avg_amount_30d", "distance_from_home_km", "account_age_days", "txn_count_24h"]:
        if (df[col] < 0).any():
            errors.append(f"{col}: negative values")

    if not df["customer_age"].between(18, 120).all():
        errors.append("customer_age: values outside 18..120")

    rate = float(df[TARGET].mean()) if len(df) else 0.0
    lo, hi = cfg["fraud_rate_range"]
    if not lo <= rate <= hi:
        errors.append(f"fraud rate {rate:.4f} outside expected range [{lo}, {hi}]")

    return errors


def main() -> None:
    cfg = load_params()["validation"]
    report: dict = {}
    failed = False
    for name, folder in [("current", RAW_CURRENT), ("reference", RAW_REFERENCE)]:
        df = pd.read_parquet(folder / BATCH_FILE)
        errors = validate_frame(df, cfg)
        report[name] = {
            "rows": len(df),
            "fraud_rate": round(float(df[TARGET].mean()), 6) if TARGET in df else None,
            "passed": not errors,
            "errors": errors,
        }
        for e in errors:
            log.error("%s: %s", name, e)
        failed |= bool(errors)
    write_json(REPORTS_DIR / "validation.json", report)
    if failed:
        sys.exit("data validation failed, see reports/validation.json")
    log.info("validation passed: %s", {k: v["rows"] for k, v in report.items()})


if __name__ == "__main__":
    main()
