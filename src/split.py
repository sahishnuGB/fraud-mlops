"""Stage 3: stratified train/test split of the current batch."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.schema import TARGET
from src.utils import BATCH_FILE, RAW_CURRENT, SPLIT_DIR, get_logger, load_params

log = get_logger("split")


def split_frame(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    params = load_params()
    df = pd.read_parquet(RAW_CURRENT / BATCH_FILE)
    train, test = split_frame(df, params["split"]["test_size"], params["seed"])
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(SPLIT_DIR / "train.parquet", index=False)
    test.to_parquet(SPLIT_DIR / "test.parquet", index=False)
    log.info(
        "train %d rows (%d fraud), test %d rows (%d fraud)",
        len(train),
        train[TARGET].sum(),
        len(test),
        test[TARGET].sum(),
    )


if __name__ == "__main__":
    main()
