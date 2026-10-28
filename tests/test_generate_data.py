import pandas as pd
import pytest

from src.generate_data import generate_batch, main
from src.schema import RAW_COLUMNS
from src.validate import validate_frame


def test_deterministic():
    a = generate_batch("2026-03", rows=2000)
    b = generate_batch("2026-03", rows=2000)
    pd.testing.assert_frame_equal(a, b)


def test_batches_differ():
    a = generate_batch("2026-03", rows=2000)
    b = generate_batch("2026-04", rows=2000)
    assert not a["amount"].equals(b["amount"])


def test_schema_and_validity(batch_df, params):
    assert list(batch_df.columns) == list(RAW_COLUMNS)
    assert validate_frame(batch_df, params["validation"]) == []
    assert batch_df["timestamp"].is_monotonic_increasing
    assert (batch_df["timestamp"].dt.strftime("%Y-%m") == "2026-01").all()


def test_fraud_rate_close_to_target(batch_df):
    assert 0.01 < batch_df["is_fraud"].mean() < 0.03


def test_drift_shifts_distributions(batch_df, drifted_df):
    assert drifted_df["amount"].median() > batch_df["amount"].median() * 1.15
    assert drifted_df["card_present"].mean() < batch_df["card_present"].mean()
    assert drifted_df["is_fraud"].mean() > batch_df["is_fraud"].mean()


def test_invalid_args():
    with pytest.raises(ValueError):
        generate_batch("2026-13")
    with pytest.raises(ValueError):
        generate_batch("2026-01", drift=1.5)


def test_cli_uploads_to_local_bucket(offline_storage):
    main(["--batch", "2026-05", "2026-06", "--rows", "500", "--drift"])
    for b in ["2026-05", "2026-06"]:
        path = offline_storage / f"raw/batch={b}/transactions.parquet"
        assert path.exists()
        assert len(pd.read_parquet(path)) == 500
