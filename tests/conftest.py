import copy

import pytest

from src.generate_data import generate_batch
from src.utils import load_params


@pytest.fixture(autouse=True)
def offline_storage(tmp_path, monkeypatch):
    """Every test uses an empty local bucket, never real S3."""
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("LOCAL_S3_DIR", str(tmp_path / "bucket"))
    return tmp_path / "bucket"


@pytest.fixture(scope="session")
def batch_df():
    return generate_batch("2026-01", rows=6000)


@pytest.fixture(scope="session")
def drifted_df():
    return generate_batch("2026-02", rows=6000, drift=1.0)


@pytest.fixture
def params():
    return copy.deepcopy(load_params())
