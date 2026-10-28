"""Stage 1: pull the current and reference batches from object storage.

aws s3 sync s3://$S3_BUCKET/raw/batch=YYYY-MM/ data/raw/current/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src import storage
from src.utils import BATCH_FILE, RAW_CURRENT, RAW_REFERENCE, batch_prefix, get_logger, load_params

log = get_logger("ingest")


def ingest_batch(batch: str, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    prefix = batch_prefix(batch)
    storage.sync_down(prefix, dest)
    path = dest / BATCH_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No {BATCH_FILE} for batch {batch} at {storage.uri(prefix)}. "
            f"Generate it with: python -m src.generate_data --batch {batch}"
        )
    log.info("ingested %s -> %s", storage.uri(prefix), dest)
    return path


def main() -> None:
    params = load_params()
    log.info("storage backend: %s", storage.describe())
    ingest_batch(params["batch"], RAW_CURRENT)
    ingest_batch(params["reference_batch"], RAW_REFERENCE)


if __name__ == "__main__":
    main()
