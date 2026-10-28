"""Download the production model from the registry (``models/prod/``) into the workspace.

Used by the deploy workflow so the image always ships the *promoted* model, never an
unpromoted candidate from the latest training run.

    python -m src.fetch_prod     # -> models/model.joblib, reports/metrics.json
"""

from __future__ import annotations

import os
import sys

from src import storage

# Only lightweight imports: the deploy job installs just PyYAML and the AWS CLI.
from src.utils import MODEL_PATH, PROD_METRICS_KEY, PROD_MODEL_KEY, PROD_PREFIX, REPORTS_DIR, get_logger, read_json

log = get_logger("fetch_prod")


def main() -> None:
    metrics_path = REPORTS_DIR / "metrics.json"
    for key, dest in [(PROD_MODEL_KEY, MODEL_PATH), (PROD_METRICS_KEY, metrics_path)]:
        if not storage.download(key, dest):
            sys.exit(f"no production model at {storage.uri(PROD_PREFIX)}: run the train workflow on main first")
    batch = read_json(metrics_path)["batch"]
    log.info("fetched prod model (batch %s) from %s", batch, storage.uri(PROD_PREFIX))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"batch={batch}\n")


if __name__ == "__main__":
    main()
