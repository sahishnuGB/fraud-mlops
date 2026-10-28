"""Publish the candidate model to ``models/prod/`` if the promote stage approved it.

The current production model is first copied to ``models/archive/<batch>-<time>/``
(the S3 lifecycle rule expires archives after 30 days).

    python -m src.publish          # respects reports/promotion.json
    python -m src.publish --force  # publish even if not approved (gate must still pass)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from src import storage
from src.promote import PROD_PREFIX, fetch_prod_metrics
from src.train import MODEL_PATH
from src.utils import REPORTS_DIR, ROOT, get_logger, read_json, write_json

log = get_logger("publish")


def git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    decision = read_json(REPORTS_DIR / "promotion.json")
    metrics = read_json(REPORTS_DIR / "metrics.json")
    if not metrics["gate_passed"]:
        sys.exit(f"refusing to publish: PR-AUC {metrics['pr_auc']} is below the gate {metrics['min_pr_auc']}")
    if not decision["promote"] and not args.force:
        log.info("not publishing: %s", decision["reason"])
        return

    now = datetime.now(UTC)
    prod = fetch_prod_metrics()
    if prod is not None:
        archive = f"models/archive/{prod.get('batch', 'unknown')}-{now:%Y%m%dT%H%M%SZ}/"
        storage.copy_prefix(PROD_PREFIX, archive)
        log.info("archived previous prod model to %s", storage.uri(archive))

    metadata = {
        "published_at": now.isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "batch": metrics["batch"],
        "model_type": metrics["model_type"],
        "sampling": metrics["sampling"],
        "pr_auc": metrics["pr_auc"],
        "reason": decision["reason"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        meta_path = Path(tmp) / "metadata.json"
        write_json(meta_path, metadata)
        # metrics.json last: promote reads it, so it should only appear once the model is in place.
        for src, name in [
            (MODEL_PATH, "model.joblib"),
            (REPORTS_DIR / "drift.json", "drift.json"),
            (meta_path, "metadata.json"),
            (REPORTS_DIR / "metrics.json", "metrics.json"),
        ]:
            storage.upload(src, PROD_PREFIX + name)
    log.info(
        "published model for batch %s (PR-AUC %.4f) to %s",
        metrics["batch"],
        metrics["pr_auc"],
        storage.uri(PROD_PREFIX),
    )


if __name__ == "__main__":
    main()
