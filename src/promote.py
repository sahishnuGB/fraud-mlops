"""Stage 8: decide whether the candidate model should replace production.

The production model (``models/prod/`` in object storage) is re-scored on the
candidate's test set, so both PR-AUCs come from the same data. If the prod model
cannot be loaded, the PR-AUC stored in ``models/prod/metrics.json`` is used instead.
The decision goes to ``reports/promotion.json``. Nothing is published here; that is
``python -m src.publish`` (run by CI on main only).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

from src import storage
from src.schema import TARGET
from src.utils import (
    FEATURES_DIR,
    PROD_METRICS_KEY,
    PROD_MODEL_KEY,
    PROD_PREFIX,
    REPORTS_DIR,
    get_logger,
    load_params,
    read_json,
    write_json,
)

log = get_logger("promote")


def fetch_prod_metrics() -> dict | None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "metrics.json"
        if not storage.download(PROD_METRICS_KEY, dest):
            return None
        return read_json(dest)


def score_prod_on(test: pd.DataFrame) -> float | None:
    """PR-AUC of the production model on ``test``, or None if it is unavailable."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "model.joblib"
        try:
            if not storage.download(PROD_MODEL_KEY, dest):
                return None
            bundle = joblib.load(dest)
            scores = bundle["pipeline"].predict_proba(test[bundle["features"]])[:, 1]
        except Exception as exc:  # incompatible pickle, missing features, ...
            log.warning("could not re-score prod model (%s); falling back to its stored metrics", exc)
            return None
    return round(float(average_precision_score(test[TARGET], scores)), 6)


def decide(candidate: dict, prod: dict | None, min_improvement: float) -> dict:
    """``prod`` holds ``pr_auc`` (ideally on the candidate's test set) and optional ``batch``."""
    cand_auc = candidate["pr_auc"]
    if not candidate["gate_passed"]:
        promote, reason = False, f"candidate PR-AUC {cand_auc:.4f} is below the gate {candidate['min_pr_auc']}"
    elif prod is None:
        promote, reason = True, "no production model yet"
    elif cand_auc >= prod["pr_auc"] + min_improvement:
        promote, reason = True, f"candidate PR-AUC {cand_auc:.4f} >= prod {prod['pr_auc']:.4f} + {min_improvement}"
    else:
        promote, reason = False, f"candidate PR-AUC {cand_auc:.4f} < prod {prod['pr_auc']:.4f} + {min_improvement}"
    return {
        "promote": promote,
        "reason": reason,
        "candidate_pr_auc": cand_auc,
        "prod_pr_auc": prod["pr_auc"] if prod else None,
        "prod_batch": prod.get("batch") if prod else None,
        "delta_pr_auc": round(cand_auc - prod["pr_auc"], 6) if prod else None,
    }


def main() -> None:
    params = load_params()
    candidate = read_json(REPORTS_DIR / "metrics.json")
    prod_metrics = fetch_prod_metrics()
    prod, rescored = None, None
    if prod_metrics is not None:
        rescored = score_prod_on(pd.read_parquet(FEATURES_DIR / "test.parquet"))
        prod = {
            "pr_auc": rescored if rescored is not None else prod_metrics["pr_auc"],
            "batch": prod_metrics.get("batch"),
        }
    decision = decide(candidate, prod, params["promote"]["min_improvement"])
    decision["batch"] = params["batch"]
    decision["prod_compared_on"] = (
        None if prod is None else "candidate test set" if rescored is not None else "prod's own test set"
    )
    decision["prod_pr_auc_at_publish"] = prod_metrics["pr_auc"] if prod_metrics else None
    decision["drift_detected"] = read_json(REPORTS_DIR / "drift.json")["drift_detected"]
    write_json(REPORTS_DIR / "promotion.json", decision)
    log.info("promote=%s: %s (prod source: %s)", decision["promote"], decision["reason"], storage.uri(PROD_PREFIX))


if __name__ == "__main__":
    main()
