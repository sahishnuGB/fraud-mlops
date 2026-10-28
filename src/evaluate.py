"""Stage 6: score the held-out test set, write metrics/plots and the PR-AUC gate result."""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.schema import TARGET  # noqa: E402
from src.train import MODEL_PATH  # noqa: E402
from src.utils import FEATURES_DIR, PLOTS_DIR, REPORTS_DIR, get_logger, load_params, write_json  # noqa: E402

log = get_logger("evaluate")


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, min_pr_auc: float) -> dict:
    pred = (scores >= threshold).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    at_p90 = recall[precision >= 0.9]
    pr_auc = float(average_precision_score(y_true, scores))
    return {
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 6),
        "threshold": threshold,
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 6),
        "best_f1": round(float(f1[best]), 6),
        "best_f1_threshold": round(float(thresholds[best]), 6) if len(thresholds) else threshold,
        "recall_at_precision_90": round(float(at_p90.max()) if at_p90.size else 0.0, 6),
        "test_rows": int(len(y_true)),
        "test_fraud": int(y_true.sum()),
        "min_pr_auc": min_pr_auc,
        "gate_passed": bool(pr_auc >= min_pr_auc),
    }


def plot_pr_curve(y_true, scores, pr_auc: float, path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"model (AP={pr_auc:.3f})")
    ax.axhline(y_true.mean(), ls="--", color="grey", label=f"baseline ({y_true.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(y_true, pred, path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, pred), display_labels=["legit", "fraud"]).plot(
        ax=ax, colorbar=False, values_format="d"
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    params = load_params()
    bundle = joblib.load(MODEL_PATH)
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    y = test[TARGET].to_numpy()
    scores = bundle["pipeline"].predict_proba(test[bundle["features"]])[:, 1]

    threshold = params["evaluate"]["threshold"]
    metrics = compute_metrics(y, scores, threshold, params["min_pr_auc"])
    metrics.update(batch=params["batch"], model_type=bundle["model_type"], sampling=bundle["sampling"])
    write_json(REPORTS_DIR / "metrics.json", metrics)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_pr_curve(y, scores, metrics["pr_auc"], PLOTS_DIR / "pr_curve.png")
    plot_confusion(y, (scores >= threshold).astype(int), PLOTS_DIR / "confusion_matrix.png")

    status = "PASSED" if metrics["gate_passed"] else "FAILED"
    log.info(
        "PR-AUC %.4f (gate >= %.2f %s), ROC-AUC %.4f",
        metrics["pr_auc"],
        params["min_pr_auc"],
        status,
        metrics["roc_auc"],
    )


if __name__ == "__main__":
    main()
