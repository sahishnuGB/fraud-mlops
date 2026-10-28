"""Stage 5: train the classifier.

params.yaml selects the model (``hgb`` = HistGradientBoosting, ``logreg``) and the
imbalance strategy (``none``, ``undersample``, ``smote``). Resampling sits inside the
imbalanced-learn pipeline, so it is only applied during ``fit`` (never to test data).
"""

from __future__ import annotations

import time

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.schema import TARGET
from src.utils import FEATURES_DIR, MODEL_PATH, MODELS_DIR, get_logger, load_params

log = get_logger("train")

SAMPLING_CHOICES = ("none", "undersample", "smote")
MODEL_CHOICES = ("hgb", "logreg")


def make_sampler(params: dict):
    kind = params["sampling"]
    cfg = params["sampling_params"]
    seed = params["seed"]
    if kind == "none":
        return None
    if kind == "undersample":
        return RandomUnderSampler(sampling_strategy=cfg["ratio"], random_state=seed)
    if kind == "smote":
        return SMOTE(sampling_strategy=cfg["ratio"], k_neighbors=cfg["smote_k_neighbors"], random_state=seed)
    raise ValueError(f"sampling must be one of {SAMPLING_CHOICES}, got {kind!r}")


def make_classifier(params: dict):
    kind = params["model"]["type"]
    seed = params["seed"]
    if kind == "hgb":
        cfg = params["model"]["hgb"]
        return HistGradientBoostingClassifier(
            learning_rate=cfg["learning_rate"],
            max_iter=cfg["max_iter"],
            max_leaf_nodes=cfg["max_leaf_nodes"],
            min_samples_leaf=cfg["min_samples_leaf"],
            l2_regularization=cfg["l2_regularization"],
            class_weight=cfg["class_weight"],
            early_stopping=cfg["early_stopping"],
            random_state=seed,
        )
    if kind == "logreg":
        cfg = params["model"]["logreg"]
        return LogisticRegression(
            C=cfg["C"], max_iter=cfg["max_iter"], class_weight=cfg["class_weight"], random_state=seed
        )
    raise ValueError(f"model.type must be one of {MODEL_CHOICES}, got {kind!r}")


def build_pipeline(params: dict) -> Pipeline:
    steps = []
    if params["model"]["type"] == "logreg":
        steps.append(("scale", StandardScaler()))
    sampler = make_sampler(params)
    if sampler is not None:
        steps.append(("sample", sampler))
    steps.append(("clf", make_classifier(params)))
    return Pipeline(steps)


def train(df: pd.DataFrame, params: dict) -> dict:
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pipe = build_pipeline(params)
    start = time.perf_counter()
    pipe.fit(X, y)
    log.info(
        "trained %s (sampling=%s) on %d rows in %.1fs",
        params["model"]["type"],
        params["sampling"],
        len(X),
        time.perf_counter() - start,
    )
    return {
        "pipeline": pipe,
        "features": list(X.columns),
        "model_type": params["model"]["type"],
        "sampling": params["sampling"],
        "batch": params["batch"],
    }


def main() -> None:
    params = load_params()
    bundle = train(pd.read_parquet(FEATURES_DIR / "train.parquet"), params)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    log.info("saved %s", MODEL_PATH)


if __name__ == "__main__":
    main()
