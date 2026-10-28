"""FastAPI service that scores raw transactions with the trained model.

    uvicorn src.serve:app --port 8000          # from the repo, after `dvc repro`
    docker run --rm -p 8000:8000 fraud-mlops   # from the Docker image (model baked in)

Interactive docs: http://localhost:8000/docs

Requests use the raw transaction schema; features are built with the same
stateless ``build_features`` the training pipeline uses, so there is no
train/serve skew.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.featurize import build_features
from src.schema import MERCHANT_CATEGORIES
from src.utils import MODELS_DIR, REPORTS_DIR, read_json

DEFAULT_MODEL_PATH = MODELS_DIR / "model.joblib"
DEFAULT_METRICS_PATH = REPORTS_DIR / "metrics.json"

MerchantCategory = Literal[tuple(MERCHANT_CATEGORIES)]  # type: ignore[valid-type]

EXAMPLE = {
    "transaction_id": "demo-0001",
    "timestamp": "2026-04-18T02:41:00",
    "amount": 1249.99,
    "merchant_category": "electronics",
    "card_present": False,
    "is_foreign": False,
    "distance_from_home_km": 412.5,
    "customer_age": 27,
    "account_age_days": 21,
    "txn_count_24h": 7,
    "avg_amount_30d": 85.4,
    "device_change": True,
}


class Transaction(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE]})

    transaction_id: str | None = None
    timestamp: datetime
    amount: float = Field(ge=0)
    merchant_category: MerchantCategory
    card_present: bool
    is_foreign: bool
    distance_from_home_km: float = Field(ge=0)
    customer_age: int = Field(ge=18, le=120)
    account_age_days: int = Field(ge=0)
    txn_count_24h: int = Field(ge=0)
    avg_amount_30d: float = Field(ge=0)
    device_change: bool


class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    transaction_id: str | None
    fraud_probability: float
    is_fraud: bool


class PredictResponse(BaseModel):
    model_batch: str
    threshold: float
    predictions: list[Prediction]


def create_app(model_path: Path | None = None, metrics_path: Path | None = None) -> FastAPI:
    model_path = Path(model_path or os.environ.get("FRAUD_MODEL_PATH") or DEFAULT_MODEL_PATH)
    metrics_path = Path(metrics_path or os.environ.get("FRAUD_METRICS_PATH") or DEFAULT_METRICS_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = joblib.load(model_path) if model_path.exists() else None
        app.state.metrics = read_json(metrics_path) if metrics_path.exists() else {}
        yield

    app = FastAPI(
        title="Fraud detection API",
        description="Scores credit card transactions with the model trained by the DVC pipeline.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def bundle() -> dict:
        if app.state.bundle is None:
            raise HTTPException(503, f"no model loaded from {model_path}; run `dvc repro` or `dvc pull` first")
        return app.state.bundle

    def default_threshold() -> float:
        # Best-F1 threshold from evaluation: a better operating point than 0.5 for rare fraud.
        m = app.state.metrics
        return float(m.get("best_f1_threshold", m.get("threshold", 0.5)))

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": app.state.bundle is not None}

    @app.get("/model")
    def model_info() -> dict:
        b = bundle()
        m = app.state.metrics
        return {
            "model_type": b["model_type"],
            "sampling": b["sampling"],
            "trained_on_batch": b["batch"],
            "features": b["features"],
            "default_threshold": default_threshold(),
            "metrics": {k: m[k] for k in ("pr_auc", "roc_auc", "best_f1", "recall_at_precision_90") if k in m},
        }

    @app.post("/predict")
    def predict(
        request: PredictRequest,
        threshold: float | None = Query(None, ge=0, le=1, description="defaults to the model's best-F1 threshold"),
    ) -> PredictResponse:
        b = bundle()
        cut = default_threshold() if threshold is None else threshold
        raw = pd.DataFrame([t.model_dump() for t in request.transactions])
        for col in ("card_present", "is_foreign", "device_change"):
            raw[col] = raw[col].astype(int)
        scores = b["pipeline"].predict_proba(build_features(raw)[b["features"]])[:, 1]
        return PredictResponse(
            model_batch=b["batch"],
            threshold=cut,
            predictions=[
                Prediction(
                    transaction_id=t.transaction_id,
                    fraud_probability=round(float(s), 6),
                    is_fraud=bool(s >= cut),
                )
                for t, s in zip(request.transactions, scores, strict=True)
            ],
        )

    return app


app = create_app()
