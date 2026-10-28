import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src.featurize import build_features
from src.serve import create_app
from src.train import train
from src.utils import ROOT, load_params, write_json

EXAMPLES = json.loads((ROOT / "examples" / "predict_request.json").read_text())


@pytest.fixture(scope="module")
def client(tmp_path_factory, batch_df):
    params = load_params()
    tmp = tmp_path_factory.mktemp("serve")
    joblib.dump(train(build_features(batch_df), params), tmp / "model.joblib")
    write_json(tmp / "metrics.json", {"pr_auc": 0.6, "roc_auc": 0.97, "best_f1_threshold": 0.3})
    with TestClient(create_app(tmp / "model.joblib", tmp / "metrics.json")) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model").json()
    assert info["model_type"] == "hgb"
    assert info["default_threshold"] == 0.3
    assert "amount_ratio_log" in info["features"]


def test_predict_examples(client):
    res = client.post("/predict", json=EXAMPLES)
    assert res.status_code == 200
    body = res.json()
    assert body["threshold"] == 0.3
    by_id = {p["transaction_id"]: p for p in body["predictions"]}
    assert all(0.0 <= p["fraud_probability"] <= 1.0 for p in by_id.values())
    assert by_id["demo-suspicious"]["fraud_probability"] > by_id["demo-legit"]["fraud_probability"]
    assert not by_id["demo-legit"]["is_fraud"]


def test_threshold_override(client):
    body = client.post("/predict", params={"threshold": 0.0}, json=EXAMPLES).json()
    assert body["threshold"] == 0.0
    assert all(p["is_fraud"] for p in body["predictions"])


@pytest.mark.parametrize(
    "field,value",
    [("merchant_category", "casino"), ("amount", -1), ("customer_age", 12), ("timestamp", "yesterday")],
)
def test_invalid_input_is_rejected(client, field, value):
    bad = json.loads(json.dumps(EXAMPLES))
    bad["transactions"][0][field] = value
    assert client.post("/predict", json=bad).status_code == 422


def test_empty_request_is_rejected(client):
    assert client.post("/predict", json={"transactions": []}).status_code == 422


def test_missing_model_returns_503(tmp_path):
    with TestClient(create_app(tmp_path / "nope.joblib", tmp_path / "nope.json")) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=EXAMPLES).status_code == 503
