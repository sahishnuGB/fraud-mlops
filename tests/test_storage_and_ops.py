import joblib
import pandas as pd
import pytest

from src import storage
from src.bump_params import main as bump_main
from src.bump_params import set_top_level
from src.featurize import build_features
from src.find_latest_batch import available_batches
from src.generate_data import generate_batch, write_batch
from src.ingest import ingest_batch
from src.promote import decide, fetch_prod_metrics, score_prod_on
from src.train import train
from src.utils import write_json


def test_local_fallback_roundtrip(tmp_path, offline_storage):
    assert not storage.is_remote()
    src = tmp_path / "x.json"
    write_json(src, {"a": 1})
    storage.upload(src, "models/prod/metrics.json")
    assert storage.exists("models/prod/metrics.json")
    assert not storage.exists("models/prod/nope.json")
    assert fetch_prod_metrics() == {"a": 1}
    storage.copy_prefix("models/prod/", "models/archive/1/")
    assert (offline_storage / "models/archive/1/metrics.json").exists()


def test_s3_mode_builds_cli_command(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setattr(storage.shutil, "which", lambda _: "aws")
    assert storage.uri("raw/batch=2026-01/") == "s3://my-bucket/raw/batch=2026-01/"
    assert storage.aws_cmd("sync", "a", "b") == [
        "aws",
        "s3",
        "sync",
        "a",
        "b",
        "--endpoint-url",
        "http://localhost:9000",
    ]


def test_s3_mode_without_cli_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    monkeypatch.setattr(storage.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="AWS CLI"):
        storage.aws_cmd("ls")


def test_ingest_and_latest_batch(tmp_path, offline_storage):
    for b in ["2026-02", "2026-01", "2026-11"]:
        write_batch(generate_batch(b, rows=200), offline_storage / f"raw/batch={b}/transactions.parquet")
    (offline_storage / "raw/not-a-batch").mkdir()
    assert available_batches() == ["2026-01", "2026-02", "2026-11"]
    path = ingest_batch("2026-02", tmp_path / "current")
    assert len(pd.read_parquet(path)) == 200
    with pytest.raises(FileNotFoundError, match="generate_data"):
        ingest_batch("2025-12", tmp_path / "missing")


def _candidate(pr_auc, passed=True):
    return {"pr_auc": pr_auc, "gate_passed": passed, "min_pr_auc": 0.5}


def test_promotion_decision():
    assert decide(_candidate(0.7), None, 0.0)["promote"]
    assert decide(_candidate(0.7), {"pr_auc": 0.65, "batch": "2026-01"}, 0.0)["promote"]
    assert not decide(_candidate(0.7), {"pr_auc": 0.72}, 0.0)["promote"]
    assert not decide(_candidate(0.7), {"pr_auc": 0.69}, 0.02)["promote"]
    assert not decide(_candidate(0.4, passed=False), None, 0.0)["promote"]


def test_prod_model_is_rescored_on_candidate_data(tmp_path, batch_df, params):
    params["model"]["hgb"]["max_iter"] = 30
    feats = build_features(batch_df)
    assert score_prod_on(feats) is None  # no prod model yet
    joblib.dump(train(feats, params), tmp_path / "model.joblib")
    storage.upload(tmp_path / "model.joblib", "models/prod/model.joblib")
    assert 0.0 < score_prod_on(feats) <= 1.0


def test_bump_params_keeps_comments(tmp_path):
    text = '# header\nbatch: "2026-04"   # month\nreference_batch: "2026-01"\nnested:\n  batch: x\n'
    out = set_top_level(text, "batch", "2026-07")
    assert out == '# header\nbatch: "2026-07"   # month\nreference_batch: "2026-01"\nnested:\n  batch: x\n'
    f = tmp_path / "params.yaml"
    f.write_text(text)
    bump_main(["--batch", "2026-08", "--reference-batch", "2026-02", "--file", str(f)])
    assert 'batch: "2026-08"' in f.read_text() and 'reference_batch: "2026-02"' in f.read_text()
    with pytest.raises(ValueError):
        bump_main(["--batch", "July", "--file", str(f)])
