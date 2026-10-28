import json

import pytest

from src import deploy_lambda, fetch_prod, storage
from src.deploy_lambda import AwsError
from src.utils import read_json, write_json

URL = "https://abc123.lambda-url.us-east-1.on.aws/"


class FakeLambda:
    """In-memory stand-in for `aws lambda ...` that tracks function state."""

    def __init__(self, can_reserve=True):
        self.exists = False
        self.url = None
        self.sids: set[str] = set()
        self.calls: list[tuple] = []
        self.can_reserve = can_reserve

    def __call__(self, *args):
        self.calls.append(args)
        cmd = args[0]
        missing = AwsError(list(args), "An error occurred (ResourceNotFoundException)")
        if cmd == "get-function":
            if not self.exists:
                raise missing
            return {}
        if cmd == "create-function":
            self.exists = True
            return {}
        if cmd == "get-function-url-config":
            if self.url is None:
                raise missing
            return {"FunctionUrl": self.url}
        if cmd == "create-function-url-config":
            self.url = URL
            return {"FunctionUrl": URL}
        if cmd == "get-policy":
            if not self.sids:
                raise missing
            return {"Policy": json.dumps({"Statement": [{"Sid": s} for s in self.sids]})}
        if cmd == "add-permission":
            self.sids.add(args[args.index("--statement-id") + 1])
            return {}
        if cmd == "put-function-concurrency" and not self.can_reserve:
            raise AwsError(list(args), "InvalidParameterValueException: unreserved concurrency")
        return {}

    def commands(self):
        return [c[0] for c in self.calls if c[0] != "wait"]


@pytest.fixture
def fake(monkeypatch):
    f = FakeLambda()
    monkeypatch.setattr(deploy_lambda, "lam", f)
    monkeypatch.setenv("LAMBDA_ROLE_ARN", "arn:aws:iam::123:role/fraud-mlops-lambda")
    return f


def test_first_deploy_creates_function_and_public_url(fake, tmp_path, monkeypatch):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    deploy_lambda.main(["--image-uri", "repo@sha256:1", "--skip-smoke-test"])
    assert fake.commands() == [
        "get-function",
        "create-function",
        "get-function-url-config",
        "create-function-url-config",
        "get-policy",
        "add-permission",
        "add-permission",
        "put-function-concurrency",
    ]
    create = next(c for c in fake.calls if c[0] == "create-function")
    assert "ImageUri=repo@sha256:1" in create and "Image" in create
    perms = [c for c in fake.calls if c[0] == "add-permission"]
    assert {c[c.index("--action") + 1] for c in perms} == {"lambda:InvokeFunctionUrl", "lambda:InvokeFunction"}
    assert all(c[c.index("--principal") + 1] == "*" for c in perms)
    assert any("--invoked-via-function-url" in c for c in perms)
    assert out.read_text() == f"url={URL}\n"


def test_redeploy_updates_image_and_keeps_permissions(fake):
    deploy_lambda.main(["--image-uri", "repo@sha256:1", "--skip-smoke-test"])
    fake.calls.clear()
    deploy_lambda.main(["--image-uri", "repo@sha256:2", "--skip-smoke-test"])
    assert fake.commands() == [
        "get-function",
        "update-function-code",
        "get-function-url-config",
        "get-policy",
        "put-function-concurrency",
    ]
    assert ("update-function-code", "--function-name", "fraud-mlops-api", "--image-uri", "repo@sha256:2") in fake.calls


def test_create_without_role_is_a_clear_error(fake, monkeypatch):
    monkeypatch.delenv("LAMBDA_ROLE_ARN")
    with pytest.raises(SystemExit, match="LAMBDA_ROLE_ARN"):
        deploy_lambda.main(["--image-uri", "x", "--skip-smoke-test"])


def test_concurrency_cap_failure_is_not_fatal(monkeypatch):
    f = FakeLambda(can_reserve=False)
    f.exists, f.url = True, URL
    monkeypatch.setattr(deploy_lambda, "lam", f)
    deploy_lambda.main(["--image-uri", "x", "--skip-smoke-test"])  # no exception


def test_smoke_test(monkeypatch):
    def fake_http(url, payload=None):
        if url.endswith("/health"):
            return {"status": "ok", "model_loaded": True}
        return {
            "model_batch": "2026-04",
            "threshold": 0.2,
            "predictions": [
                {"transaction_id": "demo-legit", "fraud_probability": 0.01},
                {"transaction_id": "demo-suspicious", "fraud_probability": good},
            ],
        }

    monkeypatch.setattr(deploy_lambda, "_http", fake_http)
    good = 0.9
    assert deploy_lambda.smoke_test(URL)["model_batch"] == "2026-04"
    good = 0.001
    with pytest.raises(RuntimeError, match="smoke test failed"):
        deploy_lambda.smoke_test(URL)


def test_fetch_prod_downloads_registry_model(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_prod, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(fetch_prod, "REPORTS_DIR", tmp_path / "reports")
    with pytest.raises(SystemExit, match="no production model"):
        fetch_prod.main()

    (tmp_path / "m.joblib").write_bytes(b"model")
    write_json(tmp_path / "metrics.json", {"batch": "2026-06", "pr_auc": 0.74})
    storage.upload(tmp_path / "m.joblib", "models/prod/model.joblib")
    storage.upload(tmp_path / "metrics.json", "models/prod/metrics.json")
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    fetch_prod.main()
    assert (tmp_path / "models" / "model.joblib").read_bytes() == b"model"
    assert read_json(tmp_path / "reports" / "metrics.json")["batch"] == "2026-06"
    assert out.read_text() == "batch=2026-06\n"
