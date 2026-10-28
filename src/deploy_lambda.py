"""Deploy the API image to AWS Lambda behind a public Function URL, then smoke-test it.

Creates the function on the first run and updates its image afterwards. Idempotent.

    python -m src.deploy_lambda --image-uri <account>.dkr.ecr.<region>.amazonaws.com/fraud-mlops-api:<tag>

Environment:
    LAMBDA_ROLE_ARN   execution role (only needed for the first create)
    LAMBDA_FUNCTION   function name (default fraud-mlops-api)
    AWS_REGION        region, as for any AWS CLI call
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from src.utils import ROOT, get_logger

log = get_logger("deploy_lambda")

DEFAULT_FUNCTION = "fraud-mlops-api"
MEMORY_MB = 1024  # more memory = more CPU = faster cold starts; still pennies per month
TIMEOUT_S = 30
# Public Function URLs need both statements since Oct 2025.
# https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html
URL_PERMISSIONS = [
    ("FunctionURLPublicInvokeUrl", "lambda:InvokeFunctionUrl", ["--function-url-auth-type", "NONE"]),
    ("FunctionURLPublicInvoke", "lambda:InvokeFunction", ["--invoked-via-function-url"]),
]


class AwsError(RuntimeError):
    def __init__(self, args: list[str], stderr: str):
        super().__init__(f"aws {' '.join(args)} failed: {stderr.strip()}")
        self.stderr = stderr

    @property
    def not_found(self) -> bool:
        return "ResourceNotFoundException" in self.stderr


def lam(*args: str) -> dict:
    """Run ``aws lambda <args>`` and return the parsed JSON output."""
    exe = shutil.which("aws")
    if exe is None:
        raise RuntimeError("the AWS CLI is required to deploy (https://aws.amazon.com/cli/)")
    proc = subprocess.run([exe, "lambda", *args, "--output", "json"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AwsError(list(args), proc.stderr)
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def function_exists(name: str) -> bool:
    try:
        lam("get-function", "--function-name", name)
        return True
    except AwsError as e:
        if e.not_found:
            return False
        raise


def create_function(name: str, image_uri: str, role_arn: str | None) -> None:
    if not role_arn:
        raise SystemExit("LAMBDA_ROLE_ARN must be set to create the function (see SETUP.md)")
    args = [
        "create-function",
        "--function-name", name,
        "--package-type", "Image",
        "--code", f"ImageUri={image_uri}",
        "--role", role_arn,
        "--architectures", "x86_64",
        "--memory-size", str(MEMORY_MB),
        "--timeout", str(TIMEOUT_S),
        "--description", "Credit card fraud scoring API (fraud-mlops)",
    ]  # fmt: skip
    for attempt in range(6):  # a freshly created IAM role takes a few seconds to propagate
        try:
            lam(*args)
            break
        except AwsError as e:
            if "cannot be assumed" not in e.stderr or attempt == 5:
                raise
            log.info("execution role not assumable yet, retrying in 10s")
            time.sleep(10)
    lam("wait", "function-active", "--function-name", name)
    log.info("created function %s", name)


def update_function(name: str, image_uri: str) -> None:
    lam("update-function-code", "--function-name", name, "--image-uri", image_uri)
    lam("wait", "function-updated", "--function-name", name)
    log.info("updated function %s -> %s", name, image_uri)


def ensure_public_url(name: str) -> str:
    try:
        url = lam("get-function-url-config", "--function-name", name)["FunctionUrl"]
    except AwsError as e:
        if not e.not_found:
            raise
        url = lam("create-function-url-config", "--function-name", name, "--auth-type", "NONE")["FunctionUrl"]
        log.info("created function URL %s", url)

    try:
        policy = json.loads(lam("get-policy", "--function-name", name)["Policy"])
        existing = {s["Sid"] for s in policy.get("Statement", [])}
    except AwsError as e:
        if not e.not_found:
            raise
        existing = set()
    for sid, action, extra in URL_PERMISSIONS:
        if sid not in existing:
            lam("add-permission", "--function-name", name, "--statement-id", sid,
                "--action", action, "--principal", "*", *extra)  # fmt: skip
    return url


def cap_concurrency(name: str, limit: int) -> None:
    """Bound worst-case cost if the public URL gets hammered."""
    try:
        lam("put-function-concurrency", "--function-name", name, "--reserved-concurrent-executions", str(limit))
        log.info("reserved concurrency capped at %d", limit)
    except AwsError as e:
        # New accounts have a total concurrency of 10, and AWS keeps 100 unreserved, so this
        # fails there. That account limit is then the cap; the budget alert is the backstop.
        log.warning("could not cap reserved concurrency (%s); relying on the account limit", e.stderr.strip())


def _http(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read())


def smoke_test(url: str, wait_s: int = 90) -> dict:
    base = url.rstrip("/")
    deadline = time.time() + wait_s
    while True:  # the first request after a deploy is a cold start
        try:
            if _http(f"{base}/health").get("model_loaded"):
                break
        except (urllib.error.URLError, TimeoutError) as e:
            if time.time() > deadline:
                raise RuntimeError(f"{base}/health did not become healthy: {e}") from e
        time.sleep(3)
    examples = json.loads((ROOT / "examples" / "predict_request.json").read_text())
    result = _http(f"{base}/predict", examples)
    scores = {p["transaction_id"]: p["fraud_probability"] for p in result["predictions"]}
    if not scores["demo-suspicious"] > scores["demo-legit"]:
        raise RuntimeError(f"smoke test failed, suspicious example did not outscore legit: {scores}")
    log.info("smoke test passed: %s", scores)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--function", default=os.environ.get("LAMBDA_FUNCTION", DEFAULT_FUNCTION))
    parser.add_argument("--reserved-concurrency", type=int, default=2)
    parser.add_argument("--skip-smoke-test", action="store_true")
    args = parser.parse_args(argv)

    if function_exists(args.function):
        update_function(args.function, args.image_uri)
    else:
        create_function(args.function, args.image_uri, os.environ.get("LAMBDA_ROLE_ARN"))
    url = ensure_public_url(args.function)
    cap_concurrency(args.function, args.reserved_concurrency)
    result = None if args.skip_smoke_test else smoke_test(url)

    log.info("live at %sdocs", url)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"url={url}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"## 🚀 Deployed\n\n- Function URL: {url}\n- Docs: {url}docs\n- Image: `{args.image_uri}`\n")
            if result:
                f.write(f"- Smoke test: model batch `{result['model_batch']}`, threshold {result['threshold']}\n")


if __name__ == "__main__":
    main()
