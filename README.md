# Fraud detection MLOps with drift monitoring, at $0

An end-to-end, reproducible pipeline for credit card fraud detection:

- **Data**: a deterministic synthetic generator writes monthly batches, optionally with drift, to S3. No Kaggle download is needed.
- **Pipeline**: [DVC](https://dvc.org): ingest → validate → split → featurize → train → evaluate → drift_check → promote
- **Storage**: S3 for raw batches, the DVC remote and the production model. MinIO works through `AWS_ENDPOINT_URL`, and with no bucket set everything falls back to a local `local_s3/` folder.
- **CI/CD**: GitHub Actions with OIDC auth to AWS (no stored keys), CML PR reports, a PR-AUC quality gate, and a monthly scheduled retrain that opens a PR with a drift summary.
- **Serving**: a FastAPI scoring service in one Docker image that runs locally and is deployed to **AWS Lambda** behind a public Function URL. Every promoted model is redeployed automatically and smoke-tested.

Dependencies are kept light: scikit-learn, imbalanced-learn, pandas, pyarrow, matplotlib, `dvc[s3]`.

```
generate_data.py ──► s3://$S3_BUCKET/raw/batch=YYYY-MM/transactions.parquet
                                  │  aws s3 sync
                                  ▼
 ingest ─► validate ─► split ─► featurize ─► train ─► evaluate ──┐
    │                                                            ├─► promote ─► reports/promotion.json
    └───────────────────────────────────────────► drift_check ───┘        │
                                                  (PSI vs reference)      │ CI on main only
                                                                          ▼
                                       python -m src.publish ─► s3://$S3_BUCKET/models/prod/
```

## Quickstart (offline, no AWS)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

git init && dvc init                 # skip if already done (this repo ships .dvc/)
make data                            # 6 monthly batches -> local_s3/raw/  (05 and 06 drifted)
make repro                           # = python -m src.configure_remote && dvc repro
dvc metrics show
make publish                         # local "prod" model -> local_s3/models/prod/
make test lint
```

No `make` (for example on plain Windows)? Every target is a one-liner in the [Makefile](Makefile). For example, `make data` runs `python -m src.generate_data --batch 2026-01 2026-02 2026-03 2026-04` and then the two `--drift` batches.

### See drift in action

```bash
python -m src.bump_params --batch 2026-06      # a drifted month
dvc repro
cat reports/drift.md                           # PSI table, drifted features flagged
dvc metrics diff                               # vs the committed baseline
```

### Serve the model (FastAPI + Docker, locally or on AWS Lambda)

A small API scores raw transactions with the trained model. It uses the same `build_features` as training, so there's no train/serve skew. The default decision threshold is the best-F1 threshold from evaluation, because 0.5 is too conservative for rare fraud.

```bash
make serve                           # = uvicorn src.serve:app --reload  -> http://localhost:8000/docs

# or as a container, with the model baked in (needs models/model.joblib from `make repro` / `dvc pull`)
make docker-build                    # = docker build -t fraud-mlops .
make docker-run                      # = docker run --rm -p 8000:8000 fraud-mlops
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request.json
# PowerShell: Invoke-RestMethod -Method Post -Uri localhost:8000/predict -ContentType application/json -InFile examples/predict_request.json
```

```json
{"model_batch": "2026-04", "threshold": 0.200206, "predictions": [
  {"transaction_id": "demo-legit",      "fraud_probability": 0.000316, "is_fraud": false},
  {"transaction_id": "demo-suspicious", "fraud_probability": 0.976372, "is_fraud": true}]}
```

| Endpoint | |
|---|---|
| `GET /docs` | interactive Swagger UI (`/` redirects here) |
| `GET /health` | liveness and whether a model is loaded (also the container `HEALTHCHECK`) |
| `GET /model` | model type, sampling, training batch, features, test metrics, default threshold |
| `POST /predict` | 1–1000 transactions → fraud probability + label. `?threshold=` overrides the default. Invalid input returns 422 |

The image is `python:3.12-slim`, runs as a non-root user, and installs only [requirements-serve.txt](requirements-serve.txt). Those are the same pins training uses, so the pickled pipeline always loads with identical library versions. A strict [.dockerignore](.dockerignore) allowlist keeps data and secrets out of the build context. CI builds the image and smoke-tests `/predict` on every push.

**On AWS Lambda:** the same image includes the [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter), which does nothing locally. In Lambda it forwards invocations to uvicorn, so the API code is identical in both places. [deploy.yaml](.github/workflows/deploy.yaml) ships the production model from the registry. It never ships an unpromoted candidate. It pushes the image to ECR and creates or updates the function and its public Function URL, then smoke-tests `/predict`. Lambda bills per request, so an idle demo costs nothing; expect roughly $0.10–1/month, mostly ECR storage. Setup and the kill switch are in [SETUP.md step 8](SETUP.md#8-deploy-the-api-to-aws-lambda-optional).

### With MinIO (S3 API, still $0)

MinIO no longer publishes community images, so [docker-compose.yml](docker-compose.yml) builds a pinned release from source. The first `up` takes a few minutes. You also need the [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on your machine.

```bash
docker compose up -d                 # S3 API :9000, console :9001, bucket "fraud-mlops"
cp minio.env.example minio.env
set -a; . ./minio.env; set +a        # S3_BUCKET + AWS_ENDPOINT_URL + minioadmin keys
make data repro push publish
```

### With AWS + GitHub Actions

See **[SETUP.md](SETUP.md)** for the zero-cost AWS setup: a zero-spend budget, one region, a private bucket with lifecycle rules, an OIDC role with least privilege, a public repo, the optional Lambda deployment, and teardown.

## Storage modes

| `S3_BUCKET` | `AWS_ENDPOINT_URL` | Backend | DVC remote |
|---|---|---|---|
| unset | – | `local_s3/` folder (`LOCAL_S3_DIR` overrides) | `local` → `local_s3/dvcstore` |
| set | unset | AWS S3 via the AWS CLI | `s3` → `s3://$S3_BUCKET/dvcstore` |
| set | set | MinIO / any S3-compatible store | same, with `endpointurl` |

DVC config cannot read environment variables, so [src/configure_remote.py](src/configure_remote.py) writes the bucket remote to the git-ignored `.dvc/config.local`. `make repro`, `make push` and both CI workflows call it for you.

Bucket layout: `raw/batch=YYYY-MM/`, `dvcstore/`, `models/prod/`, `models/archive/`.

## Data

[src/generate_data.py](src/generate_data.py) produces about 30k transactions per month with a fraud rate of about 2%. Columns: amount, merchant category, card-present, foreign, distance from home, customer age, account age, 24h velocity, 30-day average amount, and device change. Labels come from a hidden logistic risk model with unobserved noise, so the task is learnable but not trivial.

```bash
python -m src.generate_data --batch 2026-07                 # upload one month
python -m src.generate_data --batch 2026-07 --drift 0.7     # with drift (0..1, flag alone = 1.0)
python -m src.generate_data --batch 2026-07 --no-upload --out-dir /tmp/x
```

The same arguments always give identical data, because the seed is derived from `(seed, year, month, rows, drift)`. `--drift` causes both:

- **covariate drift**: inflation, more online, night and foreign transactions, fewer card-present, younger customers, more device changes
- **concept drift**: device changes become more predictive of fraud and foreign transactions less, and the fraud rate rises

## Pipeline stages

| Stage | Does | Outputs |
|---|---|---|
| `ingest` | `aws s3 sync` of `batch` and `reference_batch` (local copy offline). `always_changed`, but unchanged files skip everything downstream | `data/raw/{current,reference}` |
| `validate` | schema, dtypes, nulls, duplicates, ranges, categories, fraud-rate bounds. Fails the pipeline on errors | `reports/validation.json` |
| `split` | stratified train/test split | `data/split/` |
| `featurize` | stateless features (log amounts, amount/avg ratio, cyclic hour, night/weekend, one-hot categories) | `data/features/` |
| `train` | imblearn pipeline: optional scaler → sampler (`none`/`undersample`/`smote`) → `hgb` or `logreg` | `models/model.joblib` |
| `evaluate` | PR-AUC (average precision), ROC-AUC, P/R/F1, best-F1 threshold, recall@P90, gate | `reports/metrics.json`, PR curve, confusion matrix |
| `drift_check` | PSI per feature vs the reference batch (quantile bins from the reference) | `reports/drift.{json,md}`, PSI chart |
| `promote` | re-scores the prod model (`models/prod/`) on **the same test set** as the candidate and promotes if the gate passes and candidate PR-AUC ≥ prod + `min_improvement`. Falls back to prod's stored `metrics.json` if its model can't be loaded | `reports/promotion.json` |

`promote` only decides. `python -m src.publish` does the upload: it archives the old prod model and then uploads the model, metrics and metadata. CI runs it only on `main`.

### params.yaml

```yaml
batch: "2026-04"          # month to train on
reference_batch: "2026-01" # drift baseline
sampling: none             # none | undersample | smote
model:
  type: hgb                # hgb | logreg   (+ hyperparameters for each)
min_pr_auc: 0.5            # quality gate
psi_threshold: 0.2         # drift threshold per feature
```

Try an experiment without editing the file: `dvc exp run -S sampling=smote -S model.type=logreg`, then `dvc exp show`. DVC experiments need at least one git commit first.

Results on the default batch `2026-04` (local run):

| model | sampling | PR-AUC | gate (0.5) |
|---|---|---:|---|
| **hgb** (default) | none | **0.599** | pass |
| logreg | smote | 0.560 | pass |
| hgb | undersample | 0.554 | pass |
| logreg | none | 0.544 | pass |
| hgb | smote | 0.473 | **fail** |

On the drifted `2026-06` batch, the default model reaches PR-AUC 0.740. PSI flags `txn_count_24h`, `customer_age` and `avg_amount_30d`.

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| [ci.yaml](.github/workflows/ci.yaml) | every push / PR | `ruff check`, `ruff format --check`, `pytest` (offline, no AWS). Then trains a model offline, builds the Docker image and smoke-tests `/predict` |
| [train.yaml](.github/workflows/train.yaml) | PRs and pushes to `main` touching the pipeline | OIDC → AWS, `dvc pull` → `dvc repro` → `dvc push`, `dvc metrics diff` + `dvc params diff` vs base, CML PR comment with plots and drift table, **fails if PR-AUC < `min_pr_auc`**, publishes the model to `models/prod/` on `main` |
| [deploy.yaml](.github/workflows/deploy.yaml) | after a successful `train` on `main`, serving-code changes, manual | fetches the **prod** model from `models/prod/`, builds and pushes the image to ECR, deploys to Lambda (Function URL) and smoke-tests it. Uses a separate role that only the `production` environment can assume |
| [scheduled.yaml](.github/workflows/scheduled.yaml) | cron `0 6 2 * *` + manual | finds the newest `raw/batch=*` in S3, bumps `params.yaml`, reruns the pipeline, opens a PR whose body is the drift summary and metrics |

## Repo map

```
src/            pipeline stages + helpers (storage, schema, publish, bump_params, find_latest_batch, ...)
                serve.py = FastAPI prediction service
tests/          pytest suite, fully offline (each test gets its own temp "bucket")
dvc.yaml        pipeline definition       params.yaml   all knobs
dvc.lock        pinned hashes of the last run (commit it)
reports/*.json  metrics (git-tracked for dvc metrics diff); plots are DVC-cached
Dockerfile      API image (model baked in)    examples/  sample /predict request
infra/          S3 + ECR lifecycle rules, least-privilege IAM policies (CI, deploy, Lambda), OIDC trust policies, MinIO Dockerfile
.vscode/        settings + recommended extensions (Python, Ruff, DVC, YAML, GitHub Actions)
```
