# Zero-cost setup

Three ways to run this project, all $0:

| Mode | What you need | Cost |
|---|---|---|
| **Offline** | Python only. `S3_BUCKET` unset → `local_s3/` folder | $0 |
| **MinIO** | Docker + AWS CLI. S3 API on localhost | $0 |
| **AWS + GitHub Actions** | AWS account, public GitHub repo | $0 if you follow this guide (a few MB of S3 data) |
| **+ Live API on Lambda** (optional, step 8) | the above | about $0.10–1/month (mostly ECR image storage), covered by free-plan credits |

The first two need no account at all; see the README quickstart. The rest of this file covers the AWS + GitHub path.

> **Windows:** run every command in this guide in **Git Bash** (VS Code terminal → `+` dropdown → *Git Bash*), from the project folder with the venv active (`source .venv/Scripts/activate`). The commands use bash syntax (`export`, `$(...)`, `sed`), which PowerShell doesn't understand.

---

## What this setup uses (and what it avoids)

Uses only: **one S3 bucket**, **one IAM role** (assumed through GitHub OIDC, so no access keys), **one OIDC provider** and **one budget**. IAM, OIDC and STS are free. Budgets are free for monitoring. The optional Lambda deployment (step 8) adds one function, one ECR repository, one log group and two more roles.

The bucket holds about 1 MB per monthly batch, plus the DVC cache (a few MB) and the prod model (under 1 MB). At S3 Standard prices that comes to fractions of a cent per month, and it falls inside AWS's free usage allowance:

- Accounts created **before 15 July 2025** have the legacy 12-month Free Tier, which includes 5 GB of S3 Standard storage, 20k GET and 2k PUT requests per month.
- Accounts created **on or after 15 July 2025** get the free plan with sign-up credits instead. Usage this small draws down cents at most, and the free plan cannot be charged until you upgrade it.
- Data transfer out to the internet (to GitHub runners) is free for the first 100 GB/month.

Check the current terms at <https://aws.amazon.com/free/>. They change from time to time.

**Do not enable** any of these, because each one can cost money: SSE-KMS customer keys (use the default SSE-S3), S3 Intelligent-Tiering, S3 Inventory or Storage Lens advanced metrics, CloudTrail data events, CloudWatch alarms beyond the free quota, NAT gateways, or a second region.

---

## 1. Lock down the account

1. Enable **MFA on the root user** and stop using root.
2. Create an admin identity for yourself (IAM Identity Center, or an IAM user with MFA) and configure it locally:
   ```bash
   aws configure sso      # or: aws configure
   aws sts get-caller-identity
   ```

## 2. Create a zero-spend budget (do this first)

Console: **Billing and Cost Management → Budgets → Create budget → Use a template → Zero spend budget**, then enter your email. You get an alert as soon as any spend exceeds $0.01.

CLI equivalent:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
EMAIL=you@example.com

aws budgets create-budget --account-id "$ACCOUNT_ID" \
  --budget '{"BudgetName":"zero-spend","BudgetLimit":{"Amount":"0.01","Unit":"USD"},
             "TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers "[{\"Notification\":{\"NotificationType\":\"ACTUAL\",
     \"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":0,\"ThresholdType\":\"ABSOLUTE_VALUE\"},
     \"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$EMAIL\"}]}]"
```

Optionally also enable **Billing preferences → Receive Free Tier usage alerts**.

## 3. Pick a single region

Keep everything in one region. This guide uses `us-east-1`:

```bash
export AWS_REGION=us-east-1
export S3_BUCKET=fraud-mlops-$ACCOUNT_ID    # bucket names are global, so add your account ID
```

## 4. Create the bucket

```bash
# us-east-1 takes no LocationConstraint; any other region needs:
#   --create-bucket-configuration LocationConstraint=$AWS_REGION
aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION"

# Private bucket: block all public access (the default for new buckets, set explicitly here)
aws s3api put-public-access-block --bucket "$S3_BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Lifecycle: abort stale multipart uploads, expire old versions and archived models after 30 days
aws s3api put-bucket-lifecycle-configuration --bucket "$S3_BUCKET" \
  --lifecycle-configuration file://infra/s3-lifecycle.json
```

Leave **versioning off** (the default). The DVC cache is content-addressed, so it does not need versioning. Default encryption is SSE-S3, which is free; leave it as is.

Bucket layout:

```
s3://$S3_BUCKET/
├── raw/batch=YYYY-MM/transactions.parquet   # monthly batches (generate_data.py)
├── dvcstore/                                # DVC remote (data, features, models, plots)
└── models/
    ├── prod/      model.joblib, metrics.json, metadata.json, drift.json
    └── archive/   previous prod models (expired after 30 days)
```

## 5. GitHub OIDC provider and CI role

The role can read `raw/`, read and write `dvcstore/` and `models/`, and do nothing else ([infra/iam-policy.json](infra/iam-policy.json)). Only `main` and pull requests of **your** repo can assume it ([infra/oidc-trust-policy.json](infra/oidc-trust-policy.json)).

```bash
GITHUB_REPO=your-user/fraud-mlops   # owner/name, exactly as GitHub shows it

# Repos created after 15 July 2026 sign their OIDC tokens with immutable IDs
# (repo:owner@<owner id>/repo@<repo id>:...), so the trust policy accepts both formats.
# Create the GitHub repo (step 6a) first, then:
GITHUB_REPO_IMMUTABLE=$(curl -s "https://api.github.com/repos/$GITHUB_REPO" | python -c \
  "import sys,json; d=json.load(sys.stdin); o,r=d['full_name'].split('/'); print(f\"{o}@{d['owner']['id']}/{r}@{d['id']}\")")
echo "$GITHUB_REPO_IMMUTABLE"         # e.g. your-user@12345678/fraud-mlops@987654321

# One OIDC provider per account. Older CLI versions also require:
#   --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com

mkdir -p infra/.rendered   # filled-in copies of the policies (git-ignored)
sed -e "s|<ACCOUNT_ID>|$ACCOUNT_ID|g" -e "s|<GITHUB_REPO_IMMUTABLE>|$GITHUB_REPO_IMMUTABLE|g" \
    -e "s|<GITHUB_OWNER>/<REPO>|$GITHUB_REPO|g" \
  infra/oidc-trust-policy.json > infra/.rendered/trust.json
sed -e "s|<BUCKET>|$S3_BUCKET|g" infra/iam-policy.json > infra/.rendered/policy.json

aws iam create-role --role-name fraud-mlops-ci --max-session-duration 3600 \
  --assume-role-policy-document file://infra/.rendered/trust.json
aws iam put-role-policy --role-name fraud-mlops-ci --policy-name fraud-mlops-s3 \
  --policy-document file://infra/.rendered/policy.json

aws iam get-role --role-name fraud-mlops-ci --query Role.Arn --output text   # -> AWS_ROLE_ARN
```

## 6. GitHub repository

1. Create a **public** repository. Standard GitHub-hosted runners are free and unlimited for public repos; private repos only get a monthly minutes quota. Nothing sensitive lives in the repo: the data is synthetic and the bucket is private.
2. **Settings → Secrets and variables → Actions → Variables** (these are not secrets):
   - `S3_BUCKET` = your bucket name
   - `AWS_REGION` = `us-east-1`
   - `AWS_ROLE_ARN` = the ARN from step 5
3. **Settings → Actions → General → Workflow permissions**: select *Read and write permissions* and tick **Allow GitHub Actions to create and approve pull requests**. `scheduled.yaml` needs this to open its PR.
4. Push the code:
   ```bash
   git add -A && git commit -m "Initial pipeline"
   git branch -M main
   git remote add origin https://github.com/$GITHUB_REPO.git   # first push opens a browser sign-in
   git push -u origin main
   ```

## 7. Seed the bucket and run

Upload batches with **your own** credentials. The CI role deliberately cannot write to `raw/`.

```bash
python -m src.generate_data --batch 2026-01 2026-02 2026-03 2026-04
python -m src.generate_data --batch 2026-05 --drift 0.5
python -m src.generate_data --batch 2026-06 --drift 1.0
aws s3 ls "s3://$S3_BUCKET/raw/"

# Optional: seed the DVC cache from your machine (needs `pip install dvc-s3`; CI does this anyway)
# python -m src.configure_remote && dvc repro && dvc push
```

Then run **Actions → train → Run workflow** on `main`. That trains, pushes the DVC cache and publishes the first prod model. **Actions → scheduled-retrain → Run workflow** finds `2026-06`, retrains on it and opens a PR with the drift summary.

> Pull requests opened with the default `GITHUB_TOKEN` do not trigger other workflows, so `train.yaml` will not run on the scheduled PR. That is fine: the PR body already contains the metrics and drift report, and merging it runs `train.yaml` on `main`. If you want CI checks on that PR too, close and reopen it, or use a fine-grained PAT for `gh pr create`.

## 8. Deploy the API to AWS Lambda (optional)

This puts the FastAPI service from the README on a public HTTPS URL. The image is the same one you run locally; the [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) inside it handles the translation.

[deploy.yaml](.github/workflows/deploy.yaml) runs after every successful `train` run on `main`, whenever the serving code changes, and on demand. It always ships the **production** model from `models/prod/`, never an unpromoted candidate. It builds and pushes the image to ECR, creates or updates the function, sets up the Function URL, and smoke-tests `/predict` before it finishes.

**What it costs:** Lambda bills per request and per GB-second, so an idle demo costs nothing, and the Lambda free tier covers far more traffic than a portfolio gets. The ongoing cost is ECR storage for at most 3 images (the lifecycle policy deletes older ones), roughly $0.10 a month, plus CloudWatch logs kept for 7 days. There is no load balancer, NAT gateway or API Gateway. Once this is running, the zero-spend budget will email you about those cents, so add a second budget at around $5 as the real tripwire.

### 8a. AWS resources (one-time, with your admin credentials)

```bash
# Image repository; keep only the 3 newest images
aws ecr create-repository --repository-name fraud-mlops-api --region "$AWS_REGION"
aws ecr put-lifecycle-policy --repository-name fraud-mlops-api \
  --lifecycle-policy-text file://infra/ecr-lifecycle.json

# Log group with 7-day retention (Lambda's default keeps logs forever)
aws logs create-log-group --log-group-name /aws/lambda/fraud-mlops-api
aws logs put-retention-policy --log-group-name /aws/lambda/fraud-mlops-api --retention-in-days 7

# Fill in the placeholders of the remaining policies
fill() { sed -e "s|<ACCOUNT_ID>|$ACCOUNT_ID|g" -e "s|<REGION>|$AWS_REGION|g" \
             -e "s|<BUCKET>|$S3_BUCKET|g" -e "s|<GITHUB_REPO_IMMUTABLE>|$GITHUB_REPO_IMMUTABLE|g" \
             -e "s|<GITHUB_OWNER>/<REPO>|$GITHUB_REPO|g" "$1"; }
fill infra/lambda-exec-policy.json   > infra/.rendered/lambda-exec.json
fill infra/deploy-trust-policy.json  > infra/.rendered/deploy-trust.json
fill infra/iam-deploy-policy.json    > infra/.rendered/deploy-policy.json

# Execution role the function runs as: it can only write its own logs
aws iam create-role --role-name fraud-mlops-lambda \
  --assume-role-policy-document file://infra/lambda-trust-policy.json
aws iam put-role-policy --role-name fraud-mlops-lambda --policy-name logs \
  --policy-document file://infra/.rendered/lambda-exec.json

# Deploy role for GitHub Actions: only the repo's "production" environment can assume it,
# so pull requests (which use the CI role) can never deploy
aws iam create-role --role-name fraud-mlops-deploy --max-session-duration 3600 \
  --assume-role-policy-document file://infra/.rendered/deploy-trust.json
aws iam put-role-policy --role-name fraud-mlops-deploy --policy-name deploy \
  --policy-document file://infra/.rendered/deploy-policy.json

aws iam get-role --role-name fraud-mlops-lambda --query Role.Arn --output text   # -> LAMBDA_ROLE_ARN
aws iam get-role --role-name fraud-mlops-deploy --query Role.Arn --output text   # -> AWS_DEPLOY_ROLE_ARN
```

The deploy role ([infra/iam-deploy-policy.json](infra/iam-deploy-policy.json)) can read `models/prod/`, push to the one ECR repository, manage the one function, and pass the one execution role to Lambda. It can do nothing else.

### 8b. GitHub

1. **Settings → Environments → New environment** named `production`. Under *Deployment branches and tags*, choose *Selected branches and tags* and add `main`. Optionally add yourself as a required reviewer to approve each deploy by hand.
2. **Settings → Secrets and variables → Actions → Variables**: add `AWS_DEPLOY_ROLE_ARN` and `LAMBDA_ROLE_ARN` from 8a. `deploy.yaml` skips itself until `AWS_DEPLOY_ROLE_ARN` exists.
3. Make sure `train` has published a prod model on `main` (step 7), then run **Actions → deploy → Run workflow**.

The run summary and the repo's **Deployments** page show the live URL, e.g. `https://<id>.lambda-url.us-east-1.on.aws/docs`. The first request after a deploy is a cold start and takes a few seconds; after that, requests are fast.

```bash
URL=$(aws lambda get-function-url-config --function-name fraud-mlops-api --query FunctionUrl --output text)
curl -X POST "${URL}predict" -H "Content-Type: application/json" -d @examples/predict_request.json
```

### Abuse protection and the kill switch

The URL is public by design, so anyone with the link can call it. To bound the damage:
- Each request accepts at most 1000 transactions and has a 30 s timeout.
- The deploy script tries to cap reserved concurrency at 2. New accounts have a total concurrency of 10, and AWS requires 100 to stay unreserved, so on those accounts the cap fails with a warning and the account limit of 10 becomes the ceiling.
- The $5 budget alert is the backstop.

If you ever see unexpected traffic, throttle the function to zero. It keeps everything else in place:

```bash
aws lambda put-function-concurrency --function-name fraud-mlops-api --reserved-concurrent-executions 0
# undo:  aws lambda delete-function-concurrency --function-name fraud-mlops-api
```

The next deploy resets the cap to 2, so disable the `deploy` workflow too while you investigate.

## 9. Teardown (back to $0)

```bash
# Lambda deployment (step 8), if you did it. Deleting the function also removes its URL and policy.
aws lambda delete-function --function-name fraud-mlops-api
aws ecr delete-repository --repository-name fraud-mlops-api --force
aws logs delete-log-group --log-group-name /aws/lambda/fraud-mlops-api
aws iam delete-role-policy --role-name fraud-mlops-lambda --policy-name logs
aws iam delete-role --role-name fraud-mlops-lambda
aws iam delete-role-policy --role-name fraud-mlops-deploy --policy-name deploy
aws iam delete-role --role-name fraud-mlops-deploy

aws s3 rm "s3://$S3_BUCKET" --recursive
# Only if you turned versioning on: delete all object versions first, e.g. in the console with "Empty bucket"
aws s3api delete-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION"

aws iam delete-role-policy --role-name fraud-mlops-ci --policy-name fraud-mlops-s3
aws iam delete-role --role-name fraud-mlops-ci
aws iam delete-open-id-connect-provider \
  --open-id-connect-provider-arn "arn:aws:iam::$ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"

# Keep the zero-spend budget, it costs nothing. Or remove it:
aws budgets delete-budget --account-id "$ACCOUNT_ID" --budget-name zero-spend
```

Also disable the `scheduled-retrain` and `deploy` workflows (or delete the repo variables) so the monthly cron stops trying to reach the bucket.
