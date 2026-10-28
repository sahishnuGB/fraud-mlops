# syntax=docker/dockerfile:1
# Fraud detection API with the trained model baked in. One image, two targets:
#   local:   dvc repro && docker build -t fraud-mlops . && docker run --rm -p 8000:8000 fraud-mlops
#   Lambda:  .github/workflows/deploy.yaml builds it with the prod model and deploys it
#
# Lambda Web Adapter: inside Lambda it turns invocations into HTTP requests to uvicorn.
# Outside Lambda the extension is never started, so local `docker run` is unaffected.
# --build-arg LAMBDA_ADAPTER=0 leaves it out; BuildKit then skips the adapter stage and never
# contacts public.ecr.aws (CI does this: anonymous pulls from shared runner IPs get rate-limited).
ARG LAMBDA_ADAPTER=1

FROM public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 AS lambda-adapter
FROM scratch AS extensions-1
COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter
FROM scratch AS extensions-0
FROM extensions-${LAMBDA_ADAPTER} AS extensions

FROM python:3.12-slim
COPY --from=extensions / /

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health

WORKDIR /app

# Same pins as training, so the pickled pipeline loads with identical library versions.
# The cache mount keeps downloaded wheels between builds (a failed build resumes instead of
# starting over) without adding them to the image.
COPY requirements-serve.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-serve.txt

RUN useradd --create-home --uid 1000 app
COPY src/ src/
COPY models/model.joblib models/model.joblib
COPY reports/metrics.json reports/metrics.json

LABEL org.opencontainers.image.title="fraud-mlops" \
      org.opencontainers.image.description="Credit card fraud scoring API (FastAPI + scikit-learn)"

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
