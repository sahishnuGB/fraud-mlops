# Run inside an activated virtualenv (see README). Override with: make PYTHON=python3 ...
PYTHON ?= python
BATCHES ?= 2026-01 2026-02 2026-03 2026-04
DRIFT_BATCHES ?= 2026-05 2026-06
IMAGE ?= fraud-mlops
PORT ?= 8000

.PHONY: help install data remote repro pull push metrics plots publish test lint format check \
        serve docker-build docker-run minio-up minio-down clean

help:
	@echo "install     install runtime + dev dependencies"
	@echo "data        generate monthly batches ($(BATCHES); drifted: $(DRIFT_BATCHES)) and upload"
	@echo "remote      point DVC at s3://\$$S3_BUCKET/dvcstore (or local_s3/ when unset)"
	@echo "repro       run the DVC pipeline"
	@echo "pull/push   sync DVC cache with the remote"
	@echo "metrics     show metrics and diff against HEAD"
	@echo "publish     publish the model to models/prod/ if promote approved it"
	@echo "test/lint   pytest / ruff"
	@echo "serve       run the prediction API locally (http://localhost:$(PORT)/docs)"
	@echo "docker-build / docker-run   build and run the API image with the trained model baked in"
	@echo "minio-up    start local MinIO (S3 API on :9000, console on :9001)"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

data:
	$(PYTHON) -m src.generate_data --batch $(BATCHES)
	$(PYTHON) -m src.generate_data --batch 2026-05 --drift 0.5
	$(PYTHON) -m src.generate_data --batch 2026-06 --drift 1.0

remote:
	$(PYTHON) -m src.configure_remote

repro: remote
	dvc repro

pull: remote
	dvc pull --allow-missing

push: remote
	dvc push

metrics:
	dvc metrics show
	dvc metrics diff

plots:
	dvc plots show

publish:
	$(PYTHON) -m src.publish

test:
	$(PYTHON) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

check: lint test

serve:
	$(PYTHON) -m uvicorn src.serve:app --reload --port $(PORT)

docker-build:
	@test -f models/model.joblib || (echo "models/model.joblib missing: run 'make repro' or 'dvc pull' first" && exit 1)
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8000 --name fraud-api $(IMAGE)

minio-up:
	docker compose up -d

minio-down:
	docker compose down

clean:
	rm -rf data models reports/plots .pytest_cache .ruff_cache
