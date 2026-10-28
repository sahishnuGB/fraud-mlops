"""Shared helpers: project paths, params loading, JSON IO, logging."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = ROOT / "params.yaml"

DATA = ROOT / "data"
RAW_CURRENT = DATA / "raw" / "current"
RAW_REFERENCE = DATA / "raw" / "reference"
SPLIT_DIR = DATA / "split"
FEATURES_DIR = DATA / "features"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

BATCH_FILE = "transactions.parquet"
BATCH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_batch(batch: str) -> str:
    batch = str(batch)
    if not BATCH_RE.match(batch):
        raise ValueError(f"batch must look like YYYY-MM, got {batch!r}")
    return batch


def batch_prefix(batch: str) -> str:
    return f"raw/batch={check_batch(batch)}/"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
