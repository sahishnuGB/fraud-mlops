"""Print the newest batch (YYYY-MM) available under ``raw/`` in object storage."""

from __future__ import annotations

import sys

from src import storage
from src.utils import BATCH_RE


def available_batches() -> list[str]:
    names = (d.removeprefix("batch=") for d in storage.list_dirs("raw/") if d.startswith("batch="))
    return sorted(n for n in names if BATCH_RE.match(n))


def main() -> None:
    batches = available_batches()
    if not batches:
        sys.exit(f"no batches found under {storage.uri('raw/')}")
    print(batches[-1])


if __name__ == "__main__":
    main()
