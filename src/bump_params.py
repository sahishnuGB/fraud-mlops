"""Set ``batch`` (and optionally ``reference_batch``) in params.yaml, keeping comments intact.

python -m src.bump_params --batch 2026-07
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.utils import PARAMS_FILE, check_batch


def set_top_level(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf'^{key}:[ \t]*"?[^"#\n]*"?([ \t]*#.*)?$', re.MULTILINE)
    new, n = pattern.subn(lambda m: f'{key}: "{value}"{m.group(1) or ""}', text, count=1)
    if n == 0:
        raise KeyError(f"top-level key {key!r} not found in params file")
    return new


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--reference-batch")
    parser.add_argument("--file", type=Path, default=PARAMS_FILE)
    args = parser.parse_args(argv)

    text = args.file.read_text(encoding="utf-8")
    text = set_top_level(text, "batch", check_batch(args.batch))
    if args.reference_batch:
        text = set_top_level(text, "reference_batch", check_batch(args.reference_batch))
    args.file.write_text(text, encoding="utf-8")
    print(
        f"{args.file.name}: batch={args.batch}"
        + (f" reference_batch={args.reference_batch}" if args.reference_batch else "")
    )


if __name__ == "__main__":
    main()
