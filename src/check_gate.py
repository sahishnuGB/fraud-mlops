"""Exit non-zero when the PR-AUC quality gate failed (used by CI after reporting)."""

from __future__ import annotations

import sys

from src.utils import REPORTS_DIR, read_json


def main() -> None:
    m = read_json(REPORTS_DIR / "metrics.json")
    if not m["gate_passed"]:
        sys.exit(f"PR-AUC gate FAILED: {m['pr_auc']:.4f} < min_pr_auc {m['min_pr_auc']}")
    print(f"PR-AUC gate passed: {m['pr_auc']:.4f} >= {m['min_pr_auc']}")


if __name__ == "__main__":
    main()
