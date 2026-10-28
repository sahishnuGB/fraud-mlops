"""Stage 7: Population Stability Index of the current batch against the reference batch.

PSI = sum((cur% - ref%) * ln(cur% / ref%)) over bins.
Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.

Numeric features use quantile bins taken from the reference batch; categorical and
binary features use one bin per value. Drift is reported, not enforced: the pipeline
keeps going and the summary lands in the PR / CML report.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.schema import DRIFT_CATEGORICAL, DRIFT_NUMERIC, TARGET, monitoring_view  # noqa: E402
from src.utils import (  # noqa: E402
    BATCH_FILE,
    PLOTS_DIR,
    RAW_CURRENT,
    RAW_REFERENCE,
    REPORTS_DIR,
    get_logger,
    load_params,
    write_json,
)

log = get_logger("drift_check")

EPS = 1e-4  # floor for empty bins so ln() stays finite


def _psi_from_shares(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(ref, EPS, None)
    cur = np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference
        return psi_categorical(reference, current)
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference.dropna(), edges)[0] / max(reference.notna().sum(), 1)
    cur = np.histogram(current.dropna(), edges)[0] / max(current.notna().sum(), 1)
    return _psi_from_shares(ref, cur)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.value_counts(normalize=True)
    cur = current.value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    return _psi_from_shares(ref.reindex(cats, fill_value=0).to_numpy(), cur.reindex(cats, fill_value=0).to_numpy())


def severity(psi: float, threshold: float) -> str:
    if psi >= threshold:
        return "drift"
    if psi >= threshold / 2:
        return "warning"
    return "ok"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, threshold: float, bins: int) -> dict:
    ref, cur = monitoring_view(reference), monitoring_view(current)
    psi = {c: psi_numeric(ref[c], cur[c], bins) for c in DRIFT_NUMERIC}
    psi |= {c: psi_categorical(ref[c], cur[c]) for c in DRIFT_CATEGORICAL}
    psi = {k: round(v, 6) for k, v in psi.items()}
    drifted = sorted((k for k, v in psi.items() if v >= threshold), key=lambda k: -psi[k])
    return {
        "psi": psi,
        "max_psi": max(psi.values()),
        "psi_threshold": threshold,
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "label_rate_reference": round(float(reference[TARGET].mean()), 6),
        "label_rate_current": round(float(current[TARGET].mean()), 6),
    }


def to_markdown(report: dict, batch: str, reference_batch: str) -> str:
    icon = {"drift": "🔴 drift", "warning": "🟡 warning", "ok": "🟢 ok"}
    t = report["psi_threshold"]
    headline = (
        f"**Drift detected** in {report['n_drifted']} feature(s): {', '.join(report['drifted_features'])}"
        if report["drift_detected"]
        else "No feature exceeded the PSI threshold."
    )
    lines = [
        f"### Data drift: `{batch}` vs reference `{reference_batch}`",
        "",
        f"{headline} (threshold {t}, max PSI {report['max_psi']:.3f})",
        "",
        f"Fraud rate: {report['label_rate_reference']:.3%} (reference) -> {report['label_rate_current']:.3%} (current)",
        "",
        "| feature | PSI | status |",
        "|---|---:|---|",
    ]
    for feat, value in sorted(report["psi"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {feat} | {value:.4f} | {icon[severity(value, t)]} |")
    return "\n".join(lines) + "\n"


def plot_psi(report: dict, path) -> None:
    items = sorted(report["psi"].items(), key=lambda kv: kv[1])
    t = report["psi_threshold"]
    colors = ["#d62728" if v >= t else "#ff9f0e" if v >= t / 2 else "#2ca02c" for _, v in items]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([k for k, _ in items], [v for _, v in items], color=colors)
    ax.axvline(t, ls="--", color="black", lw=1, label=f"threshold {t}")
    ax.set(xlabel="PSI", title="Feature drift (PSI vs reference)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    params = load_params()
    reference = pd.read_parquet(RAW_REFERENCE / BATCH_FILE)
    current = pd.read_parquet(RAW_CURRENT / BATCH_FILE)
    report = drift_report(reference, current, params["psi_threshold"], params["drift"]["bins"])
    report.update(batch=params["batch"], reference_batch=params["reference_batch"])
    write_json(REPORTS_DIR / "drift.json", report)
    (REPORTS_DIR / "drift.md").write_text(
        to_markdown(report, params["batch"], params["reference_batch"]), encoding="utf-8"
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_psi(report, PLOTS_DIR / "drift_psi.png")
    log.info("max PSI %.4f, drifted: %s", report["max_psi"], report["drifted_features"] or "none")


if __name__ == "__main__":
    main()
