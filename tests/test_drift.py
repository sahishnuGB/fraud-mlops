import numpy as np
import pandas as pd

from src.drift_check import drift_report, psi_categorical, psi_numeric, to_markdown


def test_psi_identical_is_zero():
    s = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi_numeric(s, s) < 1e-9


def test_psi_detects_mean_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi_numeric(ref, pd.Series(rng.normal(1, 1, 5000))) > 0.2


def test_psi_categorical_handles_new_category():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 40 + ["b"] * 40 + ["c"] * 20)
    assert psi_categorical(ref, ref) == 0
    assert psi_categorical(ref, cur) > 0.2


def test_psi_constant_reference():
    assert psi_numeric(pd.Series([1.0] * 100), pd.Series([1.0] * 100)) == 0


def test_report_flags_drifted_batch(batch_df, drifted_df):
    stable = drift_report(batch_df, batch_df, threshold=0.2, bins=10)
    assert not stable["drift_detected"]
    drifted = drift_report(batch_df, drifted_df, threshold=0.2, bins=10)
    assert drifted["drift_detected"]
    assert {"avg_amount_30d", "txn_count_24h"} <= set(drifted["drifted_features"])
    md = to_markdown(drifted, "2026-02", "2026-01")
    assert "Drift detected" in md and "| avg_amount_30d |" in md
