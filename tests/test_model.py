import numpy as np
import pytest

from src.evaluate import compute_metrics
from src.featurize import build_features
from src.schema import TARGET
from src.split import split_frame
from src.train import build_pipeline, train


def test_features_are_numeric_and_complete(batch_df):
    feats = build_features(batch_df)
    assert len(feats) == len(batch_df)
    assert not feats.isna().any().any()
    assert all(dtype.kind in "iuf" for dtype in feats.dtypes)
    assert feats.filter(like="cat_").sum(axis=1).eq(1).all()


def test_feature_columns_stable_across_batches(batch_df, drifted_df):
    assert list(build_features(batch_df.head(10)).columns) == list(build_features(drifted_df).columns)


def test_split_is_stratified(batch_df):
    train_df, test_df = split_frame(batch_df, 0.25, 42)
    assert len(test_df) == pytest.approx(0.25 * len(batch_df), abs=1)
    assert test_df[TARGET].mean() == pytest.approx(batch_df[TARGET].mean(), abs=0.003)


@pytest.mark.parametrize("model_type", ["hgb", "logreg"])
@pytest.mark.parametrize("sampling", ["none", "undersample", "smote"])
def test_train_all_combinations(batch_df, params, model_type, sampling):
    params["model"]["type"] = model_type
    params["sampling"] = sampling
    params["model"]["hgb"]["max_iter"] = 50
    train_df, test_df = split_frame(build_features(batch_df), 0.25, 42)
    bundle = train(train_df, params)
    scores = bundle["pipeline"].predict_proba(test_df[bundle["features"]])[:, 1]
    assert scores.shape == (len(test_df),)
    m = compute_metrics(test_df[TARGET].to_numpy(), scores, 0.5, min_pr_auc=0.0)
    # Signal is learnable: far better than the ~2% random baseline.
    assert m["pr_auc"] > 0.2
    assert m["roc_auc"] > 0.8


def test_bad_model_choice(params):
    params["model"]["type"] = "xgboost"
    with pytest.raises(ValueError, match="model.type"):
        build_pipeline(params)


def test_gate():
    y = np.array([0, 0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert compute_metrics(y, perfect, 0.5, min_pr_auc=0.9)["gate_passed"]
    assert not compute_metrics(y, perfect[::-1], 0.5, min_pr_auc=0.9)["gate_passed"]
