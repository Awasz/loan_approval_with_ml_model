import mlflow
import numpy as np
import pandas as pd

from loan_approval.models.tune import most_regularized, select_trial, tune

TUNING = {
    "n_trials": 3, "timeout_seconds": 60, "n_splits": 2,
    "fixed_params": {"colsample_bytree": 0.7},
    "search_space": {
        "max_depth": {"type": "int", "low": 2, "high": 4},
        "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
    },
}


def test_tune_keeps_fixed_params_and_logs_nested_trials(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series((X["a"] + rng.normal(size=400) > 0).astype(int))

    with mlflow.start_run() as parent:
        study = tune(X, y, {"n_estimators": 30}, TUNING, early_stopping_rounds=5, random_state=0)

    assert len(study.trials) == 3
    assert set(study.best_params) == {"max_depth", "learning_rate"}
    children = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(children) == 3
    assert (children["params.colsample_bytree"] == "0.7").all()


def test_most_regularized_picks_conservative_trial_among_near_best():
    trials = pd.DataFrame({
        "trial": ["loose", "conservative", "too_weak"],
        "cv_auc": [0.700, 0.699, 0.680],
        "cv_std": [0.003, 0.003, 0.003],
        "max_depth": [8, 4, 2],
        "reg_lambda": [0.01, 10.0, 50.0],
        "min_child_weight": [1.0, 100.0, 200.0],
    })
    ranked = most_regularized(trials)
    assert ranked["trial"].tolist() == ["conservative", "loose"]  # too_weak is outside 1 std of best


def test_select_trial_returns_most_regularized_params(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series((X["a"] + rng.normal(size=400) > 0).astype(int))
    with mlflow.start_run():
        study = tune(X, y, {"n_estimators": 30}, TUNING, early_stopping_rounds=5, random_state=0)

    params, ranked = select_trial(study, "most_regularized")
    assert set(params) == {"max_depth", "learning_rate"}
    assert ranked["cv_auc"].max() == study.best_value
    assert select_trial(study, "best")[0] == study.best_params
