"""Optuna hyperparameter search for the XGBoost model, with every trial logged to MLflow."""
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from loan_approval.models.train import make_model


def suggest_params(trial: optuna.Trial, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params


def make_objective(X: pd.DataFrame, y: pd.Series, base_params: dict[str, Any], tuning_cfg: dict[str, Any],
                   early_stopping_rounds: int, random_state: int):
    """Objective: mean stratified k-fold validation AUC (maximised). Each trial is a nested MLflow run."""
    skf = StratifiedKFold(n_splits=tuning_cfg["n_splits"], shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **suggest_params(trial, tuning_cfg["search_space"]),
                  **tuning_cfg["fixed_params"]}
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params({**params, "trial": trial.number})
            aucs, best_iters = [], []
            for i, (tr, va) in enumerate(folds):
                model = make_model(params, early_stopping_rounds=early_stopping_rounds,
                                   random_state=random_state)
                model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
                aucs.append(roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
                best_iters.append(model.best_iteration + 1)
                mlflow.log_metric("cv_fold_auc", aucs[-1], step=i)
                trial.report(float(np.mean(aucs)), step=i)
                if trial.should_prune():
                    mlflow.set_tag("state", "pruned")
                    raise optuna.TrialPruned()
            score = float(np.mean(aucs))
            mlflow.log_metrics({"cv_auc_mean": score, "cv_auc_std": float(np.std(aucs)),
                                "best_iteration_mean": float(np.mean(best_iters))})
            mlflow.set_tag("state", "complete")
            trial.set_user_attr("best_iteration_mean", float(np.mean(best_iters)))
            trial.set_user_attr("cv_auc_std", float(np.std(aucs)))
        return score

    return objective


# +1: a higher value makes the model more conservative; -1: a lower value does
REGULARIZATION_DIRECTION = {
    "reg_lambda": 1, "reg_alpha": 1, "gamma": 1, "min_child_weight": 1,
    "max_depth": -1, "subsample": -1, "colsample_bytree": -1, "max_cat_threshold": -1,
}


def most_regularized(trials: pd.DataFrame, score_col: str = "cv_auc", std_col: str = "cv_std",
                     tolerance: float | None = None) -> pd.DataFrame:
    """Rank trials by how strongly they regularize, among those close to the best score.

    Candidates are trials within `tolerance` of the best score (default: the best trial's CV std).
    Each regularization parameter is ranked across candidates (higher rank = more conservative) and
    the ranks are averaged into `regularization_score`. Returns candidates, most regularized first.
    """
    best = trials.loc[trials[score_col].idxmax()]
    tolerance = best[std_col] if tolerance is None else tolerance
    cand = trials[trials[score_col] >= best[score_col] - tolerance].copy()
    params = [p for p in REGULARIZATION_DIRECTION if p in cand]
    ranks = pd.DataFrame({p: (cand[p] * REGULARIZATION_DIRECTION[p]).rank() for p in params})
    cand["regularization_score"] = ranks.mean(axis=1)
    return cand.sort_values(["regularization_score", score_col], ascending=False)


def select_trial(study: optuna.Study, selection: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Params of the chosen finished trial ('best' or 'most_regularized') and the ranking table."""
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df = df[df["state"] == "COMPLETE"]
    trials = pd.DataFrame({"trial": df["number"], "cv_auc": df["value"],
                           "cv_std": df["user_attrs_cv_auc_std"],
                           **{c.removeprefix("params_"): df[c] for c in df if c.startswith("params_")}})
    if selection == "best":
        ranked = trials.sort_values("cv_auc", ascending=False)
    elif selection == "most_regularized":
        ranked = most_regularized(trials)
    else:
        raise ValueError(f"Unknown selection {selection!r}")
    chosen = study.trials[int(ranked["trial"].iloc[0])]
    return dict(chosen.params), ranked


def tune(X: pd.DataFrame, y: pd.Series, base_params: dict[str, Any], tuning_cfg: dict[str, Any],
         early_stopping_rounds: int, random_state: int) -> optuna.Study:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    objective = make_objective(X, y, base_params, tuning_cfg, early_stopping_rounds, random_state)
    study.optimize(objective, n_trials=tuning_cfg["n_trials"], timeout=tuning_cfg["timeout_seconds"])
    return study
