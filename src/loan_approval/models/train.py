"""XGBoost training with stratified k-fold cross-validation."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from loan_approval.evaluation.metrics import classification_metrics


@dataclass
class FoldResult:
    fold: int
    best_iteration: int
    metrics: dict[str, float]


def make_model(params: dict[str, Any], **overrides: Any) -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", enable_categorical=True, **{**params, **overrides})


def cross_validate(X: pd.DataFrame, y: pd.Series, params: dict[str, Any], n_splits: int,
                   early_stopping_rounds: int, random_state: int) -> tuple[list[FoldResult], np.ndarray]:
    """Stratified k-fold CV; each fold early-stops on its own validation fold.

    Returns the per-fold results and the out-of-fold scores: every row scored by the one fold model
    that did not train on it (an honest in-time prediction for the whole dataset)."""
    results, oof = [], np.full(len(X), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = make_model(params, early_stopping_rounds=early_stopping_rounds, random_state=random_state)
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        results.append(FoldResult(fold, int(model.best_iteration), classification_metrics(y.iloc[va], oof[va])))
    return results, oof


def final_n_estimators(results: list[FoldResult]) -> int:
    """Trees for the final model: mean CV best iteration, scaled up for the extra 1/k of data."""
    k = len(results)
    return int(np.mean([r.best_iteration + 1 for r in results]) * k / (k - 1))


def fit_final(X: pd.DataFrame, y: pd.Series, params: dict[str, Any], n_estimators: int,
              random_state: int) -> XGBClassifier:
    model = make_model(params, n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y, verbose=False)
    return model
