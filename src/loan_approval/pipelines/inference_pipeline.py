"""Score loan applications: raw application fields -> clean -> features -> probability of default.

Applies the same transformations as training (data/clean.py, features/build_features.py), with the
category levels stored in the model bundle in place of the training data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from loan_approval.data.clean import replace_suspicious_values
from loan_approval.data.load import TEXT_COLUMNS
from loan_approval.features.build_features import build_features, raw_columns_for
from loan_approval.models.bundle import ModelBundle


def required_raw_columns(bundle: ModelBundle) -> list[str]:
    """Raw application fields the bundle's features are built from, plus `issue_date`."""
    return [*raw_columns_for(bundle.card["numeric"] + bundle.card["categorical"]), "issue_date"]


@dataclass
class Prepared:
    X: pd.DataFrame
    warnings: list[list[str]]  # per row: categorical values the model has not seen (scored as missing)


def prepare_features(raw: pd.DataFrame, bundle: ModelBundle) -> Prepared:
    missing = [c for c in required_raw_columns(bundle) if c not in raw]
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")
    df = raw.reset_index(drop=True).copy()
    numeric_raw = [c for c in raw_columns_for(bundle.card["numeric"]) if c not in TEXT_COLUMNS]
    df[numeric_raw] = df[numeric_raw].astype(float)
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df = replace_suspicious_values(df)

    X = build_features(df, bundle.card["numeric"], bundle.card["categorical"])
    warnings: list[list[str]] = [[] for _ in range(len(X))]
    for col, levels in bundle.categories.items():
        values = df[col].astype(object)
        unseen = values.notna() & ~values.isin(levels)
        for i in np.flatnonzero(unseen.to_numpy()):
            warnings[i].append(f"{col}={values.iloc[i]!r} not seen in training; scored as missing")
        X[col] = pd.Categorical(values.where(~unseen), categories=levels)
    return Prepared(X[bundle.features], warnings)


def predict(raw: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """One row per application: probability_of_default, decision and warnings."""
    prepared = prepare_features(raw, bundle)
    pd_scores = bundle.model.predict_proba(prepared.X)[:, 1]
    return pd.DataFrame({
        "probability_of_default": pd_scores,
        "decision": np.where(pd_scores >= bundle.threshold, "decline", "approve"),
        "warnings": prepared.warnings,
    })
