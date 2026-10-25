import numpy as np
import pandas as pd

from loan_approval.features.build_features import (
    align_categories,
    build_features,
    raw_columns_for,
    resolve_feature_set,
)


def _raw() -> pd.DataFrame:
    return pd.DataFrame({
        "term": [" 36 months", " 60 months"],
        "fico_range_low": [690.0, 700.0],
        "fico_range_high": [694.0, 704.0],
        "emp_length": ["< 1 year", None],
        "emp_title": ["Teacher", None],
        "earliest_cr_line": ["Jan-2005", "Jan-2015"],
        "issue_date": pd.to_datetime(["2015-01-01", "2015-01-01"]),
        "purpose": ["car", "wedding"],
    })


def test_raw_columns_for_expands_derived_features():
    assert raw_columns_for(["fico", "loan_amnt", "term_months"]) == [
        "fico_range_low", "fico_range_high", "loan_amnt", "term"]


def test_build_features_derivations():
    X = build_features(_raw(), ["term_months", "fico", "emp_length_years", "emp_title_missing",
                                "credit_history_months"], ["purpose"])
    assert X["term_months"].tolist() == [36.0, 60.0]
    assert X["fico"].tolist() == [692.0, 702.0]
    assert X["emp_length_years"].iloc[0] == 0.0 and np.isnan(X["emp_length_years"].iloc[1])
    assert X["emp_title_missing"].tolist() == [0.0, 1.0]
    assert round(X["credit_history_months"].iloc[0]) == 120
    assert X["purpose"].dtype == "category"


def test_align_categories_uses_train_levels_only():
    train = pd.DataFrame({"purpose": pd.Categorical(["car", "car"], categories=["car", "wedding"])})
    oot = pd.DataFrame({"purpose": pd.Categorical(["car", "wedding"])})

    train_a, oot_a = align_categories(train, oot)

    assert list(train_a["purpose"].cat.categories) == ["car"]
    assert oot_a["purpose"].isna().tolist() == [False, True]


def test_resolve_feature_set_with_base_and_exclude():
    cfg = {"features": {"sets": {
        "full": {"numeric": ["fico", "dti"], "categorical": ["purpose", "addr_state"]},
        "no_state": {"base": "full", "exclude": ["addr_state"]},
    }}}
    assert resolve_feature_set(cfg, "no_state") == (["fico", "dti"], ["purpose"])
