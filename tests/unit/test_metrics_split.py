import numpy as np
import pandas as pd

from loan_approval.data.split import split_data
from loan_approval.evaluation.metrics import (
    classification_metrics,
    confusion_at,
    ks_statistic,
    ks_threshold,
    psi,
)


def test_metrics_on_perfect_and_random_scores():
    y = np.array([0, 0, 1, 1])
    perfect = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["auc"] == 1.0 and perfect["gini"] == 1.0 and perfect["ks"] == 1.0
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_psi_is_zero_for_identical_and_positive_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=10_000)
    assert psi(a, a) < 1e-9
    assert psi(a, a + 0.3) > 0.1


def test_split_data_separates_oot_by_issue_date_and_stratifies():
    df = pd.DataFrame({
        "issue_date": pd.to_datetime(["2014-01-01"] * 80 + ["2015-08-01"] * 20),
        "Default": [0, 1] * 50,
    })
    s = split_data(df, "Default", "2015-07-01", test_size=0.25, random_state=0)
    assert len(s.oot) == 20 and (s.oot["issue_date"] >= "2015-07-01").all()
    assert len(s.train) == 60 and len(s.test) == 20
    assert s.train["Default"].mean() == s.test["Default"].mean() == 0.5


def test_ks_threshold_and_confusion():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    t, ks = ks_threshold(y, score)
    assert t == 0.7 and ks == 1.0
    cm = confusion_at(y, score, t)
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (3, 0, 0, 2)
    assert cm["approval_rate"] == 0.6 and cm["default_rate_approved"] == 0.0


def test_split_data_without_test_set_keeps_all_development_loans_in_train():
    df = pd.DataFrame({"issue_date": pd.to_datetime(["2014-01-01"] * 8 + ["2015-08-01"] * 2),
                       "Default": [0, 1] * 5})
    s = split_data(df, "Default", "2015-07-01", test_size=0, random_state=0)
    assert len(s.train) == 8 and s.test.empty and len(s.oot) == 2
