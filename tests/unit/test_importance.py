import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from loan_approval.evaluation.explain import plot_shap_waterfall, shap_explain
from loan_approval.evaluation.importance import feature_importance, plot_top_features


def test_importance_sums_to_100_and_includes_unused_features():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.normal(size=500), "noise": rng.normal(size=500),
                      "constant": np.zeros(500)})
    y = (X["signal"] + 0.3 * rng.normal(size=500) > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=2).fit(X, y)

    imp = feature_importance(model)

    assert set(imp["feature"]) == {"signal", "noise", "constant"}
    assert abs(imp["importance_pct"].sum() - 100) < 1e-9
    assert imp["feature"].iloc[0] == "signal"
    assert imp.loc[imp["feature"] == "constant", "importance_pct"].item() == 0
    assert imp["cumulative_pct"].iloc[-1] == imp["importance_pct"].sum()
    assert plot_top_features(imp, top_n=2).axes


def test_shap_contributions_add_up_to_prediction_and_waterfall_draws():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (X["a"] > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=2).fit(X, y)

    contribs, base = shap_explain(model, X)

    logit = np.log(model.predict_proba(X)[:, 1] / (1 - model.predict_proba(X)[:, 1]))
    assert np.allclose(base + contribs.sum(axis=1), logit, atol=1e-4)
    assert plot_shap_waterfall(contribs.iloc[0], X.iloc[0], base, top_n=1).axes
