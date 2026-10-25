import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from loan_approval.api.app import app
from loan_approval.api.schemas import EXAMPLE, Application
from loan_approval.config import load_config
from loan_approval.features.build_features import raw_columns_for, resolve_feature_set
from loan_approval.models.bundle import CARD_FILE, MODEL_FILE, category_levels
from loan_approval.models.train import make_model

LEVELS = {"home_ownership": ["MORTGAGE", "OWN", "RENT"], "verification_status": ["Not Verified", "Verified"],
          "purpose": ["car", "credit_card", "debt_consolidation"], "addr_state": ["CA", "NY", "TX"]}
OPTIONAL = {"emp_length", "emp_title", "issue_date", "mths_since_last_record", "mths_since_recent_bc_dlq",
            "mths_since_last_major_derog", "mths_since_recent_revol_delinq", "mths_since_last_delinq",
            "mths_since_recent_inq", "mo_sin_old_il_acct", "num_tl_120dpd_2m", "bc_util", "percent_bc_gt_75",
            "bc_open_to_buy", "mths_since_recent_bc", "revol_util", "pct_tl_nvr_dlq", "avg_cur_bal"}
# Required fields only, with categories the test model knows
APPLICATION = {**{k: v for k, v in EXAMPLE.items() if k not in OPTIONAL},
               "home_ownership": "RENT", "verification_status": "Verified", "purpose": "credit_card",
               "addr_state": "NY", "issue_date": "2015-10-01"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """API serving a small model trained on random data with the real `full` feature set."""
    numeric, categorical = resolve_feature_set(load_config(), "full")
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({c: rng.normal(size=n) for c in numeric})
    for c in categorical:
        X[c] = pd.Categorical(rng.choice(LEVELS[c], n), categories=LEVELS[c])
    y = (X["fico"] + rng.normal(size=n) > 0).astype(int)
    model = make_model({"n_estimators": 5, "max_depth": 2, "tree_method": "hist"})
    model.fit(X, y)

    bundle_dir = tmp_path_factory.mktemp("bundle")
    model.save_model(bundle_dir / MODEL_FILE)
    card = {"run_id": "0123456789abcdef", "run_name": "test_model", "features": list(X.columns),
            "numeric": numeric, "categorical": categorical, "categories": category_levels(model),
            "decision": {"threshold": 0.5}}
    (bundle_dir / CARD_FILE).write_text(json.dumps(card))

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_DIR", str(bundle_dir))
        with TestClient(app) as c:
            yield c


def test_schema_covers_exactly_the_raw_feature_columns():
    numeric, categorical = resolve_feature_set(load_config(), "full")
    assert set(Application.model_fields) - {"issue_date"} == set(raw_columns_for(numeric + categorical))


def test_only_fields_missing_in_training_are_optional():
    assert {name for name, f in Application.model_fields.items() if not f.is_required()} == OPTIONAL


def test_example_is_a_valid_application():
    Application(**EXAMPLE)


def test_to_raw_uses_lendingclub_formats():
    raw = Application(**{**APPLICATION, "earliest_cr_line": "2004-08-01"}, emp_length=0, emp_title="").to_raw()
    assert raw["term"] == " 36 months"
    assert raw["emp_length"] == "0 years"
    assert raw["emp_title"] is None
    assert raw["earliest_cr_line"] == "Aug-2004"


def test_predict_minimal_application(client):
    r = client.post("/predict", json=APPLICATION)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["probability_of_default"] < 1
    assert body["decision"] == ("decline" if body["probability_of_default"] >= 0.5 else "approve")
    assert body["threshold"] == 0.5 and body["warnings"] == [] and body["model"] == "test_model (01234567)"


def test_unseen_category_is_scored_as_missing_with_warning(client):
    r = client.post("/predict", json={**APPLICATION, "addr_state": "ZZ"})
    assert r.status_code == 200
    assert r.json()["warnings"] == ["addr_state='ZZ' not seen in training; scored as missing"]


def test_invalid_input_is_rejected(client):
    assert client.post("/predict", json={**APPLICATION, "term": 48}).status_code == 422
    assert client.post("/predict", json={**APPLICATION, "unknown_field": 1}).status_code == 422
    for required in ["loan_amnt", "tot_cur_bal"]:
        incomplete = {k: v for k, v in APPLICATION.items() if k != required}
        assert client.post("/predict", json=incomplete).status_code == 422


def test_batch_matches_single_predictions(client):
    apps = [APPLICATION, {**APPLICATION, "fico_range_low": 800, "fico_range_high": 804, "emp_length": 10,
                          "emp_title": "Nurse", "earliest_cr_line": "1999-01-01"}]
    batch = client.post("/predict/batch", json={"applications": apps}).json()["predictions"]
    single = [client.post("/predict", json=a).json()["probability_of_default"] for a in apps]
    assert [p["probability_of_default"] for p in batch] == pytest.approx(single)


def test_health_and_model_card(client):
    assert client.get("/health").json() == {"status": "ok", "model": "test_model (01234567)"}
    assert client.get("/model").json()["categories"]["addr_state"] == LEVELS["addr_state"]
