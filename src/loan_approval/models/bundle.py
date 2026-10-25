"""Self-contained model bundle for serving: the XGBoost model plus a model card (features, category
levels, decision threshold, lineage and metrics), so scoring needs neither MLflow nor the training data.

Layout of a bundle directory:
    model.ubj        XGBoost model
    model_card.json  everything else
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import yaml
from xgboost import XGBClassifier

from loan_approval.features.build_features import resolve_feature_set

MODEL_FILE = "model.ubj"
CARD_FILE = "model_card.json"
CARD_METRICS = ["auc", "gini", "ks", "pr_auc", "logloss", "brier", "default_rate", "mean_score"]


@dataclass
class ModelBundle:
    model: XGBClassifier
    card: dict[str, Any]

    @property
    def features(self) -> list[str]:
        return self.card["features"]

    @property
    def categories(self) -> dict[str, list[str]]:
        return self.card["categories"]

    @property
    def threshold(self) -> float:
        return self.card["decision"]["threshold"]


def category_levels(model: XGBClassifier) -> dict[str, list[str]]:
    """Category levels the model was trained with, read from the model itself (training order)."""
    cats = model.get_booster().get_categories(export_to_arrow=True).to_arrow()
    return {name: values.to_pylist() for name, values in cats if values is not None}


def export_bundle(model_uri: str, out_dir: str | Path) -> Path:
    """Write the bundle for an MLflow model URI (e.g. models:/loan-default-xgboost@champion)."""
    out_dir = Path(out_dir)
    client = mlflow.MlflowClient()
    model = mlflow.xgboost.load_model(model_uri)
    model_info = mlflow.models.get_model_info(model_uri)
    run = client.get_run(model_info.run_id)
    cfg = yaml.safe_load(mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/config.yaml"))
    feature_set = run.data.tags["feature_set"]
    numeric, categorical = resolve_feature_set(cfg, feature_set)
    booster = model.get_booster()
    if set(booster.feature_names) != set(numeric) | set(categorical):
        raise ValueError(f"Model features differ from feature set {feature_set!r} in the run's config")

    metrics = run.data.metrics
    if "ks_threshold_train" not in metrics:
        raise ValueError("Run has no ks_threshold_train metric; build its model report first")
    card = {
        "model_uri": model_uri,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "feature_set": feature_set,
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data": {"population": json.loads(json.dumps(cfg["data"]["population"], default=str)),
                          "oot_start": str(cfg["split"]["oot_start"]),
                          "source_sha256": run.data.tags.get("data.source_sha256")},
        "target": "probability that the loan is charged off (Default = 1)",
        "features": booster.feature_names,  # model input order
        "numeric": numeric,
        "categorical": categorical,
        "categories": category_levels(model),
        "decision": {"threshold": metrics["ks_threshold_train"],
                     "rule": "decline when probability_of_default >= threshold (KS-optimal on train)"},
        "metrics": {split: {m: metrics[f"{split}_{m}"] for m in CARD_METRICS if f"{split}_{m}" in metrics}
                    for split in ["test", "oot"]},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / MODEL_FILE)
    (out_dir / CARD_FILE).write_text(json.dumps(card, indent=2) + "\n")
    return out_dir


def load_bundle(bundle_dir: str | Path) -> ModelBundle:
    bundle_dir = Path(bundle_dir)
    model = XGBClassifier(enable_categorical=True)
    model.load_model(bundle_dir / MODEL_FILE)
    card = json.loads((bundle_dir / CARD_FILE).read_text())
    if model.get_booster().feature_names != card["features"]:
        raise ValueError(f"{bundle_dir}: model features do not match {CARD_FILE}")
    return ModelBundle(model, card)
