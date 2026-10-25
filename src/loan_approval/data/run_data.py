"""Rebuild the exact train / test / OOT data a logged MLflow run (model) was trained and evaluated on.

Runs don't store the data itself; they store everything needed to recreate it: the raw file's SHA-256
(tag data.source_sha256), the config used (config.yaml), which loan went to which split
(split_ids.parquet) and a content digest per split (param data_<split>_digest). This module rebuilds the
split and checks it against that digest, so the result is guaranteed to be the data the run used.
"""
from dataclasses import dataclass

import mlflow
import pandas as pd
import yaml

from loan_approval.data.load import source_path
from loan_approval.data.versioning import file_fingerprint, frame_digest


@dataclass
class RunDataset:
    run_id: str
    split: str
    X: pd.DataFrame  # model features, exactly as the model saw them
    y: pd.Series
    ids: pd.Series  # LendingClub loan id per row
    digest_verified: bool


def find_run_id(run_name: str) -> str:
    runs = mlflow.search_runs(filter_string=f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], output_format="list")
    if not runs:
        raise ValueError(f"No finished run named {run_name!r}")
    return runs[0].info.run_id


def load_run_dataset(run_id: str, split: str = "train", check_source: bool = True) -> RunDataset:
    """Return the `split` ('train', 'test' or 'oot') of a logged run, verified against its digest."""
    # Imported here to avoid a circular import (the training pipeline imports the data package)
    from loan_approval.pipelines.training_pipeline import prepare_data

    run = mlflow.get_run(run_id)
    cfg = yaml.safe_load(mlflow.artifacts.load_text(f"runs:/{run_id}/config.yaml"))
    target = cfg["data"]["target"]

    if check_source:
        logged_sha = run.data.tags.get("data.source_sha256")
        current_sha = file_fingerprint(source_path(cfg))["sha256"]
        if logged_sha and logged_sha != current_sha:
            raise ValueError(f"Raw file has changed since run {run_id[:8]} "
                             f"(logged sha256 {logged_sha[:12]}..., now {current_sha[:12]}...)")

    data = prepare_data(cfg, run.data.tags["feature_set"])
    if split not in data.X or data.X[split].empty:
        raise ValueError(f"Run {run_id[:8]} has no {split!r} split")
    X, y, ids = data.X[split], data.y[split], getattr(data.splits, split)["id"]

    # Cross-check membership with the run's split_ids.parquet, then the content digest
    logged_ids = pd.read_parquet(mlflow.artifacts.download_artifacts(run_id=run_id,
                                                                     artifact_path="split_ids.parquet"))
    expected = set(logged_ids.loc[logged_ids["split"] == split, "id"].astype(str))
    if set(ids.astype(str)) != expected:
        raise ValueError(f"Rebuilt {split} loans differ from the run's split_ids.parquet")
    logged_digest = run.data.params.get(f"data_{split}_digest")
    verified = logged_digest == frame_digest(X.assign(**{target: y.to_numpy()}))
    if logged_digest and not verified:
        raise ValueError(f"Rebuilt {split} data does not match the digest logged by run {run_id[:8]}")
    return RunDataset(run_id, split, X, y, ids, verified)
