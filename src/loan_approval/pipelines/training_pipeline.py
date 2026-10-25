"""End-to-end training: load -> clean -> features -> split -> k-fold CV -> final fit -> evaluate -> MLflow."""
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.xgboost
import optuna
import pandas as pd
import yaml
from xgboost import XGBClassifier

from loan_approval.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from loan_approval.data.clean import replace_suspicious_values
from loan_approval.data.load import load_population, source_path
from loan_approval.data.split import Splits, split_data, summarize
from loan_approval.data.versioning import file_fingerprint, frame_digest
from loan_approval.evaluation.importance import feature_importance, plot_top_features
from loan_approval.evaluation.metrics import classification_metrics, psi
from loan_approval.features.build_features import (
    align_categories,
    build_features,
    raw_columns_for,
    resolve_feature_set,
)
from loan_approval.models.train import cross_validate, final_n_estimators, fit_final
from loan_approval.models.tune import most_regularized, select_trial, tune
from loan_approval.utils.logger import get_logger

log = get_logger(__name__)


def setup_mlflow(cfg: dict[str, Any]) -> str:
    """Point MLflow at the project's tracking store and create the experiment if needed."""
    ml = cfg["mlflow"]
    uri = ml["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{PROJECT_ROOT / uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(ml["experiment_name"]) is None:
        mlflow.create_experiment(ml["experiment_name"],
                                 artifact_location=(PROJECT_ROOT / ml["artifact_root"]).as_uri())
    mlflow.set_experiment(ml["experiment_name"])
    return uri


def log_feature_importance(model: XGBClassifier, title: str, top_n: int = 20,
                           run_id: str | None = None) -> pd.DataFrame:
    """Log importance (% of total gain, sums to 100) as CSV and a top-N chart to the active run,
    or to `run_id` if given (e.g. to update an existing run)."""
    importance = feature_importance(model)
    fig = plot_top_features(importance, top_n, title)
    client = mlflow.MlflowClient()
    run_id = run_id or mlflow.active_run().info.run_id
    client.log_text(run_id, importance.to_csv(index=False, float_format="%.6g"), "feature_importance.csv")
    client.log_figure(run_id, fig, f"feature_importance_top{top_n}.png")
    plt.close(fig)
    return importance


def log_dataset_versions(source: Path, splits: Splits, frames: dict[str, pd.DataFrame],
                         ys: dict[str, pd.Series], target: str) -> None:
    """Record which data the run used: source file fingerprint, per-split MLflow datasets, split ids."""
    mlflow.set_tags({f"data.source_{k}": v for k, v in file_fingerprint(source).items()})
    for name, X in frames.items():
        data = X.assign(**{target: ys[name].to_numpy()})
        mlflow.log_param(f"data_{name}_digest", frame_digest(data))
        with warnings.catch_warnings():
            # The 0/1 integer target triggers MLflow's "integer columns can't hold NaN" hint, and
            # MLflow registers its local-file source class twice, which triggers an ambiguity warning
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted")
            dataset = mlflow.data.from_pandas(data, source=str(source), name=f"lending_club_{name}",
                                              targets=target)
            mlflow.log_input(dataset, context=name)
    ids = pd.concat([pd.DataFrame({"id": getattr(splits, name)["id"].to_numpy(), "split": name})
                     for name in frames])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "split_ids.parquet"
        ids.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))


@dataclass
class PreparedData:
    splits: Splits
    summary: pd.DataFrame
    X: dict[str, pd.DataFrame]  # keys: train, test, oot
    y: dict[str, pd.Series]


def prepare_data(cfg: dict[str, Any], feature_set: str) -> PreparedData:
    target, split_cfg = cfg["data"]["target"], cfg["split"]
    numeric, categorical = resolve_feature_set(cfg, feature_set)

    log.info("Loading population (feature set %r)", feature_set)
    df = replace_suspicious_values(load_population(cfg, raw_columns_for(numeric + categorical)))
    splits = split_data(df, target, split_cfg["oot_start"], split_cfg["test_size"],
                        split_cfg["random_state"])
    summary = summarize(splits, target)
    log.info("Splits:\n%s", summary)

    names = ["train", "test", "oot"]
    X = align_categories(*(build_features(getattr(splits, n), numeric, categorical) for n in names))
    return PreparedData(splits, summary, dict(zip(names, X)),
                        {n: getattr(splits, n)[target] for n in names})


def run(config_path: str | Path = DEFAULT_CONFIG, feature_set: str | None = None,
        run_name: str | None = None, params_override: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None, data: PreparedData | None = None) -> dict[str, dict[str, float]]:
    """Train with k-fold CV, fit the final model and evaluate on train / test / OOT, logging to MLflow.

    `params_override` replaces model params from the config (e.g. Optuna's best params).
    """
    cfg = load_config(config_path)
    target, split_cfg, model_cfg = cfg["data"]["target"], cfg["split"], cfg["model"]
    feature_set = feature_set or cfg["features"]["default_set"]
    seed = split_cfg["random_state"]
    params = {**model_cfg["params"], **(params_override or {})}

    data = data or prepare_data(cfg, feature_set)
    splits, split_summary, ys = data.splits, data.summary, data.y
    X_train = data.X["train"]
    frames = {k: X for k, X in data.X.items() if len(X)}  # train, test (unless test_size = 0), oot
    has_test = "test" in frames

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=run_name or f"xgb_{feature_set}"):
        mlflow.set_tags({"model": model_cfg["name"], "feature_set": feature_set,
                         "oot_start": split_cfg["oot_start"],
                         "split_design": "train/test/oot" if has_test else "kfold/oot",
                         "population": str(cfg["data"]["population"]), **(tags or {})})
        mlflow.log_params({**params,
                           "early_stopping_rounds": model_cfg["early_stopping_rounds"],
                           "n_splits": split_cfg["n_splits"], "test_size": split_cfg["test_size"],
                           "random_state": seed, "n_features": X_train.shape[1],
                           **{f"n_{k}": len(ys[k]) for k in frames}})
        log.info("Logging dataset versions")
        log_dataset_versions(source_path(cfg), splits, frames, ys, target)

        log.info("Running %d-fold CV on %d rows", split_cfg["n_splits"], len(X_train))
        folds, oof = cross_validate(X_train, ys["train"], params, split_cfg["n_splits"],
                                    model_cfg["early_stopping_rounds"], seed)
        mlflow.log_metrics({f"oof_{k}": v for k, v in classification_metrics(ys["train"], oof).items()})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oof_predictions.parquet"
            pd.DataFrame({"id": splits.train["id"].to_numpy(), target: ys["train"].to_numpy(),
                          "oof_score": oof}).to_parquet(path, index=False)
            mlflow.log_artifact(str(path))
        fold_table = pd.DataFrame([{"fold": f.fold, "best_iteration": f.best_iteration, **f.metrics}
                                   for f in folds])
        for f in folds:
            mlflow.log_metrics({f"cv_fold_{k}": v for k, v in f.metrics.items()}, step=f.fold)
        for k in ["auc", "gini", "ks", "pr_auc", "logloss"]:
            mlflow.log_metrics({f"cv_{k}_mean": fold_table[k].mean(), f"cv_{k}_std": fold_table[k].std()})

        n_estimators = final_n_estimators(folds)
        mlflow.log_param("final_n_estimators", n_estimators)
        log.info("CV AUC %.4f ± %.4f; fitting final model with %d trees",
                 fold_table["auc"].mean(), fold_table["auc"].std(), n_estimators)
        model = fit_final(X_train, ys["train"], params, n_estimators, seed)

        scores = {k: model.predict_proba(X)[:, 1] for k, X in frames.items()}
        results = {k: classification_metrics(ys[k], scores[k]) for k in frames}
        for k, m in results.items():
            mlflow.log_metrics({f"{k}_{name}": v for name, v in m.items()})
        # Score stability: OOT against the in-time hold-out (test, or the out-of-fold scores)
        reference = scores["test"] if has_test else oof
        mlflow.log_metric(f"oot_score_psi_vs_{'test' if has_test else 'oof'}", psi(reference, scores["oot"]))

        log_feature_importance(model, title=f"XGBoost ({feature_set} features)")
        mlflow.log_text(split_summary.to_csv(), "split_summary.csv")
        mlflow.log_text(fold_table.to_csv(index=False), "cv_folds.csv")
        mlflow.log_text(yaml.safe_dump(cfg, sort_keys=False), "config.yaml")
        mlflow.xgboost.log_model(model, name="model")

        log.info("AUC %s | out-of-fold %.4f", " | ".join(f"{k} {m['auc']:.4f}" for k, m in results.items()),
                 classification_metrics(ys["train"], oof)["auc"])
    return results


def tune_and_train(config_path: str | Path = DEFAULT_CONFIG, feature_set: str | None = None,
                   n_trials: int | None = None) -> dict[str, dict[str, float]]:
    """Optuna search on the training set only (test and OOT stay untouched), logged as a parent MLflow
    run with one nested run per trial; then a full training run with the best params."""
    cfg = load_config(config_path)
    feature_set = feature_set or cfg["features"]["default_set"]
    tuning_cfg = {**cfg["tuning"], **({"n_trials": n_trials} if n_trials else {})}
    model_cfg, seed = cfg["model"], cfg["split"]["random_state"]
    data = prepare_data(cfg, feature_set)

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=f"optuna_{feature_set}") as parent:
        mlflow.set_tags({"stage": "tuning", "feature_set": feature_set})
        mlflow.log_params({"n_trials": tuning_cfg["n_trials"], "timeout_seconds": tuning_cfg["timeout_seconds"],
                           "tuning_n_splits": tuning_cfg["n_splits"],
                           **{f"fixed_{k}": v for k, v in tuning_cfg["fixed_params"].items()}})
        mlflow.log_dict(tuning_cfg["search_space"], "search_space.yaml")
        log.info("Optuna: %d trials, %d-fold CV, fixed %s", tuning_cfg["n_trials"], tuning_cfg["n_splits"],
                 tuning_cfg["fixed_params"])
        study = tune(data.X["train"], data.y["train"], model_cfg["params"], tuning_cfg,
                     model_cfg["early_stopping_rounds"], seed)

        trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        states = trials["state"].value_counts().to_dict()
        selection = tuning_cfg.get("selection", "best")
        chosen_params, ranked = select_trial(study, selection)
        chosen = {**chosen_params, **tuning_cfg["fixed_params"]}
        chosen_trial = f"trial_{int(ranked['trial'].iloc[0]):03d}"
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_params({f"selected_{k}": v for k, v in chosen.items()})
        mlflow.set_tags({"selection": selection, "selected_trial": chosen_trial})
        mlflow.log_metrics({"best_cv_auc": study.best_value, "selected_cv_auc": float(ranked["cv_auc"].iloc[0]),
                            "n_complete": states.get("COMPLETE", 0), "n_pruned": states.get("PRUNED", 0)})
        mlflow.log_text(trials.to_csv(index=False), "optuna_trials.csv")
        mlflow.log_text(ranked.to_csv(index=False), "trial_selection.csv")
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        mlflow.log_figure(ax.figure, "optuna_history.png")
        plt.close(ax.figure)
        log.info("Best trial %d: CV AUC %.4f. Selected (%s): %s, CV AUC %.4f, params %s",
                 study.best_trial.number, study.best_value, selection, chosen_trial,
                 ranked["cv_auc"].iloc[0], chosen)

    suffix = "regularized" if selection == "most_regularized" else "tuned"
    return run(config_path, feature_set, run_name=f"xgb_{feature_set}_{suffix}", params_override=chosen,
               tags={"stage": "tuned", "tuning_run_id": parent.info.run_id, "selected_trial": chosen_trial,
                     "selection": selection}, data=data)


def train_most_regularized(config_path: str | Path = DEFAULT_CONFIG,
                           tuning_run_id: str | None = None) -> dict[str, dict[str, float]]:
    """Train a final model with the most regularized finished trial (within 1 CV std of the best)
    of an Optuna tuning run (default: the latest one)."""
    cfg = load_config(config_path)
    setup_mlflow(cfg)
    client = mlflow.MlflowClient()
    if tuning_run_id is None:
        tuning_run_id = mlflow.search_runs(filter_string="tags.stage = 'tuning'",
                                           order_by=["attributes.start_time DESC"]).iloc[0]["run_id"]
    feature_set = client.get_run(tuning_run_id).data.tags["feature_set"]
    space, fixed = cfg["tuning"]["search_space"], cfg["tuning"]["fixed_params"]

    children = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{tuning_run_id}' and tags.state = 'complete'")
    trials = pd.DataFrame({"trial": children["tags.mlflow.runName"],
                           "cv_auc": children["metrics.cv_auc_mean"],
                           "cv_std": children["metrics.cv_auc_std"],
                           **{p: children[f"params.{p}"].astype(float) for p in space}})
    ranked = most_regularized(trials)
    client.log_text(tuning_run_id, ranked.to_csv(index=False), "regularized_selection.csv")
    chosen = ranked.iloc[0]
    params = {p: int(chosen[p]) if spec["type"] == "int" else float(chosen[p]) for p, spec in space.items()}
    log.info("Selected %s (CV AUC %.4f, regularization score %.2f) from %d finished trials",
             chosen["trial"], chosen["cv_auc"], chosen["regularization_score"], len(trials))

    return run(config_path, feature_set, run_name=f"xgb_{feature_set}_regularized",
               params_override={**params, **fixed},
               tags={"stage": "tuned", "tuning_run_id": tuning_run_id, "selected_trial": chosen["trial"],
                     "selection": "most regularized within 1 CV std of best"})
