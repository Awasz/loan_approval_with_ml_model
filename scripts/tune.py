"""CLI entry point: Optuna hyperparameter search, then a full training run with the best params."""
import argparse

from loan_approval.pipelines.training_pipeline import tune_and_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune XGBoost with Optuna and log everything to MLflow")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--feature-set", default=None, help="Feature set from the config (default: features.default_set)")
    parser.add_argument("--n-trials", type=int, default=None, help="Override tuning.n_trials")
    args = parser.parse_args()
    tune_and_train(args.config, feature_set=args.feature_set, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
