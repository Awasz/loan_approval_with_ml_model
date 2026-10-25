"""CLI entry point for running the training pipeline."""
import argparse

from loan_approval.pipelines.training_pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the loan default model and log it to MLflow")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--feature-set", default=None, help="Feature set from the config (default: features.default_set)")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run(args.config, feature_set=args.feature_set, run_name=args.run_name)


if __name__ == "__main__":
    main()
