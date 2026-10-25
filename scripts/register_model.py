"""CLI entry point: register a logged run's model in the MLflow Model Registry and point an alias at it."""
import argparse

import mlflow

from loan_approval.config import load_config
from loan_approval.data.run_data import find_run_id
from loan_approval.pipelines.training_pipeline import setup_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    who = parser.add_mutually_exclusive_group()
    who.add_argument("--run-id")
    who.add_argument("--run-name", default="xgb_full_regularized", help="Latest finished run with this name")
    parser.add_argument("--model-name", default="loan-default-xgboost")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    setup_mlflow(load_config(args.config))
    run_id = args.run_id or find_run_id(args.run_name)
    logged = mlflow.search_logged_models(filter_string=f"source_run_id = '{run_id}'", output_format="list")
    if not logged:
        raise ValueError(f"Run {run_id[:8]} has no logged model")
    version = mlflow.register_model(f"models:/{logged[0].model_id}", args.model_name)
    mlflow.MlflowClient().set_registered_model_alias(args.model_name, args.alias, version.version)
    print(f"{args.model_name} v{version.version} (run {run_id[:8]}) -> models:/{args.model_name}@{args.alias}")


if __name__ == "__main__":
    main()
