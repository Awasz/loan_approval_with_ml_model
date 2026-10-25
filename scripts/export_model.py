"""CLI entry point: export a registered model to a self-contained bundle (model.ubj + model_card.json)
that the API serves without MLflow or the training data."""
import argparse

from loan_approval.config import PROJECT_ROOT, load_config
from loan_approval.models.bundle import export_bundle
from loan_approval.pipelines.training_pipeline import setup_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Only used to find the MLflow store")
    parser.add_argument("--model-uri", default="models:/loan-default-xgboost@champion")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "models" / "champion"))
    args = parser.parse_args()

    setup_mlflow(load_config(args.config))
    out = export_bundle(args.model_uri, args.out)
    print(f"{args.model_uri} -> {out}")


if __name__ == "__main__":
    main()
