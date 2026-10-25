"""CLI entry point: train a final model with the most regularized finished trial of an Optuna run."""
import argparse

from loan_approval.pipelines.training_pipeline import train_most_regularized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tuning-run-id", default=None, help="Optuna parent run (default: latest)")
    args = parser.parse_args()
    train_most_regularized(args.config, tuning_run_id=args.tuning_run_id)


if __name__ == "__main__":
    main()
