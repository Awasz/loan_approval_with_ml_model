"""CLI entry point: build the model report (importance, SHAP, PDP, OOT calibration) for a logged run."""
import argparse

from loan_approval.pipelines.report_pipeline import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default="xgb_full_regularized", help="Used when --run-id is not given")
    args = parser.parse_args()
    build_report(args.config, run_id=args.run_id, run_name=args.run_name)


if __name__ == "__main__":
    main()
