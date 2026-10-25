"""CLI entry point: write the data funnel (rows and columns at each step, and why) to reports/."""
import argparse

from loan_approval.pipelines.report_pipeline import build_funnel_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    build_funnel_report(parser.parse_args().config)


if __name__ == "__main__":
    main()
