"""Export the exact train / test / OOT data a logged run used to a parquet file (verified by digest)."""
import argparse
from pathlib import Path

from loan_approval.config import PROJECT_ROOT, load_config
from loan_approval.data.run_data import find_run_id, load_run_dataset
from loan_approval.pipelines.training_pipeline import setup_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--run-id")
    who.add_argument("--run-name", help="Latest finished run with this name, e.g. xgb_full_regularized")
    parser.add_argument("--split", default="train", choices=["train", "test", "oot"])
    parser.add_argument("--out", default=None, help="Default: data/processed/<run name>_<split>.parquet")
    parser.add_argument("--config", default="configs/config.yaml", help="Only used to find the MLflow store")
    args = parser.parse_args()

    setup_mlflow(load_config(args.config))
    run_id = args.run_id or find_run_id(args.run_name)
    ds = load_run_dataset(run_id, args.split)
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "data" / "processed" / f"{args.run_name or run_id[:8]}_{args.split}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.X.assign(id=ds.ids.to_numpy(), **{ds.y.name: ds.y.to_numpy()}).to_parquet(out, index=False)
    print(f"{len(ds.X):,} rows x {ds.X.shape[1]} features -> {out} (digest verified: {ds.digest_verified})")


if __name__ == "__main__":
    main()
