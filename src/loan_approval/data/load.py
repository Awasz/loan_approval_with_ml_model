"""Load the modelling population from the original LendingClub accepted-loans CSV."""
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from loan_approval.config import PROJECT_ROOT

# Raw columns that are text or dates; every other requested column is cast to float
TEXT_COLUMNS = {
    "term", "emp_title", "emp_length", "home_ownership", "verification_status", "purpose",
    "addr_state", "zip_code", "title", "desc", "earliest_cr_line", "application_type",
}


def _month(col: str) -> pl.Expr:
    return pl.concat_str(pl.lit("01-"), pl.col(col)).str.to_date("%d-%b-%Y")


def _as_date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def source_path(cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / cfg["paths"]["raw_data"] / cfg["data"]["original_file"]


def load_population(cfg: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    """Resolved loans of fully matured vintages, with `id`, `issue_date`, the target and `columns`.

    Target: 1 = Charged Off, 0 = Fully Paid (incl. the "Does not meet the credit policy" variants).
    Loans still running at the snapshot, and the 40 loans with status "Default", are excluded.
    """
    data_cfg = cfg["data"]
    target = data_cfg["target"]
    pop = data_cfg["population"]
    path = source_path(cfg)

    status = pl.col("loan_status")
    term = pl.col("term").str.extract(r"(\d+)").cast(pl.Int32)
    matured = pl.lit(False)
    for months, last_issue in pop["last_issue_month_by_term"].items():
        matured = matured | ((term == int(months)) & (pl.col("issue_date") <= _as_date(last_issue)))

    raw_columns = list(dict.fromkeys(["id", "issue_d", "loan_status", "term", *columns]))
    numeric = [c for c in raw_columns if c not in TEXT_COLUMNS | {"id", "issue_d", "loan_status"}]
    df = (
        pl.scan_csv(path, infer_schema=False)
        .filter(pl.col("loan_amnt").is_not_null())  # summary/footer rows
        .select(raw_columns)
        .with_columns(
            pl.when(status.str.ends_with("Charged Off")).then(1)
            .when(status.str.ends_with("Fully Paid")).then(0)
            .otherwise(None).alias(target),
            issue_date=_month("issue_d"),
        )
        .filter(pl.col(target).is_not_null())
        .filter(pl.col("issue_date") >= _as_date(pop["start"]))
        .filter(matured)
        .with_columns([pl.col(c).cast(pl.Float64) for c in numeric])
        .drop("issue_d", "loan_status")
        .collect(engine="streaming")
        .to_pandas()
    )
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    return df
