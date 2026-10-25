"""Build data_dictionary.csv for the original LendingClub accepted-loans CSV.

Sources:
- data/external/LCDataDictionary.xlsx (https://resources.lendingclub.com/LCDataDictionary.xlsx)
  - LoanStats sheet: description of every column
  - browseNotes sheet: fields shown to investors when a loan was listed, i.e. known before funding
- Header line of the accepted CSV (the 151 columns to describe)
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# browseNotes names that don't map to a CSV column by camelCase -> snake_case alone
BROWSE_ALIASES = {
    "is_inc_v": "verification_status",
    "verified_status_joint": "verification_status_joint",
    "mths_since_most_recent_inq": "mths_since_recent_inq",
    "mths_since_oldest_il_open": "mo_sin_old_il_acct",
    "percent_bc_gt75": "percent_bc_gt_75",
    "delinq2_yrs": "delinq_2yrs",
    "inq_last6_mths": "inq_last_6mths",
    "acc_open_past24_mths": "acc_open_past_24mths",
}
LOANSTATS_ALIASES = {"verified_status_joint": "verification_status_joint"}

IDENTIFIER = {"id", "member_id", "url"}
TARGET_SOURCE = {"loan_status"}
LENDER_DECISION = {
    "grade", "sub_grade", "int_rate", "installment", "initial_list_status",
    "funded_amnt", "funded_amnt_inv", "policy_code", "disbursement_method",
}
ORIGINATION_META = {"issue_d"}
# Credit-bureau history pulled at application but missing from browseNotes
APPLICATION_OVERRIDES = {"mths_since_recent_bc_dlq"}

# Original column -> column in the 15-column filtered dataset
FILTERED_DATASET_MAP = {
    "id": "id", "issue_d": "issue_d", "annual_inc": "revenue", "dti": "dti_n",
    "loan_amnt": "loan_amnt", "fico_range_low": "fico_n", "fico_range_high": "fico_n",
    "emp_length": "emp_length", "purpose": "purpose", "home_ownership": "home_ownership_n",
    "addr_state": "addr_state", "zip_code": "zip_code", "loan_status": "Default",
    "title": "title", "desc": "desc",
}


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name.strip()).lower()


def classify(col: str, shown_at_listing: bool) -> str:
    if col in IDENTIFIER:
        return "identifier"
    if col in TARGET_SOURCE:
        return "target_source"
    if col in LENDER_DECISION:
        return "lender_decision"
    if col in ORIGINATION_META:
        return "origination_meta"
    if shown_at_listing or col in APPLICATION_OVERRIDES:
        return "application"
    return "post_origination_leakage"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xlsx", default=ROOT / "data/external/LCDataDictionary.xlsx", type=Path)
    parser.add_argument("--csv", default=ROOT / cfg["paths"]["raw_data"] / cfg["data"]["original_file"],
                        type=Path)
    parser.add_argument("--out", default=ROOT / cfg["data"]["data_dictionary"], type=Path)
    args = parser.parse_args()

    with open(args.csv) as f:
        columns = f.readline().strip().split(",")

    sheets = pd.read_excel(args.xlsx, sheet_name=["LoanStats", "browseNotes"])
    loanstats = sheets["LoanStats"].dropna(subset=["LoanStatNew"])
    descriptions = {
        LOANSTATS_ALIASES.get(n.strip(), n.strip()): str(d).strip()
        for n, d in zip(loanstats["LoanStatNew"], loanstats["Description"])
    }
    browse = {
        BROWSE_ALIASES.get(snake(n), snake(n))
        for n in sheets["browseNotes"]["BrowseNotesFile"].dropna().astype(str)
    }

    rows = []
    for col in columns:
        shown = col in browse
        group = classify(col, shown)
        rows.append({
            "column": col,
            "description": descriptions.get(col, ""),
            "shown_at_listing": shown,
            "group": group,
            "usable_for_model": group == "application",
            "filtered_dataset_column": FILTERED_DATASET_MAP.get(col, ""),
        })
    df = pd.DataFrame(rows)

    # Review report: rule-based group disagrees with the browseNotes evidence
    unmatched_browse = sorted(browse - set(columns))
    overridden_in = df[~df["shown_at_listing"] & (df["group"] == "application")]["column"].tolist()
    shown_not_app = df[df["shown_at_listing"] & (df["group"] != "application")][["column", "group"]]
    print(f"{len(df)} columns -> {args.out.relative_to(ROOT)}")
    print("Missing descriptions:", df.loc[df["description"] == "", "column"].tolist())
    print("browseNotes names not matched to a CSV column:", unmatched_browse)
    print("Application by override (not in browseNotes):", overridden_in)
    print("In browseNotes but not 'application':", shown_not_app.to_dict("records"))
    print(df["group"].value_counts().to_string())

    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
