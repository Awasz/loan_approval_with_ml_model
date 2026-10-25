"""Request and response schemas. Field names follow the LendingClub columns (see data_dictionary.csv);
a few are taken in a friendlier form and converted back to the raw format by `Application.to_raw`.

A field is optional only if it was sometimes missing in the training data (2013-01 to 2015-06), where
missing has a meaning: the event never happened ("months since last delinquency"), the borrower has no
such account, or employment details were not given. All other fields were always present in training;
the model reads a missing value there as unusual, which inflates the predicted risk (omitting the bureau
fields roughly doubles the average PD), so they are required.
"""
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Opt = float | None  # optional: missing in training, None = no such event / account

EXAMPLE = {
    "loan_amnt": 12000, "term": 36, "annual_inc": 65000, "dti": 18.5,
    "fico_range_low": 690, "fico_range_high": 694, "emp_length": 4, "emp_title": "Teacher",
    "home_ownership": "MORTGAGE", "verification_status": "Source Verified", "purpose": "debt_consolidation",
    "addr_state": "CA", "earliest_cr_line": "2004-08-01", "issue_date": "2015-10-01",
    "delinq_2yrs": 0, "inq_last_6mths": 1, "open_acc": 11, "pub_rec": 0, "revol_bal": 9500,
    "revol_util": 48.2, "total_acc": 24, "collections_12_mths_ex_med": 0, "acc_now_delinq": 0,
    "tot_coll_amt": 0, "tot_cur_bal": 152000, "total_rev_hi_lim": 19700, "acc_open_past_24mths": 4,
    "avg_cur_bal": 13800, "bc_open_to_buy": 6200, "bc_util": 52.0, "chargeoff_within_12_mths": 0,
    "delinq_amnt": 0, "mo_sin_old_il_acct": 120, "mo_sin_old_rev_tl_op": 134, "mo_sin_rcnt_rev_tl_op": 7,
    "mo_sin_rcnt_tl": 5, "mort_acc": 1, "mths_since_recent_bc": 14, "mths_since_recent_inq": 4,
    "num_accts_ever_120_pd": 0, "num_actv_bc_tl": 3, "num_actv_rev_tl": 5, "num_bc_sats": 4,
    "num_bc_tl": 7, "num_il_tl": 6, "num_op_rev_tl": 8, "num_rev_accts": 14, "num_tl_120dpd_2m": 0,
    "num_tl_30dpd": 0, "num_tl_90g_dpd_24m": 0, "num_tl_op_past_12m": 2, "pct_tl_nvr_dlq": 100,
    "percent_bc_gt_75": 25, "pub_rec_bankruptcies": 0, "tax_liens": 0, "total_bal_ex_mort": 31000,
    "total_bc_limit": 13000, "total_il_high_credit_limit": 26000,
}


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    # Loan and borrower
    loan_amnt: float = Field(gt=0, description="Requested loan amount (USD)")
    term: Literal[36, 60] = Field(description="Term in months")
    annual_inc: float = Field(ge=0, description="Self-reported annual income (USD)")
    dti: float = Field(description="Debt-to-income ratio (%), excluding mortgage and this loan")
    fico_range_low: float = Field(ge=300, le=850)
    fico_range_high: float = Field(ge=300, le=850)
    emp_length: int | None = Field(None, ge=0, le=10, description="Years employed; 0 = < 1 year, 10 = 10+")
    emp_title: str | None = Field(None, description="Job title; only whether it is given is used")
    home_ownership: str = Field(description="MORTGAGE, RENT, OWN or ANY")
    verification_status: str = Field(description="Not Verified, Source Verified or Verified")
    purpose: str = Field(description="e.g. debt_consolidation, credit_card, home_improvement")
    addr_state: str = Field(min_length=2, max_length=2, description="US state code")
    earliest_cr_line: date = Field(description="Month the earliest credit line was opened")
    issue_date: date = Field(default_factory=date.today,
                             description="Application / issue date, used for credit-history length")

    # Credit bureau: always present in training, so required
    delinq_2yrs: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    total_acc: float
    collections_12_mths_ex_med: float
    acc_now_delinq: float
    tot_coll_amt: float
    tot_cur_bal: float
    total_rev_hi_lim: float
    acc_open_past_24mths: float
    chargeoff_within_12_mths: float
    delinq_amnt: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_rev_accts: float
    num_tl_30dpd: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    pub_rec_bankruptcies: float
    tax_liens: float
    total_bal_ex_mort: float
    total_bc_limit: float
    total_il_high_credit_limit: float

    # Credit bureau: None = the event never happened / no such account (share missing in training)
    mths_since_last_record: Opt = Field(None, description="None = no public record (83%)")
    mths_since_recent_bc_dlq: Opt = Field(None, description="None = no bankcard delinquency (75%)")
    mths_since_last_major_derog: Opt = Field(None, description="None = no major derogatory (73%)")
    mths_since_recent_revol_delinq: Opt = Field(None, description="None = no revolving delinquency (66%)")
    mths_since_last_delinq: Opt = Field(None, description="None = no delinquency (51%)")
    mths_since_recent_inq: Opt = Field(None, description="None = no inquiry on file (10%)")
    mo_sin_old_il_acct: Opt = Field(None, description="None = no installment account (3.7%)")
    num_tl_120dpd_2m: Opt = Field(None, description="Missing in 2.6% of training loans")
    bc_util: Opt = Field(None, description="None = no bankcard (1.0%)")
    percent_bc_gt_75: Opt = Field(None, description="None = no bankcard (1.0%)")
    bc_open_to_buy: Opt = Field(None, description="None = no bankcard (1.0%)")
    mths_since_recent_bc: Opt = Field(None, description="None = no bankcard (0.9%)")
    revol_util: Opt = Field(None, description="Missing in 0.05% of training loans")
    pct_tl_nvr_dlq: Opt = Field(None, description="Missing in 0.04% of training loans")
    avg_cur_bal: Opt = Field(None, description="Missing in 0.003% of training loans")

    def to_raw(self) -> dict[str, Any]:
        """The application in the raw LendingClub format that the feature pipeline expects."""
        raw = self.model_dump()
        raw["term"] = f" {self.term} months"
        raw["emp_length"] = None if self.emp_length is None else f"{self.emp_length} years"
        raw["emp_title"] = self.emp_title or None  # an empty title counts as missing, as in the raw CSV
        raw["earliest_cr_line"] = self.earliest_cr_line.strftime("%b-%Y")
        return raw


class BatchRequest(BaseModel):
    applications: list[Application] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    probability_of_default: float
    decision: Literal["approve", "decline"]
    threshold: float
    warnings: list[str] = []


class BatchResponse(BaseModel):
    model: str
    predictions: list[Prediction]


class PredictResponse(Prediction):
    model: str
