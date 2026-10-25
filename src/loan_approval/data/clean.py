"""Cleaning of the LendingClub accepted-loans data.

Placeholder codes and implausible values were found in notebooks/02-eda-original-accepted.ipynb
(counts are for resolved loans, 2007-2018).
"""
import pandas as pd

# Codes used instead of a real measurement ("not calculable" / "no such account")
SENTINEL_CODES: dict[str, list[float]] = {
    "dti": [-1, 999],  # 2 and 38 loans
    "mo_sin_old_il_acct": [999],  # 1 loan; next largest is 822 months
    "total_rev_hi_lim": [9_999_999],  # 3 loans; next largest is ~2.0M
}

# Values above these bounds are not plausible measurements
UPPER_BOUNDS: dict[str, float] = {
    "revol_util": 500,  # 1 loan at 892.3%; next largest is 366.6%
}

# Not cleaned on purpose:
# - annual_inc == 0: only joint applications, where the co-borrower has the income
# - dti > 100: only joint applications (individual DTI against a small own income)


def replace_suspicious_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with placeholder codes and implausible values set to NaN."""
    df = df.copy()
    for col, codes in SENTINEL_CODES.items():
        if col in df:
            df[col] = df[col].mask(df[col].isin(codes))
    for col, bound in UPPER_BOUNDS.items():
        if col in df:
            df[col] = df[col].mask(df[col] > bound)
    return df
