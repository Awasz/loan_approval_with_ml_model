import numpy as np
import pandas as pd

from loan_approval.data.clean import replace_suspicious_values


def test_replaces_sentinel_codes_and_implausible_values():
    df = pd.DataFrame({
        "dti": [-1.0, 999.0, 17.5, 150.0],
        "mo_sin_old_il_acct": [999.0, 822.0, 130.0, np.nan],
        "total_rev_hi_lim": [9_999_999.0, 2_013_133.0, 25_400.0, 0.0],
        "revol_util": [892.3, 366.6, 50.3, 0.0],
        "annual_inc": [0.0, 65_000.0, 1_000.0, 110_000_000.0],
    })

    out = replace_suspicious_values(df)

    assert out["dti"].isna().tolist() == [True, True, False, False]
    assert out["mo_sin_old_il_acct"].isna().tolist() == [True, False, False, True]
    assert out["total_rev_hi_lim"].isna().tolist() == [True, False, False, False]
    assert out["revol_util"].isna().tolist() == [True, False, False, False]
    pd.testing.assert_series_equal(out["annual_inc"], df["annual_inc"])


def test_does_not_modify_input_and_ignores_missing_columns():
    df = pd.DataFrame({"dti": [999.0], "loan_amnt": [1000.0]})

    out = replace_suspicious_values(df)

    assert df["dti"].iloc[0] == 999.0
    assert np.isnan(out["dti"].iloc[0])
    assert out["loan_amnt"].iloc[0] == 1000.0
