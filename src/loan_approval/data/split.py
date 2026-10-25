"""Development / out-of-time split and in-time train/test split."""
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Splits:
    train: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame


def split_data(df: pd.DataFrame, target: str, oot_start: str, test_size: float,
               random_state: int) -> Splits:
    """Loans issued from `oot_start` onwards form the OOT sample; the rest is split
    into stratified train/test (in-time) sets. With `test_size=0` all development loans are
    train (k-fold CV supplies the in-time evaluation) and `test` is empty."""
    is_oot = df["issue_date"] >= pd.Timestamp(oot_start)
    dev, oot = df[~is_oot], df[is_oot]
    if test_size == 0:
        return Splits(train=dev, test=dev.iloc[:0], oot=oot)
    train, test = train_test_split(dev, test_size=test_size, stratify=dev[target],
                                   random_state=random_state)
    return Splits(train=train, test=test, oot=oot)


def summarize(splits: Splits, target: str) -> pd.DataFrame:
    rows = {}
    for name, part in vars(splits).items():
        if part.empty:
            continue
        rows[name] = {
            "rows": len(part),
            "default_rate": part[target].mean(),
            "issue_from": part["issue_date"].min().date().isoformat(),
            "issue_to": part["issue_date"].max().date().isoformat(),
            "share_60_month": (part["term"].str.strip() == "60 months").mean(),
        }
    return pd.DataFrame(rows).T
