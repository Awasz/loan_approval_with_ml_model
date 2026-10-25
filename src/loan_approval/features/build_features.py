"""Feature engineering for the loan default model."""
import pandas as pd

# Raw columns needed to derive the engineered features
DERIVED_FROM = {
    "term_months": ["term"],
    "fico": ["fico_range_low", "fico_range_high"],
    "emp_length_years": ["emp_length"],
    "emp_title_missing": ["emp_title"],
    "credit_history_months": ["earliest_cr_line"],
}


def raw_columns_for(features: list[str]) -> list[str]:
    """Raw source columns needed to build `features`."""
    cols: list[str] = []
    for f in features:
        cols.extend(DERIVED_FROM.get(f, [f]))
    return list(dict.fromkeys(cols))


def build_features(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Row-wise transformations only (no fitted state), so it is safe to apply before splitting."""
    out = pd.DataFrame(index=df.index)
    derived = {
        "term_months": lambda: df["term"].str.extract(r"(\d+)", expand=False).astype(float),
        "fico": lambda: (df["fico_range_low"] + df["fico_range_high"]) / 2,
        "emp_length_years": lambda: (
            df["emp_length"].str.replace("< 1", "0", regex=False)
            .str.extract(r"(\d+)", expand=False).astype(float)
        ),
        "emp_title_missing": lambda: df["emp_title"].isna().astype(float),
        "credit_history_months": lambda: (
            (df["issue_date"] - pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")).dt.days / 30.44
        ),
    }
    for f in numeric:
        out[f] = derived[f]() if f in derived else df[f].astype(float)
    for f in categorical:
        out[f] = df[f].astype("category")
    return out


def align_categories(train: pd.DataFrame, *others: pd.DataFrame) -> list[pd.DataFrame]:
    """Restrict every categorical column to the levels observed in `train`.

    Returns [train, *others]; levels unseen in train become NaN in the other frames.
    """
    cats = train.select_dtypes("category").columns
    levels = {c: train[c].cat.remove_unused_categories().cat.categories for c in cats}
    aligned = []
    for frame in (train, *others):
        frame = frame.copy()
        for c in cats:
            values = frame[c].astype(object)
            frame[c] = pd.Categorical(values.where(values.isin(levels[c])), categories=levels[c])
        aligned.append(frame)
    return aligned


def resolve_feature_set(cfg: dict, name: str) -> tuple[list[str], list[str]]:
    """(numeric, categorical) features of a configured set. A set can reuse another one with
    `base: <set>` and drop features with `exclude: [...]`."""
    spec = cfg["features"]["sets"][name]
    if "base" in spec:
        numeric, categorical = resolve_feature_set(cfg, spec["base"])
    else:
        numeric, categorical = list(spec["numeric"]), list(spec["categorical"])
    drop = set(spec.get("exclude", []))
    return [f for f in numeric if f not in drop], [f for f in categorical if f not in drop]
