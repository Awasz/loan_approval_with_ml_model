"""Data funnel: how the raw LendingClub file shrinks to the modelling dataset, in rows and columns."""
from datetime import date
from typing import Any

import pandas as pd
import polars as pl

from loan_approval.config import PROJECT_ROOT
from loan_approval.data.load import source_path
from loan_approval.features.build_features import DERIVED_FROM, raw_columns_for, resolve_feature_set

# Application-time columns left out of the model, by reason
TEXT_COLUMNS = {"desc", "title", "zip_code"}
NEAR_DUPLICATES = {"num_sats", "tot_hi_cred_lim", "num_rev_tl_bal_gt_0"}  # |r| > 0.95 with a kept column


def _is_joint(col: str) -> bool:
    return col.startswith("sec_app_") or "joint" in col or col == "application_type"


def row_funnel(cfg: dict[str, Any]) -> pd.DataFrame:
    target = cfg["data"]["target"]
    pop, split = cfg["data"]["population"], cfg["split"]
    status = pl.col("loan_status")
    df = (
        pl.scan_csv(source_path(cfg), infer_schema=False)
        .select("loan_amnt", "loan_status", "issue_d", "term")
        .with_columns(
            pl.when(status.str.ends_with("Charged Off")).then(1)
            .when(status.str.ends_with("Fully Paid")).then(0).alias(target),
            issue_date=pl.concat_str(pl.lit("01-"), pl.col("issue_d")).str.to_date("%d-%b-%Y", strict=False),
            term_m=pl.col("term").str.extract(r"(\d+)").cast(pl.Int32),
        )
        .collect(engine="streaming")
    )
    matured = pl.lit(False)
    for months, last in pop["last_issue_month_by_term"].items():
        matured = matured | ((pl.col("term_m") == int(months)) &
                             (pl.col("issue_date") <= date.fromisoformat(str(last))))
    start = date.fromisoformat(str(pop["start"]))

    raw = df
    s1 = raw.filter(pl.col("loan_amnt").is_not_null())
    s2 = s1.filter(pl.col(target).is_not_null())
    s3 = s2.filter(pl.col("issue_date") >= start)
    s4 = s3.filter(matured)
    unresolved = s1.filter(pl.col(target).is_null())["loan_status"].value_counts().sort("count", descending=True)
    immature = s3.filter(~matured).group_by("term_m").agg(pl.len(), pl.col("issue_date").min()).sort("term_m")
    stages = [  # name, frame, why, breakdown of what was removed; default rate shown once labels exist
        ("Raw accepted-loans file", raw, False,
         "All loans LendingClub funded 2007-2018 (Kaggle accepted_2007_to_2018Q4.csv).", ""),
        ("Remove footer rows", s1, False,
         "Summary lines such as 'Total amount funded in policy code 1', not loans.", ""),
        ("Keep resolved loans", s2, True,
         ("Only Fully Paid (target 0) and Charged Off (target 1) have a known outcome. Loans still running "
         "or late at the 2018 Q4 snapshot have no final label; status 'Default' is not yet charged off."),
         "; ".join(f"{s}: {n:,}" for s, n in unresolved.rows())),
        (f"Issued from {start:%Y-%m}", s3, True,
         ("Most credit-bureau features (num_*, mo_sin_*, bc_util, mort_acc ...) are only filled in from 2013; "
         "older loans would be mostly missing values."), ""),
        ("Fully matured vintages only", s4, True,
         ("Loans issued later had mostly not finished by the snapshot; the ones that had are biased towards "
         "early defaults and early pay-offs, so their default rate is not representative."),
         "; ".join(f"{t}-month issued from {d:%Y-%m}: {n:,}" for t, n, d in immature.rows())),
    ]
    rows, prev = [], None
    for name, frame, labelled, why, detail in stages:
        rows.append({"stage": name, "rows": frame.height,
                     "removed": None if prev is None else prev - frame.height,
                     "default_rate": frame[target].mean() if labelled else None,
                     "why": why, "detail": detail})
        prev = frame.height

    is_oot = pl.col("issue_date") >= date.fromisoformat(str(split["oot_start"]))
    dev, oot = s4.filter(~is_oot), s4.filter(is_oot)
    if split["test_size"] == 0:
        rows.append({"stage": "Development (k-fold)", "rows": dev.height, "default_rate": dev[target].mean(),
                     "detail": "",
                     "why": f"All loans issued before {split['oot_start']}: {split['n_splits']}-fold stratified CV "
                            "for tuning and in-time evaluation (out-of-fold scores), then the final fit."})
    else:
        n_test = round(dev.height * split["test_size"])
        train_pct, test_pct = int(100 * (1 - split["test_size"])), int(100 * split["test_size"])
        rows += [
            {"stage": "Train", "rows": dev.height - n_test, "default_rate": dev[target].mean(), "detail": "",
             "why": f"{train_pct}% of development loans (issued before {split['oot_start']}), stratified random; "
                    "used for cross-validation, tuning and fitting."},
            {"stage": "Test", "rows": n_test, "default_rate": dev[target].mean(), "detail": "",
             "why": f"{test_pct}% of development loans: same period, never trained on."},
        ]
    rows.append({"stage": "OOT", "rows": oot.height, "default_rate": oot[target].mean(), "detail": "",
                 "why": f"Issued from {split['oot_start']}: the most recent matured months, held out to test on "
                        "future loans."})
    return pd.DataFrame(rows)


def column_funnel(cfg: dict[str, Any], feature_set: str = "full") -> pd.DataFrame:
    dd = pd.read_csv(PROJECT_ROOT / cfg["data"]["data_dictionary"])
    numeric, categorical = resolve_feature_set(cfg, feature_set)
    features = numeric + categorical
    used_raw = set(raw_columns_for(features))
    excluded = sorted(set(raw_columns_for(cfg["features"]["sets"][feature_set].get("exclude", []))))
    app = dd.loc[dd["group"] == "application", "column"]
    unused = sorted(set(app) - used_raw - set(excluded))
    joint = [c for c in unused if _is_joint(c)]
    text = [c for c in unused if c in TEXT_COLUMNS]
    dups = [c for c in unused if c in NEAR_DUPLICATES]
    late = [c for c in unused if c not in set(joint) | set(text) | set(dups)]
    by_group = dd.groupby("group")["column"].apply(list)
    meta = by_group.get("origination_meta", []) + by_group.get("target_source", [])
    merged = [f for f, src in DERIVED_FROM.items() if f in features and len(src) > 1]
    merged_text = [f"{' + '.join(DERIVED_FROM[f])} -> {f}" for f in merged]

    steps = [
        ("Raw columns", None, [], "All columns in the source file."),
        ("Remove post-origination leakage", -len(by_group["post_origination_leakage"]),
         by_group["post_origination_leakage"],
         ("Recorded after the loan was issued (payments, recoveries, last FICO, hardship, settlement): they "
         "reveal the outcome and do not exist when an application is scored.")),
        ("Remove lender decisions", -len(by_group["lender_decision"]), by_group["lender_decision"],
         ("LendingClub's own pricing and funding (grade, int_rate, installment, funded amounts): its risk "
         "verdict, not borrower information.")),
        ("Remove identifiers", -len(by_group["identifier"]), by_group["identifier"],
         "Row identifiers and URLs carry no predictive information (member_id is always empty)."),
        ("Set aside target source and issue date", -len(meta), meta,
         "loan_status becomes the target; issue_d is used to build the population and the OOT split."),
        ("Remove free text / high-cardinality", -len(text), text,
         "Free text (desc 90% empty, title duplicates purpose) and zip_code (913 levels, overlaps addr_state)."),
        ("Remove joint-application fields", -len(joint), joint,
         "Joint applications started late 2015; these are 100% empty in the matured population (239 joint loans)."),
        ("Remove fields only collected from 2016", -len(late), late,
         "Bureau fields LendingClub added in late 2015: ~97% empty in the modelling population."),
        ("Remove near-duplicates", -len(dups), dups,
         ("Correlation above 0.95 with a kept column (num_sats ~ open_acc, tot_hi_cred_lim ~ tot_cur_bal, "
         "num_rev_tl_bal_gt_0 ~ num_actv_rev_tl).")),
        ("Combine / transform", -sum(len(DERIVED_FROM[f]) - 1 for f in merged), merged_text,
         ("The two FICO bounds become one feature (their midpoint). term, emp_length, emp_title and "
         "earliest_cr_line are converted one-to-one (months, years, missing flag, credit-history months).")),
    ]
    if excluded:
        steps.append((f"Excluded in feature set '{feature_set}'", -len(excluded), excluded,
                      ("Left out by design for this experiment (e.g. addr_state: weak signal that gain "
                      "importance overstates, and a fairness / regulatory concern).")))
    rows, n = [], len(dd)
    for name, change, cols, why in steps:
        n = n + (change or 0)
        rows.append({"stage": name, "columns": n, "removed": None if change is None else -change,
                     "why": why, "which": ", ".join(cols)})
    out = pd.DataFrame(rows)
    assert out["columns"].iloc[-1] == len(features), (out["columns"].iloc[-1], len(features))
    return out


def plot_funnels(rows: pd.DataFrame, cols: pd.DataFrame):
    """Side-by-side funnel charts: loans (left) and columns (right) remaining after each step."""
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1]})
    for ax, df, value, colour, unit in [(ax_r, rows, "rows", "#2a6fdb", "loans"),
                                        (ax_c, cols, "columns", "#e07a2f", "columns")]:
        df = df.iloc[::-1].reset_index(drop=True)
        widths = df[value].to_numpy(dtype=float)
        ax.barh(range(len(df)), widths, left=(widths.max() - widths) / 2, color=colour, alpha=0.85)
        for y, (stage, v) in enumerate(zip(df["stage"], widths)):
            ax.text(widths.max() / 2, y, f"{int(v):,}", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
        ax.set_yticks(range(len(df)), df["stage"], fontsize=9)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{unit.capitalize()} remaining after each step", fontsize=11)
    fig.tight_layout()
    return fig
