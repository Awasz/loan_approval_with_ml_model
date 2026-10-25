"""Feature importance as a share of the model's total gain."""
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier


def feature_importance(model: XGBClassifier, importance_type: str = "total_gain") -> pd.DataFrame:
    """Importance per feature, normalised so that `importance_pct` sums to 100.

    `total_gain` is the loss reduction summed over every split on the feature, so each feature's
    share of the total is meaningful. Features never used in a split get 0%.
    """
    booster = model.get_booster()
    scores = pd.Series(booster.get_score(importance_type=importance_type), dtype=float)
    imp = scores.reindex(booster.feature_names).fillna(0.0)
    df = (imp.rename("importance").rename_axis("feature").reset_index()
          .sort_values("importance", ascending=False, ignore_index=True))
    df["importance_pct"] = 100 * df["importance"] / df["importance"].sum()
    df["cumulative_pct"] = df["importance_pct"].cumsum()
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def plot_top_features(importance: pd.DataFrame, top_n: int = 20, title: str = "") -> plt.Figure:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top_n + 1.5))
    ax.barh(top["feature"], top["importance_pct"], color="#2a6fdb")
    for y, pct in enumerate(top["importance_pct"]):
        ax.text(pct, y, f" {pct:.1f}%", va="center", fontsize=8)
    covered = importance["importance_pct"].head(top_n).sum()
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title(f"{title}\nTop {top_n} features = {covered:.1f}% of total importance".strip())
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig
