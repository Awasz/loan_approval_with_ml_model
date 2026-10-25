"""Model explanation: SHAP values, partial dependence and binned calibration."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier


def shap_explain(model: XGBClassifier, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact TreeSHAP contributions (log-odds units) from XGBoost, one column per feature, and the
    base value (the model's average log-odds). Per row: base + sum(contributions) = predicted log-odds."""
    contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index), float(contribs[0, -1])


def shap_values(model: XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    return shap_explain(model, X)[0]


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def _fmt_value(v) -> str:
    if pd.isna(v):
        return "missing"
    if isinstance(v, (int, float, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def plot_shap_waterfall(contribs: pd.Series, values: pd.Series, base_value: float, top_n: int = 12,
                        title: str = "") -> plt.Figure:
    """How one prediction is built: start at the base value, each feature pushes the log-odds up (red,
    towards default) or down (blue); the smallest contributions are grouped as 'other features'."""
    order = contribs.abs().sort_values(ascending=False).index
    top, rest = order[:top_n], order[top_n:]
    steps = [(f"{f} = {_fmt_value(values[f])}", float(contribs[f])) for f in top]
    if len(rest):
        steps.append((f"{len(rest)} other features", float(contribs[rest].sum())))
    final = base_value + float(contribs.sum())

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(steps) + 2))
    running = base_value
    for i, (label, c) in enumerate(steps):
        y = len(steps) - i
        ax.barh(y, c, left=running, color="#d1495b" if c > 0 else "#2a6fdb", height=0.6)
        ax.text(running + c + (0.005 if c >= 0 else -0.005), y, f"{c:+.3f}", va="center",
                ha="left" if c >= 0 else "right", fontsize=8)
        running += c
    ax.set_yticks(range(len(steps), 0, -1), [s[0] for s in steps], fontsize=9)
    ax.axvline(base_value, color="0.5", ls="--", lw=0.8)
    ax.axvline(final, color="0.2", lw=1)
    ax.set_xlabel("log-odds of default")
    ax.set_title(f"{title}\nbase {base_value:.3f} (PD {100 * _sigmoid(base_value):.1f}%) -> "
                 f"prediction {final:.3f} (PD {100 * _sigmoid(final):.1f}%)".strip(), fontsize=10)
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig


def shap_importance(shap: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature, also as a % of the total (sums to 100)."""
    imp = shap.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()
    imp["shap_pct"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp.rename_axis("feature").reset_index()


def plot_shap_bar(importance: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top_n + 1.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2a6fdb")
    for y, (v, pct) in enumerate(zip(top["mean_abs_shap"], top["shap_pct"])):
        ax.text(v, y, f" {pct:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title(f"Mean |SHAP| importance, top {top_n}")
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(shap: pd.DataFrame, X: pd.DataFrame, features: list[str],
                       max_points: int = 4000, seed: int = 0) -> plt.Figure:
    """One row per feature: x = SHAP value, colour = the feature's value (low blue -> high red).
    Categorical features are drawn in grey."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(shap), size=min(max_points, len(shap)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(features) + 1.5))
    cmap = plt.get_cmap("coolwarm")
    for row, f in enumerate(features[::-1]):
        s = shap[f].to_numpy()[idx]
        y = row + rng.uniform(-0.3, 0.3, size=len(s))
        if isinstance(X[f].dtype, pd.CategoricalDtype):
            ax.scatter(s, y, s=3, color="0.55", alpha=0.5, linewidths=0)
        else:
            v = X[f].to_numpy()[idx].astype(float)
            ok = ~np.isnan(v)
            colour = pd.Series(v[ok]).rank(pct=True).to_numpy()
            ax.scatter(s[~ok], y[~ok], s=3, color="0.8", alpha=0.5, linewidths=0)  # missing values
            ax.scatter(s[ok], y[ok], s=3, c=colour, cmap=cmap, vmin=0, vmax=1, alpha=0.6, linewidths=0)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(range(len(features)), features[::-1])
    ax.set_xlabel("SHAP value (log-odds; > 0 pushes towards default)")
    ax.set_title("SHAP summary: colour = feature value (blue low, red high, grey categorical/missing)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label="feature value percentile", fraction=0.03, pad=0.01)
    fig.tight_layout()
    return fig


def partial_dependence(model: XGBClassifier, X: pd.DataFrame, feature: str, n_points: int = 20,
                       grid_source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Average predicted default probability over the rows of `X` when `feature` is set to each grid value.
    Grid: quantiles 2%-98% of the feature in `grid_source` (default `X`), avoiding sparse tails."""
    source = (X if grid_source is None else grid_source)[feature].astype(float)
    grid = np.unique(np.nanquantile(source, np.linspace(0.02, 0.98, n_points)))
    rows = []
    for value in grid:
        Xg = X.copy()
        Xg[feature] = value
        rows.append((value, float(model.predict_proba(Xg)[:, 1].mean())))
    return pd.DataFrame(rows, columns=[feature, "mean_predicted_pd"])


def plot_pdp_grid(model: XGBClassifier, X: pd.DataFrame, features: list[str], baseline: float,
                  grid_source: pd.DataFrame | None = None,
                  ncols: int = 4) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    curves = {}
    for ax, f in zip(axes.flat, features):
        pdp = curves[f] = partial_dependence(model, X, f, grid_source=grid_source)
        ax.plot(pdp[f], 100 * pdp["mean_predicted_pd"], marker="o", ms=3, color="#2a6fdb")
        ax.axhline(100 * baseline, color="0.5", ls="--", lw=0.8)
        ax.set_title(f, fontsize=10)
        ax.set_ylabel("predicted default %", fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Partial dependence (dashed: average predicted default rate)", y=1.0)
    fig.tight_layout()
    return fig, curves


def calibration_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by score decile (decile 1 = lowest predicted risk)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=range(1, bins + 1))
    t = df.groupby("decile", observed=True).agg(
        loans=("y", "size"), score_min=("p", "min"), score_max=("p", "max"),
        predicted_rate=("p", "mean"), observed_rate=("y", "mean"), defaults=("y", "sum"))
    t["observed_minus_predicted"] = t["observed_rate"] - t["predicted_rate"]
    return t.reset_index()


def plot_calibration(table: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = table["decile"].astype(int)
    ax.bar(x, 100 * table["observed_rate"], color="#9cc3f5", label="observed default rate")
    ax.plot(x, 100 * table["predicted_rate"], "o-", color="#d1495b", label="mean predicted PD")
    ax.set_xticks(x)
    ax.set_xlabel("Score decile (1 = lowest predicted risk)")
    ax.set_ylabel("default rate %")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrices(matrices: dict[str, dict[str, float]], threshold: float,
                            set_on: str = "train") -> plt.Figure:
    """One 2x2 matrix per sample: rows = actual, columns = predicted (score >= threshold = default)."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.6 * len(matrices), 4.2))
    for ax, (name, cm) in zip(np.atleast_1d(axes), matrices.items()):
        grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        share = grid / grid.sum()
        ax.imshow(share, cmap="Blues", vmin=0, vmax=share.max() * 1.15)
        labels = [["TN: good, approved", "FP: good, rejected"], ["FN: default, approved", "TP: default, rejected"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{grid[i, j]:,}\n({100 * share[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > share.max() * 0.6 else "black")
        ax.set_xticks([0, 1], ["predicted good\n(approve)", "predicted default\n(reject)"])
        ax.set_yticks([0, 1], ["actual good", "actual default"])
        ax.set_title(f"{name}: recall {100 * cm['recall']:.1f}%, FPR {100 * cm['false_positive_rate']:.1f}%",
                     fontsize=10)
    fig.suptitle(f"Confusion matrices at KS threshold {threshold:.4f} (set on {set_on})", y=1.02)
    fig.tight_layout()
    return fig
