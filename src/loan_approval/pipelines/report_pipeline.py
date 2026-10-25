"""Model report for a logged training run: gain importance, SHAP, partial dependence, OOT calibration
and how the train / test / OOT samples were built. Written as one HTML file and logged to the run."""
import base64
import html
import io
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd

from loan_approval.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from loan_approval.data.funnel import column_funnel, plot_funnels, row_funnel
from loan_approval.evaluation.explain import (
    calibration_table,
    plot_calibration,
    plot_confusion_matrices,
    plot_pdp_grid,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall,
    shap_explain,
    shap_importance,
)
from loan_approval.evaluation.importance import feature_importance, plot_top_features
from loan_approval.evaluation.metrics import confusion_at, ks_threshold
from loan_approval.features.build_features import resolve_feature_set
from loan_approval.pipelines.training_pipeline import prepare_data, setup_mlflow
from loan_approval.utils.logger import get_logger

log = get_logger(__name__)

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;
     padding:0 16px;color:#1d2330;line-height:1.5}
h1{font-size:1.6rem}h2{margin-top:2.2rem;border-bottom:1px solid #dde2ea;padding-bottom:4px}
table{border-collapse:collapse;font-size:.85rem;margin:8px 0}
th,td{border:1px solid #dde2ea;padding:4px 8px;text-align:right}th{background:#f3f5f9}
td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto}
.note{background:#f6f8fb;border-left:3px solid #2a6fdb;padding:8px 12px;margin:12px 0}
.kpi{display:flex;gap:12px;flex-wrap:wrap}.kpi div{background:#f3f5f9;padding:8px 14px;border-radius:6px}
"""


def _img(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _table(df: pd.DataFrame, pct_cols=(), rate_cols=(), digits: int = 3) -> str:
    fmt = df.copy()
    for c in fmt.columns:
        if c in pct_cols:
            fmt[c] = fmt[c].map(lambda v: f"{v:.2f}%")
        elif c in rate_cols:
            fmt[c] = fmt[c].map(lambda v: f"{100 * v:.2f}%")
        elif pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda v: f"{v:,.{digits}f}")
    return fmt.to_html(index=False, border=0, escape=True)


def funnel_html(cfg: dict) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Row and column funnel (sizes and why each step shrinks) as an HTML fragment, plus the tables."""
    rows, cols = row_funnel(cfg), column_funnel(cfg)
    final = rows[rows["stage"] == "Fully matured vintages only"].iloc[0]
    fragment = f"""
{_img(plot_funnels(rows.iloc[:5], cols))}
<p>The modelling dataset keeps <b>{final['rows']:,} of {rows['rows'].iloc[0]:,} loans
({100 * final['rows'] / rows['rows'].iloc[0]:.1f}%)</b> and turns <b>{cols['columns'].iloc[0]} raw columns into
{cols['columns'].iloc[-1]} model features</b>, plus the target and the issue date used for splitting.</p>
<h3>Rows</h3>
{_table(rows.rename(columns={'detail': 'what was removed'}), rate_cols=('default_rate',))}
<h3>Columns</h3>
{_table(cols.rename(columns={'which': 'columns removed'}))}
"""
    return fragment, rows, cols


def build_funnel_report(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    cfg = load_config(config_path)
    fragment, rows, cols = funnel_html(cfg)
    out_dir = PROJECT_ROOT / cfg["paths"]["reports"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "data_funnel.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Data funnel</title>"
                   f"<style>{CSS}</style></head><body><h1>Data funnel: raw file to modelling dataset</h1>"
                   f"{fragment}</body></html>")
    rows.to_csv(out_dir / "data_funnel_rows.csv", index=False)
    cols.to_csv(out_dir / "data_funnel_columns.csv", index=False)
    log.info("Funnel written to %s", out)
    return out


def find_run(run_id: str | None, run_name: str | None) -> mlflow.entities.Run:
    if run_id:
        return mlflow.get_run(run_id)
    runs = mlflow.search_runs(filter_string=f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], output_format="list")
    if not runs:
        raise ValueError(f"No finished run named {run_name!r}")
    return runs[0]


def build_report(config_path: str | Path = DEFAULT_CONFIG, run_id: str | None = None,
                 run_name: str = "xgb_full_regularized", min_importance_pct: float = 1.0,
                 shap_sample: int = 20_000, pdp_sample: int = 5_000, seed: int = 42) -> Path:
    cfg = load_config(config_path)
    setup_mlflow(cfg)
    run = find_run(run_id, run_name)
    run_id, tags, metrics = run.info.run_id, run.data.tags, run.data.metrics
    feature_set = tags["feature_set"]
    logged = mlflow.search_logged_models(filter_string=f"source_run_id = '{run_id}'", output_format="list")
    model = mlflow.xgboost.load_model(f"models:/{logged[0].model_id}")
    log.info("Report for run %s (%s), feature set %r", run.info.run_name, run_id[:8], feature_set)

    data = prepare_data(cfg, feature_set)
    X_oot, y_oot = data.X["oot"], data.y["oot"]
    numeric_list, categorical_list = resolve_feature_set(cfg, feature_set)
    numeric = set(numeric_list)
    has_test = len(data.X["test"]) > 0
    target = cfg["data"]["target"]

    # 1. Gain importance, features above the threshold
    gain = feature_importance(model)
    gain_top = gain[gain["importance_pct"] > min_importance_pct]

    # 2. SHAP on an OOT sample
    X_shap = X_oot.sample(min(shap_sample, len(X_oot)), random_state=seed)
    shap, base_value = shap_explain(model, X_shap)
    shap_imp = shap_importance(shap)
    beeswarm_features = shap_imp["feature"].head(20).tolist()

    # 2b. Waterfall for one OOT loan: a loan that defaulted, with a predicted PD at the 90th percentile
    # of OOT scores (a typical high-risk loan the model flags), so the example is reproducible
    shap_scores = pd.Series(model.predict_proba(X_shap)[:, 1], index=X_shap.index)
    defaulted = shap_scores[y_oot.loc[X_shap.index] == 1]
    example = (defaulted - shap_scores.quantile(0.9)).abs().idxmin()
    # The loan id stays in the local MLflow run only; the HTML report is shared, so it doesn't name the loan
    example_id = data.splits.oot.loc[example, "id"]
    waterfall = plot_shap_waterfall(shap.loc[example], X_shap.loc[example], base_value,
                                    title="One OOT loan (actual: defaulted)")

    # 3. Partial dependence for numeric features above the threshold
    pdp_features = [f for f in gain_top["feature"] if f in numeric]
    X_pdp = X_oot.sample(min(pdp_sample, len(X_oot)), random_state=seed)
    oot_scores = model.predict_proba(X_oot)[:, 1]
    # Grid values from train: OOT has only 36-month loans, so its own grid would miss term = 60
    pdp_fig, _ = plot_pdp_grid(model, X_pdp, pdp_features, baseline=float(oot_scores.mean()),
                               grid_source=data.X["train"])

    # 4. Binned predicted vs observed on OOT
    calib = calibration_table(y_oot, oot_scores)

    # 5. Confusion matrices at the KS threshold. Scores: train (in-sample), the in-time hold-out (test, or
    # the out-of-fold CV scores when there is no test set) and OOT. The threshold is set on the hold-out
    # when it is out-of-fold, otherwise on train, and applied unchanged to every sample.
    scores = {"train": model.predict_proba(data.X["train"])[:, 1]}
    ys = {"train": data.y["train"].to_numpy()}
    if has_test:
        scores["test"], ys["test"] = model.predict_proba(data.X["test"])[:, 1], data.y["test"].to_numpy()
        threshold_on = "train"
    else:
        oof = pd.read_parquet(mlflow.MlflowClient().download_artifacts(run_id, "oof_predictions.parquet"))
        oof = data.splits.train[["id"]].merge(oof, on="id", how="left")
        scores["oof"], ys["oof"] = oof["oof_score"].to_numpy(), oof[target].to_numpy()
        threshold_on = "oof"
    scores["oot"], ys["oot"] = oot_scores, y_oot.to_numpy()
    threshold, ref_ks = ks_threshold(ys[threshold_on], scores[threshold_on])
    label = {"train": "Train (in-sample)", "test": "Test", "oof": "Out-of-fold", "oot": "OOT"}
    matrices = {label[n]: confusion_at(ys[n], s, threshold) for n, s in scores.items()}
    own_ks = {n: ks_threshold(ys[n], s) for n, s in scores.items()}
    cm_table = pd.DataFrame(matrices).T.rename_axis("sample").reset_index()
    cm_table["own_ks_threshold"] = [own_ks[n][0] for n in scores]
    cm_table["ks"] = [own_ks[n][1] for n in scores]

    funnel_fragment, funnel_rows, funnel_cols = funnel_html(cfg)
    split = data.summary.reset_index(names="sample")
    p, s = cfg["data"]["population"], cfg["split"]
    params = {k.removeprefix("params."): v for k, v in run.data.params.items()}
    kpis = "".join(f"<div><b>{k}</b><br>{metrics[m]:.4f}</div>" for k, m in [
        ("CV AUC", "cv_auc_mean"), ("Out-of-fold AUC", "oof_auc"), ("Train AUC", "train_auc"),
        ("Test AUC", "test_auc"), ("OOT AUC", "oot_auc"), ("OOT Gini", "oot_gini"), ("OOT KS", "oot_ks"),
        ("OOT score PSI vs test", "oot_score_psi_vs_test"), ("OOT score PSI vs OOF", "oot_score_psi_vs_oof"),
    ] if m in metrics)
    if has_test:
        design = f"""<li><b>Train / test:</b> the remaining development loans (issued {split.loc[0, 'issue_from']} to
{split.loc[0, 'issue_to']}) split {int(100 * (1 - s['test_size']))}/{int(100 * s['test_size'])} at random,
stratified on the target. <i>Why:</i> test measures in-time generalisation on the same period; comparing
it with OOT shows whether performance holds over time.</li>
<li><b>Cross-validation:</b> {s['n_splits']}-fold stratified CV on train only; each fold early-stops on its
own validation fold, and the final model's number of trees comes from the folds' best iterations.
Hyperparameter search (Optuna) also used train only.</li>"""
    else:
        design = f"""<li><b>Development, k-fold (no separate test set):</b> all {split.loc[0, 'rows']:,} loans issued
{split.loc[0, 'issue_from']} to {split.loc[0, 'issue_to']} are split into {s['n_splits']} stratified folds.
Each fold's model trains on the other {s['n_splits'] - 1} folds and early-stops on its own fold; every loan is
scored once by the model that never saw it (<b>out-of-fold</b> scores). <i>Why:</i> it uses all development
data for both training and in-time evaluation instead of locking 20% away in a test set; the out-of-fold
scores play the role of the test set. The final model is then fitted on all development loans, with the
number of trees taken from the folds' best iterations. Optuna tuning used {cfg['tuning']['n_splits']}-fold CV
on the same development loans.</li>"""
    excluded = cfg["features"]["sets"][feature_set].get("exclude", [])
    gain_note = ("Gain favours high-cardinality categoricals such as <code>addr_state</code>: the trees can split "
                 "its 49 levels many ways and pick up noise. SHAP below measures each feature's actual effect on "
                 "predictions and is the more reliable ranking." if "addr_state" in categorical_list else
                 f"Excluded from this model: {', '.join(f'<code>{c}</code>' for c in excluded) or 'none'}. Gain can "
                 "still favour categoricals with many levels (purpose); SHAP below is the more reliable ranking.")
    chosen = {k: params[k] for k in ["max_depth", "min_child_weight", "subsample", "colsample_bytree",
                                     "reg_lambda", "reg_alpha", "gamma", "max_cat_threshold",
                                     "learning_rate", "final_n_estimators"] if k in params}

    body = f"""
<h1>Loan default model report: {html.escape(run.info.run_name)}</h1>
<p>MLflow run <code>{run_id}</code>; feature set <b>{feature_set}</b> ({X_oot.shape[1]} features);
selected trial: <b>{html.escape(tags.get('selected_trial', 'n/a'))}</b>
({html.escape(tags.get('selection', 'config params'))}).</p>
<div class="kpi">{kpis}</div>
<p>Hyperparameters: {html.escape(', '.join(f'{k}={v}' for k, v in chosen.items()))}</p>

<h2>1. Data funnel and how the samples were chosen</h2>
{funnel_fragment}
<h3>Samples</h3>
{_table(split, rate_cols=('default_rate', 'share_60_month'))}
<ul>
<li><b>Population:</b> loans with a final outcome (Fully Paid = 0, Charged Off = 1) from vintages that had
fully run off by the data snapshot (2018 Q4): 36-month loans issued {p['start']} to
{p['last_issue_month_by_term'][36]}, 60-month loans issued up to {p['last_issue_month_by_term'][60]}.
<i>Why:</i> in later vintages most loans were still running, and the ones already finished are biased
towards early defaults and early pay-offs, so their default rates are not representative.</li>
<li><b>OOT (out-of-time):</b> every loan issued from {s['oot_start']}, i.e. the most recent fully matured
months. <i>Why:</i> it mimics deployment, where the model scores loans issued after its training
period. It is never used for training, tuning or early stopping, so it is an honest estimate of
performance on future business. It has no 60-month loans (none issued after 2013 had matured).</li>
{design}
</ul>

<h2>2. Feature importance by gain (features above {min_importance_pct:g}%)</h2>
<p>Share of the model's total gain (loss reduction summed over all splits on the feature);
all {len(gain)} features sum to 100%. {len(gain_top)} features exceed {min_importance_pct:g}% and together
account for {gain_top['importance_pct'].sum():.1f}%.</p>
{_img(plot_top_features(gain_top.reset_index(drop=True), top_n=len(gain_top), title='Gain importance'))}
{_table(gain_top[['rank', 'feature', 'importance_pct', 'cumulative_pct']], pct_cols=('importance_pct', 'cumulative_pct'))}
<div class="note">{gain_note}</div>

<h2>3. SHAP importance (OOT sample of {len(X_shap):,} loans)</h2>
<p>Mean absolute SHAP value: how far, on average, a feature moves an individual prediction (in
log-odds). Computed with XGBoost's exact TreeSHAP.</p>
{_img(plot_shap_bar(shap_imp))}
<h3>SHAP summary (feature × effect × feature value)</h3>
<p>Each dot is one loan. Horizontal position is the feature's effect on that loan's predicted risk
(right = riskier); colour is the feature's value (blue low, red high). For example, red dots on the right
for <code>dti</code> mean high DTI raises predicted default risk.</p>
{_img(plot_shap_beeswarm(shap, X_shap, beeswarm_features))}
{_table(shap_imp.head(20), pct_cols=('shap_pct',), digits=4)}
<h3>One OOT loan explained (SHAP waterfall)</h3>
<p>This loan defaulted, and its predicted PD
(<b>{100 * shap_scores[example]:.1f}%</b>) sits at the 90th percentile of OOT scores, a typical high-risk
loan the model flags. The chart starts from the model's average (base value) and adds each feature's
contribution in log-odds: red bars push towards default, blue bars away. The 12 largest contributions are
shown; the rest are grouped.</p>
{_img(waterfall)}

<h2>4. Partial dependence: numeric features above {min_importance_pct:g}% gain</h2>
<p>Average predicted default probability over {len(X_pdp):,} OOT loans when the feature is set to each
value (2nd to 98th percentile of the training data), all other features left as they are. It shows the
shape of the model's learned relationship; flat curves mean little effect. Curves at the edges rest on
fewer loans, so wiggles there are less reliable.</p>
{_img(pdp_fig)}

<h2>5. Predicted vs observed default rate on OOT, by score decile</h2>
<p>All {len(y_oot):,} OOT loans, sorted into 10 equal groups by predicted risk. The bars are the default rate
that actually happened; the line is the model's average predicted probability. Close agreement means the
predictions are calibrated (usable as probabilities); a steep rise across deciles means good ranking.</p>
{_img(plot_calibration(calib, 'OOT: observed vs predicted default rate by decile'))}
{_table(calib, rate_cols=('predicted_rate', 'observed_rate', 'observed_minus_predicted'), digits=4)}
<p>Lowest-risk decile: {100 * calib['observed_rate'].iloc[0]:.1f}% observed default rate; highest-risk decile:
{100 * calib['observed_rate'].iloc[-1]:.1f}% ({calib['observed_rate'].iloc[-1] / calib['observed_rate'].iloc[0]:.1f}×).
Largest gap between observed and predicted in any decile:
{100 * calib['observed_minus_predicted'].abs().max():.2f} percentage points.</p>

<h2>6. Confusion matrices at the KS threshold</h2>
<p>The KS threshold is the score where the defaulters' and non-defaulters' score distributions are furthest
apart (maximum of recall minus false-positive rate). It is set on <b>{label[threshold_on].lower()}</b> scores
(<b>{threshold:.4f}</b>, KS {ref_ks:.3f}) and applied unchanged to every sample, as it would be in production.
Train (in-sample) is shown for reference only: the model has seen those loans, so it looks better there.
Loans scoring at or above it are predicted to default (rejected); the rest are approved.</p>
{_img(plot_confusion_matrices(matrices, threshold, set_on=label[threshold_on].lower()))}
{_table(cm_table[['sample', 'tn', 'fp', 'fn', 'tp', 'recall', 'false_positive_rate', 'precision', 'accuracy',
                  'approval_rate', 'default_rate_approved', 'default_rate_rejected', 'own_ks_threshold', 'ks']],
        rate_cols=('recall', 'false_positive_rate', 'precision', 'accuracy', 'approval_rate',
                   'default_rate_approved', 'default_rate_rejected'), digits=4)}
<p><i>own_ks_threshold</i> and <i>ks</i> are each sample's own optimum, for reference only; the matrices all
use the same threshold. The KS threshold maximises separation, not profit: a lender would set the cut-off
from the cost of a default versus the margin lost on a rejected good loan.</p>
"""
    out_dir = PROJECT_ROOT / cfg["paths"]["reports"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"model_report_{run.info.run_name}.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Model report: "
                   f"{html.escape(run.info.run_name)}</title><style>{CSS}</style></head><body>{body}</body></html>")

    client = mlflow.MlflowClient()
    client.log_artifact(run_id, str(out), "report")
    client.log_text(run_id, gain_top.to_csv(index=False), "report/gain_importance_above_1pct.csv")
    client.log_text(run_id, shap_imp.to_csv(index=False), "report/shap_importance.csv")
    client.log_text(run_id, calib.to_csv(index=False), "report/oot_calibration_deciles.csv")
    client.log_text(run_id, cm_table.to_csv(index=False), "report/confusion_matrices_ks_threshold.csv")
    client.log_metric(run_id, f"ks_threshold_{threshold_on}", threshold)
    for name in scores:
        for k in ["recall", "false_positive_rate", "precision", "approval_rate", "default_rate_approved"]:
            client.log_metric(run_id, f"{name}_{k}_at_ks", matrices[label[name]][k])
    client.log_text(run_id, f"loan id {example_id}\n" + pd.DataFrame(
        {"value": X_shap.loc[example].astype(str), "shap": shap.loc[example]}).sort_values(
        "shap", key=abs, ascending=False).to_csv(), "report/shap_waterfall_example.csv")
    client.log_text(run_id, funnel_rows.to_csv(index=False), "report/data_funnel_rows.csv")
    client.log_text(run_id, funnel_cols.to_csv(index=False), "report/data_funnel_columns.csv")
    log.info("Report written to %s and logged to run %s", out, run_id[:8])
    return out
