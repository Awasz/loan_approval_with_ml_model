# Loan Approval: LendingClub default model

An XGBoost model that predicts, at application time, the probability that a LendingClub loan will be
charged off. It is trained on 580k fully matured loans (2013–2015) with 61 application-time
features, and every run is tracked in MLflow.

**Selected model:** `xgb_full_regularized`, registered as `models:/loan-default-xgboost@champion`.

| Out-of-time (Jul–Dec 2015) | AUC | Gini | KS | Default rate of approved loans at the KS cut-off |
|---|---|---|---|---|
| | 0.703 | 0.405 | 0.293 | 9.2% (vs 14.7% overall), approving 65% |

How the model was built, why it was chosen and its limitations are in
**[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Data

The data files are not in the repository. Download them into `data/`:

| File | Put it in | Link | Needed for |
|---|---|---|---|
| `accepted_2007_to_2018Q4.csv` (1.6 GB) | `data/raw/` | [Kaggle: All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (CC0, unzip the archive) | Training, reports, everything |
| `LC_loans_granting_model_dataset.csv` (167 MB) | `data/raw/` | [Zenodo record 11295916](https://zenodo.org/records/11295916) | Notebook 01 only |
| `LCDataDictionary.xlsx` | `data/external/` | [LendingClub](https://resources.lendingclub.com/LCDataDictionary.xlsx) ([mirror](https://figshare.com/articles/dataset/Lending_club_dataset_description/20016077)) | Rebuilding `data_dictionary.csv` only |

The selected model was trained on the file with SHA-256
`3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a` (1,675,133,810 bytes). Check
your copy with `sha256sum data/raw/accepted_2007_to_2018Q4.csv`. Rebuilding a run's data refuses
to proceed if the file differs.

### Data sources and licenses

This repository contains no loan-level data, only code, aggregate results and a model trained on the
data. Download the data from its sources under their own terms:

- **LendingClub accepted loans 2007–2018Q4**, published on Kaggle by Nathan George and labelled
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) by the uploader. The data was
  originally released by LendingClub, whose own terms may also apply.
- **Lending Club loan dataset for granting models** by M. J. Ariza-Garzón, M. Sanz-Guerrero and
  J. Arroyo Gallardo, [Zenodo record 11295916](https://zenodo.org/records/11295916), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is used here only for exploration
  (notebook 01) and was not modified or redistributed.
- **Column descriptions** in `data_dictionary.csv` are taken from LendingClub's
  [`LCDataDictionary.xlsx`](https://resources.lendingclub.com/LCDataDictionary.xlsx), © LendingClub.

This is an independent project, not affiliated with or endorsed by LendingClub.

> **Not for credit decisions.** This model is an educational and portfolio project. Using a credit
> model for real lending decisions is regulated (in the US, for example, ECOA / Regulation B and FCRA,
> including adverse-action notices and fair-lending and disparate-impact testing). This model has not
> been validated for that; in particular, it uses the borrower's state (`addr_state`), which would
> need a fair-lending review.

## Setup

Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
make dev          # package in editable mode + API (fastapi, uvicorn) + dev tools
```

## Usage

All commands run from the repository root. `CONFIG=...` switches the config (default
`configs/config.yaml`).

| Command | What it does |
|---|---|
| `make train` | Train on the `full` feature set with the config's params: 5-fold CV, final fit, test + OOT evaluation, logged to MLflow |
| `make train-baseline` | The same on the `baseline` (15-column equivalent) set, for comparison |
| `make tune` | Optuna search on train, then a final run with the selected params |
| `make train-regularized` | Final run with the most regularized trial (within 1 CV std of the best) of the latest tuning run |
| `make report` | HTML model report (importance, SHAP, PDP, calibration, KS threshold) into `reports/` and the run |
| `make register` | Register `xgb_full_regularized` in the MLflow Model Registry with alias `champion` |
| `make export-model` | Export the champion to `models/champion/` (`model.ubj` + `model_card.json`) for the API |
| `make serve` | Run the API at <http://localhost:8000> (interactive docs at `/docs`) |
| `make docker-build` / `make docker-run` | Build and run the API image |
| `make funnel` | Row and column funnel from raw file to modelling data, into `reports/` |
| `make data-dictionary` | Rebuild `data_dictionary.csv` from `LCDataDictionary.xlsx` |
| `make mlflow-ui` | Browse runs at <http://localhost:5000> |
| `make test` / `make lint` | Unit tests / ruff |

**Reproducing the selected model from scratch:**

```bash
make tune                                   # Optuna search, logged as run optuna_full
python scripts/train_regularized.py --tuning-run-id <optuna_full run id>
make report && make register
```

Pass `--tuning-run-id` explicitly; without it the script uses the latest tuning run of any feature
set. The original run was stopped after 6 of 40 trials, so a complete search may select different
parameters.

**Exporting a run's exact data** (rebuilt from the raw file and checked against the run's digests):

```bash
python scripts/export_run_data.py --run-name xgb_full_regularized --split oot
# -> data/processed/xgb_full_regularized_oot.parquet
```

## API

A FastAPI service scores loan applications with the exported model bundle. It needs neither MLflow
nor the training data at runtime.

```bash
make export-model     # once, after make register: writes models/champion/
make serve            # or: make docker-build && make docker-run
```

| Endpoint | |
|---|---|
| `POST /predict` | One application → probability of default, approve / decline, warnings |
| `POST /predict/batch` | `{"applications": [...]}`, up to 1,000 per call |
| `GET /model` | Model card: run, features, category levels, threshold, test / OOT metrics |
| `GET /health` | Liveness and the loaded model |
| `GET /docs` | Interactive OpenAPI docs, with a full example request |

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @request.json
```
```json
{"probability_of_default": 0.0782, "decision": "approve", "threshold": 0.1565,
 "warnings": [], "model": "xgb_full_regularized (c4193833)"}
```

**Input.** Fields use LendingClub's column names (descriptions in `data_dictionary.csv`). A few use
friendlier formats:

- `term` is `36` or `60`;
- `emp_length` is years from 0 to 10, where 10 means 10+;
- `earliest_cr_line` and `issue_date` are ISO dates.

**Required vs optional.** A field is optional only if it was sometimes missing in training, where
missing means something: the event never happened (the `mths_since_*` fields), the borrower has no
such account, or employment details weren't given. The other 36 credit-bureau fields were always
present in training and are required. Leaving them out would make the model read every applicant as
unusual: in a test on 20,000 OOT loans, omitting them raised the mean PD from 14% to 32%.

**Decisions.** `decline` when the PD is at least 0.1565, the KS-optimal cut-off from training. A
category the model never saw (e.g. `addr_state: "PR"`) is scored as missing and reported in
`warnings`.

**Verified:** the API reproduces the training pipeline's scores exactly, with 0.0 difference on all
162,236 OOT loans through the inference pipeline and on 5,000 of them through HTTP.

### Scoring in Python

```python
import mlflow
from loan_approval.config import load_config
from loan_approval.data.run_data import find_run_id, load_run_dataset
from loan_approval.pipelines.training_pipeline import setup_mlflow

setup_mlflow(load_config())
model = mlflow.xgboost.load_model("models:/loan-default-xgboost@champion")

oot = load_run_dataset(find_run_id("xgb_full_regularized"), "oot")
pd_scores = model.predict_proba(oot.X)[:, 1]   # probability of default
declined = pd_scores >= 0.1565                 # KS cut-off chosen on train
```

To score raw applications without MLflow, use the bundle and the inference pipeline:

```python
from loan_approval.models.bundle import load_bundle
from loan_approval.pipelines.inference_pipeline import predict

scores = predict(raw_applications_df, load_bundle("models/champion"))  # raw LendingClub columns
```

## Project structure

```
├── ARCHITECTURE.md        # how the model was built and why
├── configs/
│   ├── config.yaml                 # population, split, model params, search space, feature sets
│   └── config_kfold_no_state.yaml  # experiment: k-fold only, without addr_state
├── data/                  # git-ignored contents: raw/, external/, processed/
├── data_dictionary.csv    # all 151 raw columns: description, group, usable or leakage
├── docker/Dockerfile      # API image
├── models/champion/       # exported model bundle served by the API
├── notebooks/             # 01: 15-column file EDA; 02: original file, leakage, snapshot bias
├── reports/               # model report and data funnel (HTML + CSV)
├── scripts/               # CLI entry points (one per Makefile target)
├── src/loan_approval/
│   ├── data/              # load, clean, split, versioning, run-data rebuild, funnel
│   ├── features/          # feature sets and derived features
│   ├── api/               # FastAPI app and request / response schemas
│   ├── models/            # XGBoost CV training, Optuna tuning, model bundle export / load
│   ├── evaluation/        # metrics, importance, SHAP / PDP / calibration
│   ├── pipelines/         # training + tuning, inference, report pipelines
│   ├── utils/             # logging
│   └── config.py
└── tests/unit/
```

## Experiment tracking

Runs go to the MLflow experiment `loan-default-xgboost`, stored locally in `mlflow.db` and
`mlruns/` (both git-ignored, so a fresh clone starts empty). Each run logs its parameters,
CV/train/test/OOT metrics, config, CV folds, out-of-fold predictions, feature importance and the
model. It also records the data version: the raw file's SHA-256, per-split content digests and
`split_ids.parquet`.

Set `LOG_LEVEL` (default `INFO`) to change log verbosity.

## Reports

- [Model report: xgb_full_regularized](reports/model_report_xgb_full_regularized.html). Download it
  and open it in a browser; GitHub shows HTML as source.
- [Data funnel](reports/data_funnel.html), with tables in
  [rows](reports/data_funnel_rows.csv) and [columns](reports/data_funnel_columns.csv).

## License

The code is released under the [MIT License](LICENSE). The data is not part of this repository and
keeps its own licenses (see [Data sources and licenses](#data-sources-and-licenses)).

## Links

- Data: [Kaggle LendingClub accepted loans](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
  [Zenodo granting-model dataset](https://zenodo.org/records/11295916),
  [LendingClub data dictionary](https://resources.lendingclub.com/LCDataDictionary.xlsx)
- Tools: [FastAPI](https://fastapi.tiangolo.com/),
  [XGBoost categorical support](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html),
  [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/),
  [Optuna](https://optuna.readthedocs.io/)
