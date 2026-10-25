.PHONY: install dev lint test data-dictionary funnel train train-baseline tune train-regularized report register mlflow-ui

CONFIG ?= configs/config.yaml

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts

test:
	pytest

data-dictionary:
	python scripts/build_data_dictionary.py

funnel:
	python scripts/data_funnel.py --config $(CONFIG)

train:
	python scripts/train.py --config $(CONFIG) --feature-set full

train-baseline:
	python scripts/train.py --config $(CONFIG) --feature-set baseline

tune:
	python scripts/tune.py --config $(CONFIG) --feature-set full

train-regularized:
	python scripts/train_regularized.py --config $(CONFIG)

report:
	python scripts/model_report.py --config $(CONFIG) --run-name xgb_full_regularized

register:
	python scripts/register_model.py --config $(CONFIG) --run-name xgb_full_regularized

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
