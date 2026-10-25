"""FastAPI service that scores loan applications with the exported model bundle.

Run: uvicorn loan_approval.api.app:app (bundle directory from MODEL_DIR, default models/champion).
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from loan_approval.api.schemas import (
    Application,
    BatchRequest,
    BatchResponse,
    Prediction,
    PredictResponse,
)
from loan_approval.config import PROJECT_ROOT
from loan_approval.models.bundle import ModelBundle, load_bundle
from loan_approval.pipelines.inference_pipeline import predict
from loan_approval.utils.logger import get_logger

log = get_logger(__name__)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "champion"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
    app.state.bundle = load_bundle(model_dir)
    log.info("Loaded model %s from %s", app.state.bundle.card["run_name"], model_dir)
    yield


app = FastAPI(title="Loan default model", version="0.1.0", lifespan=lifespan,
              description="Probability that a LendingClub loan application is charged off, with an "
                          "approve / decline decision at the model's KS-optimal threshold.")


def _bundle(request: Request) -> ModelBundle:
    return request.app.state.bundle


def _model_name(bundle: ModelBundle) -> str:
    return f"{bundle.card['run_name']} ({bundle.card['run_id'][:8]})"


def _score(applications: list[Application], bundle: ModelBundle) -> list[Prediction]:
    raw = pd.DataFrame([a.to_raw() for a in applications])
    try:
        scored = predict(raw, bundle)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return [Prediction(probability_of_default=row.probability_of_default, decision=row.decision,
                       threshold=bundle.threshold, warnings=row.warnings)
            for row in scored.itertuples()]


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "model": _model_name(_bundle(request))}


@app.get("/model")
def model_card(request: Request) -> dict[str, Any]:
    """The model card: lineage, features, category levels, decision threshold and test / OOT metrics."""
    return _bundle(request).card


@app.post("/predict")
def predict_one(application: Application, request: Request) -> PredictResponse:
    bundle = _bundle(request)
    prediction = _score([application], bundle)[0]
    return PredictResponse(model=_model_name(bundle), **prediction.model_dump())


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    bundle = _bundle(request)
    return BatchResponse(model=_model_name(bundle), predictions=_score(batch.applications, bundle))
